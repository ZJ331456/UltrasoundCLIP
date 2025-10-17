"""
训练器模块
负责模型的训练和验证逻辑
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import gc
import math
from typing import Dict, Any, Optional
from loss import LossFactory

# lora相关
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# EMA相关
try:
    from torch_ema import ExponentialMovingAverage
    EMA_AVAILABLE = True
except ImportError:
    print("Warning: torch_ema not available. EMA functionality will be disabled.")
    EMA_AVAILABLE = False

class Trainer:
    """CLIP模型训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict[str, Any],
                 device: torch.device):
        """
        初始化训练器
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 训练配置
            device: 计算设备
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        #lora配置以及应用
        self.lora_config = self.config['lora']
        self.use_lora = self.lora_config['enabled']

        if self.use_lora:
            self.model = self._apply_lora_to_model()
            print("  - lora微调：应用")
        else:
            print("  - lora微调：禁用")

        # 创建损失函数
        self.criterion = self._create_loss_function()
        # 多视角损失权重（fused/global/local）
        mv_cfg = self.config.get('multi_view', {})
        self.mv_weight_fused = float(mv_cfg.get('weight_fused', 1.0))
        self.mv_weight_global = float(mv_cfg.get('weight_global', 0.5))
        self.mv_weight_local = float(mv_cfg.get('weight_local', 0.5))
        
        # 创建辅助损失函数（mask监督和attention正则化）
        self._setup_auxiliary_losses()
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 混合精度训练支持
        self.use_mixed_precision = self.config.get('mixed_precision', False)
        if self.use_mixed_precision:
            try:
                # 尝试新的API
                from torch.amp import GradScaler
                self.scaler = GradScaler('cuda')
                print("  - 混合精度训练: 启用 (使用torch.amp)")
            except ImportError:
                # 回退到旧的API
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
                print("  - 混合精度训练: 启用 (使用torch.cuda.amp)")
        else:
            self.scaler = None
            print("  - 混合精度训练: 禁用")
        
        # EMA支持
        self.use_ema = self.config.get('use_ema', False)
        self.ema = None
        if self.use_ema and EMA_AVAILABLE:
            self.ema_config = self.config.get('ema_config', {})
            self._setup_ema()
            print("  - EMA: 启用")
        else:
            print("  - EMA: 禁用")
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # 混合精度训练状态跟踪
        self._unscaled_this_step = False
        
        # 第三轮新增：正则化配置
        self.regularization_config = self.config.get('regularization', {})
        self.use_enhanced_regularization = self.regularization_config.get('dropout_enhanced', False)
        if self.use_enhanced_regularization:
            print(f"  - 增强正则化: 启用")
            print(f"    - 随机深度: {self.regularization_config.get('stochastic_depth', 0.0)}")
            print(f"    - 特征噪声: {self.regularization_config.get('feature_noise_std', 0.0)}")
        
        print(f"训练器初始化完成:")
        print(f"  - 损失函数: {type(self.criterion).__name__}")
        print(f"  - 优化器: {type(self.optimizer).__name__}")
        print(f"  - 学习率调度器: {type(self.scheduler).__name__}")
    
    def _setup_auxiliary_losses(self):
        """设置辅助损失函数（mask监督和attention正则化）"""
        loss_config = self.config.get('loss', {})
        
        # Mask监督损失配置
        mask_config = loss_config.get('mask_supervision', {})
        self.mask_supervision_enabled = mask_config.get('enabled', False)
        if self.mask_supervision_enabled:
            self.mask_supervision_weight = mask_config.get('weight', 0.1)
            self.mask_supervision_loss = LossFactory.create_loss(
                mask_config.get('type', 'mask_supervision'),
                bce_weight=mask_config.get('bce_weight', 0.5),
                dice_weight=mask_config.get('dice_weight', 0.5),
                pos_weight=mask_config.get('pos_weight', 2.0)
            )
            print(f"  - Mask监督损失: 启用，权重={self.mask_supervision_weight}")
        else:
            self.mask_supervision_loss = None
        
        # Attention正则化配置
        attn_reg_config = loss_config.get('attention_regularization', {})
        self.attention_regularization_enabled = attn_reg_config.get('enabled', False)
        if self.attention_regularization_enabled:
            # 熵正则化
            self.attention_entropy_loss = LossFactory.create_loss(
                attn_reg_config.get('entropy_type', 'attention_entropy'),
                entropy_weight=attn_reg_config.get('entropy_weight', 0.01),
                sparsity_weight=attn_reg_config.get('sparsity_weight', 0.01),
                target_sparsity=attn_reg_config.get('target_sparsity', 0.3)
            )
            
            # Inside/Outside对比损失
            self.inside_outside_enabled = mask_config.get('enabled', False)  # 需要mask才能使用
            if self.inside_outside_enabled:
                self.inside_outside_weight = attn_reg_config.get('inside_outside_weight', 0.05)
                self.attention_io_loss = LossFactory.create_loss(
                    attn_reg_config.get('inside_outside_type', 'attention_inside_outside'),
                    margin=attn_reg_config.get('margin', 0.3)
                )
                print(f"  - Attention正则化: 启用（熵+IO对比），IO权重={self.inside_outside_weight}")
            else:
                self.attention_io_loss = None
                print(f"  - Attention正则化: 启用（仅熵正则化）")
        else:
            self.attention_entropy_loss = None
            self.attention_io_loss = None
    
    def _apply_lora_to_model(self) -> nn.Module:
        """应用lora方式到模型上面"""
        lora_cfg = self.lora_config

        # 获取Lora配置参数
        r = int(lora_cfg['r'])
        alpha = int(lora_cfg['alpha'])
        dropout = float(lora_cfg['dropout'])
        bias = str(lora_cfg['bias'])
        task_type = getattr(TaskType, str(lora_cfg['task_type']).upper())
        
        # 根据模型的类型选择不同的target_modules
        model_type = self._detect_model_type()
        target_modules = lora_cfg['target_modules']

        if target_modules is None:
            if model_type == "qwen_vl":
                # Qwen2.5-VL 模型的默认目标模块
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif model_type == "bert":
                # BERT 模型的默认目标模块
                target_modules = ["query", "key", "value", "dense"]
            elif model_type == "roberta":
                # RoBERTa 模型的默认目标模块
                target_modules = ["query", "key", "value", "dense"]
            else:
                # 通用 transformer 目标模块
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        print(f"  - LoRA 配置: r={r}, alpha={alpha}, dropout={dropout}")
        print(f"  - 目标模块: {target_modules}")

        # 创建Lora配置
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias=bias,
            task_type=task_type,
            target_modules=target_modules,
            inference_mode=False,
        )
        # 注意不是inference_model
        # inference_mode=False: 训练模式，允许梯度计算和参数更新
        # inference_mode=True: 推理模式，冻结所有参数，不计算梯度
        # 在模型上应用lora
        peft_model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数统计一下
        peft_model.print_trainable_parameters()

        # 冻结非lora参数
        self._freeze_non_lora_parameters()

        return peft_model

    def _detect_model_type(self) -> str:
        """检测模型类型好选择合适的目标模块"""
        # 这部分我觉得不太需要，因为可能不止这一些的，还不如直接在配置文件里面写好！
        model_str = str(type(self.model)).lower()
        
        if "qwen" in model_str:
            return "qwen_vl"
        elif "bert" in model_str:
            return "bert"
        elif "roberta" in model_str:
            return "roberta"
        elif "convnext" in model_str:
            return "convnext"
        else:
            return "generic"

    def _freeze_non_lora_parameters(self):
        """冻结非lora参数,只训练Lora的参数"""
        if not self.use_lora:
            return
        
        # 获取所有的参数
        # all_params = list(self.model.parameters())
        all_params = list(self.model.named_parameters())

        # 统计参数量
        total_params = 0
        trainable_params = 0
        frozen_params = 0

        for name, param in all_params:
            total_params += param.numel()

            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()

        print(f"  - 总参数: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")
        print(f"  - 冻结参数: {frozen_params:,}")
        print(f"  - 可训练比例: {trainable_params/total_params*100:.2f}%")

    def _setup_ema(self):
        """设置EMA"""
        if not EMA_AVAILABLE:
            return
            
        decay = self.ema_config.get('decay', 0.9999)
        self.ema_update_after_step = self.ema_config.get('update_after_step', 100)
        self.ema_warmup_steps = self.ema_config.get('warmup_steps', 1000)
        self.ema_decay_warmup = self.ema_config.get('decay_warmup', True)
        
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=decay)
        print(f"    - EMA decay: {decay}")
        print(f"    - EMA update after step: {self.ema_update_after_step}")
        print(f"    - EMA warmup steps: {self.ema_warmup_steps}")

    def _update_ema(self):
        """更新EMA"""
        if not self.use_ema or self.ema is None:
            return
        
        # 只在指定步数后开始更新EMA
        if self.global_step >= self.ema_update_after_step:
            # 如果启用EMA decay warmup
            if self.ema_decay_warmup and self.global_step < self.ema_warmup_steps:
                # 在warmup期间逐渐增加decay
                progress = (self.global_step - self.ema_update_after_step) / (self.ema_warmup_steps - self.ema_update_after_step)
                progress = min(1.0, max(0.0, progress))
                # 从0.9逐渐增加到目标decay
                current_decay = 0.9 + (self.ema.decay - 0.9) * progress
                self.ema.decay = current_decay
            
            self.ema.update()

    def _get_ema_model_for_validation(self):
        """获取用于验证的EMA模型"""
        if not self.use_ema or self.ema is None:
            return self.model
        
        use_ema_for_validation = self.ema_config.get('use_ema_for_validation', True)
        if use_ema_for_validation and self.global_step >= self.ema_update_after_step:
            # 临时应用EMA权重
            self.ema.store()
            self.ema.copy_to()
            return self.model
        else:
            return self.model

    def _restore_model_from_ema(self):
        """从EMA恢复原始模型权重"""
        if not self.use_ema or self.ema is None:
            return
        
        use_ema_for_validation = self.ema_config.get('use_ema_for_validation', True)
        if use_ema_for_validation and self.global_step >= self.ema_update_after_step:
            # 恢复原始权重
            self.ema.restore()

    def _check_gradients_finite(self) -> bool:
        """检查模型梯度是否有效（非NaN/Inf）"""
        for param in self.model.parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    return False
        return True
    
    def _validate_batch_data(self, batch: Dict, batch_idx: int):
        """简化的批次数据验证，避免tensor布尔判断错误"""
        try:
            # 基本类型检查
            if not isinstance(batch, dict):
                raise ValueError(f"Batch应该是字典类型，但得到 {type(batch)}")
            
            # 检查必要键
            available_keys = list(batch.keys())
            has_images = any(key in batch for key in ['images', 'image'])
            has_texts = any(key in batch for key in ['texts', 'text', 'text_tokens'])
            
            if not has_images:
                raise ValueError(f"Batch中缺少图像数据，可用键: {available_keys}")
            if not has_texts:
                raise ValueError(f"Batch中缺少文本数据，可用键: {available_keys}")
            
            # 简化的数据验证 - 避免复杂的tensor检查
            images = batch.get('images') or batch.get('image')
            texts = batch.get('texts') or batch.get('text')
            
            # 基本非空检查
            if images is None:
                raise ValueError("图像数据为None")
            if texts is None and 'text_tokens' not in batch:
                raise ValueError("文本数据为None")
            
            # 简单的数量一致性检查（仅对明确的容器类型）
            if isinstance(images, list) and isinstance(texts, list):
                if len(images) != len(texts):
                    raise ValueError(f"图像数量 ({len(images)}) 与文本数量 ({len(texts)}) 不匹配")
                
        except Exception as e:
            print(f"Batch {batch_idx} 数据验证失败: {e}")
            raise e
    
    def _create_loss_function(self) -> nn.Module:
        """创建损失函数"""
        loss_config = self.config['loss']
        
        # 检查是否使用优化的损失函数
        use_optimized_loss = loss_config.get('use_optimized_loss', False)
        
        # 根据损失函数类型，只传递支持的参数
        if loss_config['type'] == 'simplified_clip_stable' or use_optimized_loss:
            # 优化的简化稳定损失函数
            loss_fn = LossFactory.create_loss(
                loss_type='simplified_clip_stable',
                temperature=loss_config.get('temperature', 0.07),
                label_smoothing=loss_config.get('label_smoothing', 0.1),
                numerical_stability=loss_config.get('numerical_stability', {})
            )
            
            # 第三轮新增：自适应温度支持
            adaptive_temp = loss_config.get('adaptive_temperature', {})
            if adaptive_temp.get('enabled', False):
                loss_fn = self._wrap_with_adaptive_temperature(loss_fn, adaptive_temp)
                print(f"    - 启用自适应温度调节")
            
            return loss_fn
            
        elif loss_config['type'] == 'ultrasound_simple':
            # UltrasoundSimpleLoss 支持的完整参数
            return LossFactory.create_loss(
                loss_type=loss_config['type'],
                temperature=loss_config.get('temperature', 0.1),
                label_smoothing=loss_config.get('label_smoothing', 0.1),
                hard_negative_ratio=loss_config.get('hard_negative_ratio', 0.3),
                margin=loss_config.get('margin', 0.2),
                use_soft_label=loss_config.get('use_soft_label', True),
                soft_label_temp=loss_config.get('soft_label_temp', 2.0),
                soft_label_weight=loss_config.get('soft_label_weight', 0.6)
            )
        elif loss_config['type'] == 'ultrasound_advanced':
            # UltrasoundAdvancedLoss 支持的完整参数
            return LossFactory.create_loss(
                loss_type=loss_config['type'],
                temperature=loss_config.get('temperature', 0.05),
                use_adaptive_temperature=loss_config.get('use_adaptive_temperature', True),
                adaptive_temp_config=loss_config.get('adaptive_temp_config', {}),
                hard_negative_mining=loss_config.get('hard_negative_mining', {}),
                multi_scale_loss=loss_config.get('multi_scale_loss', {}),
                semantic_consistency=loss_config.get('semantic_consistency', {}),
                feature_diversity=loss_config.get('feature_diversity', {}),
                label_smoothing=loss_config.get('label_smoothing', 0.2),
                numerical_stability=loss_config.get('numerical_stability', {})
            )
        else:
            # 其他损失函数使用完整参数集
            return LossFactory.create_loss(
                loss_type=loss_config['type'],
                temperature=loss_config['temperature'],
                margin=loss_config.get('margin', 0.2),
                hard_ratio=loss_config.get('hard_ratio', 0.3),
                focal_gamma=loss_config.get('focal_gamma', 2.0),
                similarity_threshold=loss_config.get('similarity_threshold', 0.8),
                contrastive_scale=loss_config.get('contrastive_scale', 1.0),
                dynamic_temp_target=loss_config.get('dynamic_temp_target', 0.6),
                dynamic_temp_gain=loss_config.get('dynamic_temp_gain', 2.0),
                semi_hard_window=loss_config.get('semi_hard_window', 0.05)
            )
    
    def _wrap_with_adaptive_temperature(self, loss_fn, adaptive_config):
        """包装损失函数以支持自适应温度"""
        class AdaptiveTemperatureLossWrapper(nn.Module):
            def __init__(self, base_loss, min_temp, max_temp, adaptation_rate):
                super().__init__()
                self.base_loss = base_loss
                self.min_temp = min_temp
                self.max_temp = max_temp
                self.adaptation_rate = adaptation_rate
                self.current_temp = base_loss.temperature
                
            def forward(self, image_features, text_features):
                # 计算特征相似度统计
                with torch.no_grad():
                    sim_matrix = torch.matmul(image_features, text_features.T)
                    pos_sim = sim_matrix.diag().mean()
                    neg_sim = sim_matrix[~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)].mean()
                    sim_gap = pos_sim - neg_sim
                    
                    # 根据相似度差距调整温度
                    if sim_gap < 0.1:  # 正负样本差距太小，降低温度
                        target_temp = self.min_temp
                    elif sim_gap > 0.5:  # 差距太大，提高温度
                        target_temp = self.max_temp
                    else:
                        # 线性插值
                        ratio = (sim_gap - 0.1) / 0.4
                        target_temp = self.min_temp + (self.max_temp - self.min_temp) * ratio
                    
                    # 平滑更新温度
                    self.current_temp = (1 - self.adaptation_rate) * self.current_temp + self.adaptation_rate * target_temp
                    self.current_temp = max(self.min_temp, min(self.max_temp, self.current_temp))
                
                # 更新基础损失函数的温度
                original_temp = self.base_loss.temperature
                self.base_loss.temperature = self.current_temp
                
                # 计算损失
                loss = self.base_loss(image_features, text_features)
                
                # 恢复原始温度
                self.base_loss.temperature = original_temp
                
                return loss
        
        return AdaptiveTemperatureLossWrapper(
            loss_fn,
            adaptive_config.get('min_temp', 0.04),
            adaptive_config.get('max_temp', 0.1),
            adaptive_config.get('adaptation_rate', 0.01)
        )
    
    def _apply_regularization(self, image_features, text_features):
        """应用训练时正则化技术"""
        # 特征噪声注入
        feature_noise_std = self.regularization_config.get('feature_noise_std', 0.0)
        if feature_noise_std > 0:
            image_noise = torch.randn_like(image_features) * feature_noise_std
            text_noise = torch.randn_like(text_features) * feature_noise_std
            image_features = image_features + image_noise
            text_features = text_features + text_noise
        
        # 特征Dropout（随机置零部分特征维度）
        feature_dropout = self.regularization_config.get('feature_dropout', 0.0)
        if feature_dropout > 0:
            if torch.rand(1).item() < feature_dropout:
                # 随机选择一些特征维度置零
                img_mask = torch.rand(image_features.shape[-1], device=image_features.device) > feature_dropout
                txt_mask = torch.rand(text_features.shape[-1], device=text_features.device) > feature_dropout
                image_features = image_features * img_mask.float()
                text_features = text_features * txt_mask.float()
        
        return image_features, text_features
    
    def _compute_gradient_penalty(self, image_features, text_features):
        """计算梯度惩罚正则化项"""
        # 简单的梯度范数惩罚
        img_grad_norm = torch.norm(torch.autograd.grad(
            outputs=image_features.sum(),
            inputs=image_features,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0], p=2, dim=-1).mean()
        
        txt_grad_norm = torch.norm(torch.autograd.grad(
            outputs=text_features.sum(),
            inputs=text_features,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0], p=2, dim=-1).mean()
        
        return (img_grad_norm + txt_grad_norm) / 2
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器，支持分层学习率还有新添加的lora优化"""
        opt_config = self.config['optimizer']
        
        # 检查是否使用优化的学习率策略
        use_optimized_lr_strategy = self.config.get('use_optimized_lr_strategy', False)
        
        if opt_config['type'] == 'adamw_optimized' or use_optimized_lr_strategy:
            return self._create_optimized_optimizer(opt_config)
        elif opt_config['type'] == 'adamw':
            return self._create_standard_optimizer(opt_config)
        else:
            raise ValueError(f"不支持的优化器类型: {opt_config['type']}")

    def _create_standard_optimizer(self, opt_config):
        """创建标准优化器（原有逻辑）"""
        # 分层学习率：视觉编码器较小学习率，投影层较大学习率
        param_groups = []
        base_lr = opt_config['lr']

        def is_norm_or_bias(n: str):
            return n.endswith('bias') or 'norm' in n.lower() or 'layernorm' in n.lower()

        lora_params, lora_norm_bias = [], []
        main_weights, main_norm_bias = [], []
        
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                # 跳过冻结的参数
                continue

            if self.use_lora and 'lora' in name.lower():
                (lora_norm_bias if is_norm_or_bias(name) else lora_params).append(p)
            else:
                # 其他参数归入文本参数组
                (main_norm_bias if is_norm_or_bias(name) else main_weights).append(p)
        
        # lora组:高LR\无衰减
        if lora_params:
            param_groups.append({'params': lora_params, 'lr': base_lr * 3.0})
        if lora_norm_bias:
            param_groups.append({'params': lora_norm_bias, 'lr': base_lr * 3.0, 'weight_decay': 0.0})
        
        # 主模型组:Norm/Bias无衰减,其他正常衰减
        if main_norm_bias:
            param_groups.append({'params': main_norm_bias, 'lr': base_lr * 0.1, 'weight_decay': 0.0})
        if main_weights:
            param_groups.append({'params': main_weights, 'lr': base_lr * 0.1, 'weight_decay': opt_config['weight_decay']})
        
        # 确保至少有一个参数组
        if not param_groups:
            raise ValueError("没有可训练的参数组")
        
        return optim.AdamW(
            param_groups,
            betas=opt_config['betas'],
            eps=opt_config['eps'],
            amsgrad=opt_config.get('amsgrad', False)
        )

    def _create_optimized_optimizer(self, opt_config):
        """创建优化的优化器（对照组实验专用）"""
        base_lr = opt_config['lr']
        use_layerwise_decay = opt_config.get('layerwise_lr_decay', False)
        layer_decay_rate = opt_config.get('layer_decay_rate', 0.95)
        exclude_bias_norm = opt_config.get('exclude_bias_norm_from_decay', True)
        
        # 第三轮新增：差分学习率支持
        differential_lr = opt_config.get('differential_lr', {})
        use_differential_lr = differential_lr.get('enabled', False)
        
        def is_norm_or_bias(n: str):
            return n.endswith('bias') or 'norm' in n.lower() or 'layernorm' in n.lower()

        def get_module_type(name: str):
            """确定参数所属的模块类型"""
            if 'vision' in name or 'image' in name or 'visual' in name or 'convnext' in name:
                return 'vision'
            elif 'text' in name or 'roberta' in name or 'bert' in name:
                return 'text'
            elif 'projection' in name or 'head' in name or 'classifier' in name:
                return 'projection'
            else:
                return 'other'

        param_groups = []
        
        if use_differential_lr:
            # 差分学习率策略
            vision_lr_scale = differential_lr.get('vision_lr_scale', 0.5)
            text_lr_scale = differential_lr.get('text_lr_scale', 1.0)
            projection_lr_scale = differential_lr.get('projection_lr_scale', 2.0)
            
            module_params = {
                'vision': {'weights': [], 'bias_norm': []},
                'text': {'weights': [], 'bias_norm': []},
                'projection': {'weights': [], 'bias_norm': []},
                'other': {'weights': [], 'bias_norm': []}
            }
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                module_type = get_module_type(name)
                if exclude_bias_norm and is_norm_or_bias(name):
                    module_params[module_type]['bias_norm'].append(param)
                else:
                    module_params[module_type]['weights'].append(param)
            
            # 为每个模块创建参数组
            lr_scales = {
                'vision': vision_lr_scale,
                'text': text_lr_scale,
                'projection': projection_lr_scale,
                'other': 1.0
            }
            
            for module_type, params in module_params.items():
                module_lr = base_lr * lr_scales[module_type]
                
                if params['weights']:
                    param_groups.append({
                        'params': params['weights'],
                        'lr': module_lr,
                        'weight_decay': opt_config['weight_decay']
                    })
                
                if params['bias_norm']:
                    param_groups.append({
                        'params': params['bias_norm'],
                        'lr': module_lr,
                        'weight_decay': 0.0
                    })
            
            print(f"    - 差分学习率: 视觉{vision_lr_scale}x, 文本{text_lr_scale}x, 投影{projection_lr_scale}x")
            
        elif use_layerwise_decay:
            # 分层学习率衰减策略
            layer_params = {}
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # 确定层数
                layer_id = self._get_layer_id(name)
                if layer_id not in layer_params:
                    layer_params[layer_id] = {'weights': [], 'bias_norm': []}
                
                if exclude_bias_norm and is_norm_or_bias(name):
                    layer_params[layer_id]['bias_norm'].append(param)
                else:
                    layer_params[layer_id]['weights'].append(param)
            
            # 为每一层创建参数组
            max_layer_id = max(layer_params.keys()) if layer_params else 0
            for layer_id, params in layer_params.items():
                # 计算该层的学习率
                layer_lr = base_lr * (layer_decay_rate ** (max_layer_id - layer_id))
                
                if params['weights']:
                    param_groups.append({
                        'params': params['weights'],
                        'lr': layer_lr,
                        'weight_decay': opt_config['weight_decay']
                    })
                
                if params['bias_norm']:
                    param_groups.append({
                        'params': params['bias_norm'],
                        'lr': layer_lr,
                        'weight_decay': 0.0
                    })
        else:
            # 简单的bias/norm排除策略
            weights, bias_norm = [], []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                if exclude_bias_norm and is_norm_or_bias(name):
                    bias_norm.append(param)
                else:
                    weights.append(param)
            
            if weights:
                param_groups.append({
                    'params': weights,
                    'lr': base_lr,
                    'weight_decay': opt_config['weight_decay']
                })
            
            if bias_norm:
                param_groups.append({
                    'params': bias_norm,
                    'lr': base_lr,
                    'weight_decay': 0.0
                })
        
        if not param_groups:
            raise ValueError("没有可训练的参数组")
        
        print(f"    - 使用优化的优化器策略")
        print(f"    - 差分学习率: {use_differential_lr}")
        print(f"    - 分层学习率衰减: {use_layerwise_decay}")
        if use_layerwise_decay:
            print(f"    - 层衰减率: {layer_decay_rate}")
        print(f"    - Bias/Norm不衰减: {exclude_bias_norm}")
        
        return optim.AdamW(
            param_groups,
            betas=opt_config['betas'],
            eps=opt_config['eps'],
            amsgrad=opt_config.get('amsgrad', False)
        )

    def _get_layer_id(self, name: str) -> int:
        """获取参数所在的层ID"""
        # 简单的层ID提取逻辑
        if 'encoder' in name:
            # 寻找layer数字
            import re
            matches = re.findall(r'layer\.(\d+)', name)
            if matches:
                return int(matches[0])
            matches = re.findall(r'layers\.(\d+)', name)
            if matches:
                return int(matches[0])
        
        # 默认层ID
        if 'embedding' in name:
            return 0
        elif 'encoder' in name:
            return 5  # 中间层
        elif 'projection' in name or 'head' in name:
            return 10  # 输出层
        else:
            return 3  # 默认层
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """创建学习率调度器"""
        sched_config = self.config['scheduler']
        sched_type = sched_config['type']
        
        if sched_type == 'cosine_with_warmup':
            # 自定义余弦预热调度器
            return self._create_cosine_warmup_scheduler(sched_config)
        elif sched_type == 'cosine_with_linear_warmup':
            # 优化的线性warmup余弦调度器（对照组实验专用）
            return self._create_optimized_cosine_scheduler(sched_config)
        elif sched_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['max_epochs'],
                eta_min=sched_config['eta_min']
            )
        elif sched_type == 'cosine_with_restarts':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=sched_config['T_0'],
                T_mult=sched_config['T_mult'],
                eta_min=sched_config['eta_min']
            )
        elif sched_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        else:
            return optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0)
    
    def _create_cosine_warmup_scheduler(self, sched_config):
        """创建余弦预热调度器"""
        class CosineWarmupScheduler:
            def __init__(self, optimizer, warmup_steps, max_steps, eta_min=0, warmup_init_lr=None):
                self.optimizer = optimizer
                self.warmup_steps = warmup_steps
                self.max_steps = max_steps
                self.eta_min = eta_min
                self.base_lr = optimizer.param_groups[0]['lr']
                self.warmup_init_lr = warmup_init_lr if warmup_init_lr is not None else self.base_lr * 0.1
                self.step_count = 0
            
            def step(self):
                self.step_count += 1
                if self.step_count <= self.warmup_steps:
                    # 预热阶段：从初始学习率线性增长到基础学习率
                    lr = self.warmup_init_lr + (self.base_lr - self.warmup_init_lr) * (self.step_count / self.warmup_steps)
                else:
                    # 余弦衰减阶段
                    progress = (self.step_count - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                    # lr = self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (1 + torch.cos(torch.pi * progress))
                    lr = self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (1 + torch.cos(torch.pi * torch.tensor(progress)))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            def state_dict(self):
                return {'step_count': self.step_count}
            
            def load_state_dict(self, state_dict):
                self.step_count = state_dict['step_count']
        
        return CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=sched_config['warmup_steps'],
            max_steps=sched_config['max_steps'],
            eta_min=sched_config['eta_min'],
            warmup_init_lr=sched_config.get('warmup_init_lr', None)
        )

    def _create_optimized_cosine_scheduler(self, sched_config):
        """创建优化的余弦调度器（对照组实验专用）"""
        class OptimizedCosineScheduler:
            def __init__(self, optimizer, warmup_steps, max_steps, eta_min=0, 
                        warmup_init_lr=None, restart_on_plateau=False, 
                        plateau_patience=5, cooldown_factor=0.5):
                self.optimizer = optimizer
                self.warmup_steps = warmup_steps
                self.max_steps = max_steps
                self.eta_min = eta_min
                self.base_lrs = [group['lr'] for group in optimizer.param_groups]
                self.warmup_init_lr = warmup_init_lr if warmup_init_lr is not None else min(self.base_lrs) * 0.1
                self.step_count = 0
                
                # 高级功能
                self.restart_on_plateau = restart_on_plateau
                self.plateau_patience = plateau_patience
                self.cooldown_factor = cooldown_factor
                self.plateau_counter = 0
                self.best_metric = None
                self.last_restart_step = 0
            
            def step(self, metrics=None):
                self.step_count += 1
                
                # 检查是否需要重启
                if self.restart_on_plateau and metrics is not None:
                    self._check_plateau_restart(metrics)
                
                # 计算当前步数（考虑重启）
                current_step = self.step_count - self.last_restart_step
                current_max_steps = self.max_steps - self.last_restart_step
                
                if current_step <= self.warmup_steps:
                    # 线性warmup阶段
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        base_lr = self.base_lrs[i]
                        lr = self.warmup_init_lr + (base_lr - self.warmup_init_lr) * (current_step / self.warmup_steps)
                        param_group['lr'] = lr
                else:
                    # 余弦衰减阶段
                    progress = (current_step - self.warmup_steps) / (current_max_steps - self.warmup_steps)
                    progress = min(1.0, progress)  # 确保不超过1
                    
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        base_lr = self.base_lrs[i]
                        lr = self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                        param_group['lr'] = lr
            
            def _check_plateau_restart(self, metrics):
                """检查是否需要在平台期重启"""
                current_metric = metrics.get('val_r1', metrics.get('val_loss', None))
                if current_metric is None:
                    return
                
                if self.best_metric is None:
                    self.best_metric = current_metric
                    return
                
                # 检查是否有改进（假设越大越好，如R@1）
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.plateau_counter = 0
                else:
                    self.plateau_counter += 1
                
                # 如果达到平台期，执行重启
                if self.plateau_counter >= self.plateau_patience:
                    print(f"Learning rate restart at step {self.step_count} due to plateau")
                    # 降低基础学习率
                    for i in range(len(self.base_lrs)):
                        self.base_lrs[i] *= self.cooldown_factor
                    
                    # 重置计数器
                    self.last_restart_step = self.step_count
                    self.plateau_counter = 0
                    self.best_metric = current_metric
            
            def state_dict(self):
                return {
                    'step_count': self.step_count,
                    'base_lrs': self.base_lrs,
                    'best_metric': self.best_metric,
                    'plateau_counter': self.plateau_counter,
                    'last_restart_step': self.last_restart_step
                }
            
            def load_state_dict(self, state_dict):
                self.step_count = state_dict['step_count']
                self.base_lrs = state_dict['base_lrs']
                self.best_metric = state_dict.get('best_metric')
                self.plateau_counter = state_dict.get('plateau_counter', 0)
                self.last_restart_step = state_dict.get('last_restart_step', 0)
        
        print(f"    - 使用优化的余弦调度器")
        print(f"    - 自适应重启: {sched_config.get('restart_on_plateau', False)}")
        
        return OptimizedCosineScheduler(
            self.optimizer,
            warmup_steps=sched_config['warmup_steps'],
            max_steps=sched_config['max_steps'],
            eta_min=sched_config['eta_min'],
            warmup_init_lr=sched_config.get('warmup_init_lr', None),
            restart_on_plateau=sched_config.get('restart_on_plateau', False),
            plateau_patience=sched_config.get('plateau_patience', 5),
            cooldown_factor=sched_config.get('cooldown_factor', 0.5)
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch，支持梯度累计 - 增强错误处理和性能监控"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        successful_batches = 0  # 记录成功处理的batch数量
        avg_loss = 0.0  # 初始化avg_loss避免引用错误
        skipped_batches = 0  # 跳过的batch数量
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} Training")
        
        # 梯度累计参数
        grad_accum_steps = self.config['gradient_accumulation_steps']
        assert grad_accum_steps > 0, "梯度累计步数必须大于0"
        
        # 梯度裁剪-->防止梯度爆炸
        grad_clip_val = self.config['gradient_clip_val']
        
        # 性能监控
        import time
        epoch_start_time = time.time()
        slow_batch_threshold = 30.0  # 单个batch超过30秒认为是异常慢
        fast_batch_count = 0  # 正常速度batch计数
        
        # 早期失败检测
        max_consecutive_failures = 50  # 最多连续失败50个batch
        consecutive_failures = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            batch_start_time = time.time()
            
            # 早期失败检测
            if consecutive_failures >= max_consecutive_failures:
                print(f"ERROR: 连续失败 {consecutive_failures} 个batch，提前结束epoch")
                break
            
            # 添加调试信息（启用debug模式）
            debug_mode = False  # 训练稳定后屏蔽调试信息
            
            try:
                # 简化的数据验证
                if not isinstance(batch, dict):
                    raise ValueError(f"Batch应该是字典，但得到 {type(batch)}")
                if 'images' not in batch and 'image' not in batch:
                    raise ValueError(f"Batch缺少图像数据，可用键: {list(batch.keys())}")
                
                # 数据移到设备
                # 支持不同的键名格式
                if 'images' in batch:
                    images = batch['images']
                elif 'image' in batch:
                    images = batch['image']
                else:
                    raise KeyError(f"Batch中找不到图像数据，可用键: {list(batch.keys())}")
                
                # 验证图像tensor的一致性
                # 主要是保证数据一致性，然后筛选图像尺寸
                if isinstance(images, torch.Tensor):
                    # 检查batch中所有图像的尺寸是否一致
                    if len(images.shape) == 4:  # [B, C, H, W]
                        batch_size, channels, height, width = images.shape
                        if channels not in [1, 3, 4]:  # 支持的通道数
                            print(f"警告: 异常的通道数 {channels}，跳过batch {batch_idx}")
                            skipped_batches += 1
                            consecutive_failures += 1
                            continue
                        if height != width or height < 32:  # 基本尺寸验证
                            print(f"警告: 异常的图像尺寸 {height}x{width}，跳过batch {batch_idx}")
                            skipped_batches += 1
                            consecutive_failures += 1
                            continue
                    elif isinstance(images, list):
                        # 如果是图像列表，检查每个图像
                        skip_this_batch = False
                        for i, img in enumerate(images):
                            if isinstance(img, torch.Tensor) and len(img.shape) != 3:
                                print(f"警告: 图像 {i} 尺寸异常 {img.shape}，跳过batch {batch_idx}")
                                skip_this_batch = True
                                break
                        if skip_this_batch:
                            skipped_batches += 1
                            consecutive_failures += 1
                            continue
                    
                    images = images.to(self.device)
                else:
                    # 如果不是tensor，可能是PIL图像列表或路径
                    pass  # 交给模型处理
                
                # 对于Qwen2.5-VL CLIP模型，使用原始文本而不是tokenized的结果---内部有
                if hasattr(self.model, 'config') and 'qwen' in str(type(self.model)).lower():
                    # 使用原始文本
                    if 'texts' in batch:
                        texts = batch['texts']  # 这是原始文本列表
                    elif 'text' in batch:
                        text_data = batch['text']
                        # 如果是字符串，转换为列表；如果已经是列表，直接使用
                        texts = [text_data] if isinstance(text_data, str) else text_data
                    else:
                        raise KeyError(f"Batch中找不到文本数据，可用键: {list(batch.keys())}")
                    text_inputs = texts
                else:
                    # 使用tokenized的结果
                    text_tokens = batch['text_tokens']
                    if isinstance(text_tokens, dict):
                        text_inputs = {k: v.to(self.device) for k, v in text_tokens.items()}
                    else:
                        text_inputs = text_tokens.to(self.device)
                
                # 前向传播（支持混合精度）
                if self.use_mixed_precision:
                    try:
                        # 尝试新的API
                        from torch.amp import autocast
                        autocast_ctx = autocast('cuda')
                    except ImportError:
                        # 回退到旧的API
                        from torch.cuda.amp import autocast
                        autocast_ctx = autocast()
                    # autocast_ctx是一个上下文管理器，用于启用混合精度训练。
                    # 它通过自动选择合适的数值精度（FP16 或 FP32），在保证计算稳定性的同时提高训练效率并减少显存占用。
                    with autocast_ctx:
                        # # 为模型前向传播添加debug信息
                        # if debug_mode:
                        # print(f"  开始前向传播...")
                        # print(f"  输入图像形状: {images.shape if hasattr(images, 'shape') else type(images)}")
                        # print(f"  输入文本类型: {type(text_inputs)}")
                    
                        # image_features, text_features = self.model(images, text_inputs, debug=debug_mode)
                        masks = batch.get('mask') if isinstance(batch, dict) else None  # 注意：数据集中使用'mask'而不是'masks'
                                attention_map = None
                            
                            if hasattr(self.model, 'core'):
                        # 如果有 core 属性，直接调用核心模型
                            if masks is not None:
                                result = self.model.core(images, text_inputs, masks=masks)
                        else:
                                result = self.model.core(images, text_inputs)
                                image_features = result['image_features']
                                text_features = result['text_features']
                                image_global = result.get('image_global', None)
                                image_local = result.get('image_local', None)
                                attention_map = result.get('attention_map', None)
                        else:
                        # 否则使用原来的方式
                                model_output = self.model(images, text_inputs, masks=masks, debug=debug_mode)
                        # 处理可能返回3个值的情况（包含attention_map）
                        if isinstance(model_output, tuple) and len(model_output) == 3:
                                image_features, text_features, attention_map = model_output
                        else:
                                image_features, text_features = model_output
        
                        # 第三轮新增：训练时的正则化技术
                        if self.use_enhanced_regularization and self.model.training:
                                image_features, text_features = self._apply_regularization(image_features, text_features)
        
                        # if debug_mode:
                        # print(f"  图像特征形状: {image_features.shape}")
                        # print(f"  文本特征形状: {text_features.shape}")

                        # 多视角损失组合
                                loss = 0.0
                        # 融合视角
                                loss = loss + self.mv_weight_fused * self.criterion(image_features, text_features)
                        # 全局视角
                        if image_global is not None:
                                loss = loss + self.mv_weight_global * self.criterion(image_global, text_features)
                        # 局部视角
                        if image_local is not None:
                                loss = loss + self.mv_weight_local * self.criterion(image_local, text_features)
                    
                        # 添加mask监督损失
                            if attention_map is not None and masks is not None and self.mask_supervision_enabled:
                                mask_loss = self.mask_supervision_loss(attention_map, masks)
                                loss = loss + self.mask_supervision_weight * mask_loss
                    
                        # 添加attention正则化损失
                        if attention_map is not None and self.attention_regularization_enabled:
                        # 熵正则化
                                entropy_loss = self.attention_entropy_loss(attention_map)
                                loss = loss + entropy_loss
                        
                        # Inside/Outside对比损失
                            if masks is not None and self.inside_outside_enabled:
                                io_loss = self.attention_io_loss(attention_map, masks)
                                loss = loss + self.inside_outside_weight * io_loss
                    
                        # 第三轮新增：梯度惩罚正则化
                        if self.use_enhanced_regularization:
                            gradient_penalty = self.regularization_config.get('gradient_penalty_weight', 0.0)
                        if gradient_penalty > 0:
                            grad_penalty = self._compute_gradient_penalty(image_features, text_features)
                                loss = loss + gradient_penalty * grad_penalty
                        if not torch.isfinite(loss):
                            print("  警告: loss 非有限 (NaN/Inf)，跳过该batch")
                        # 清理梯度和scaler状态
                            self.optimizer.zero_grad()
                        if self.use_mixed_precision:
                        # 重置mixed precision状态
                            self._unscaled_this_step = False
                            continue
                        # 归一化损失（用于梯度累计）
                                loss = loss / grad_accum_steps
                
                        # 反向传播
                            self.scaler.scale(loss).backward()
                        else:
                        # # 为模型前向传播添加debug信息
                        # if debug_mode:
                        # print(f"  开始前向传播（标准精度）...")
                        # print(f"  输入图像形状: {images.shape if hasattr(images, 'shape') else type(images)}")
                        # print(f"  输入文本类型: {type(text_inputs)}")
                
                        # image_features, text_features = self.model(images, text_inputs, debug=debug_mode)
                        masks = batch.get('mask') if isinstance(batch, dict) else None  # 注意：数据集中使用'mask'而不是'masks'
                                attention_map = None
                
                            if hasattr(self.model, 'core'):
                        # 如果有 core 属性，直接调用核心模型
                            if masks is not None:
                                result = self.model.core(images, text_inputs, masks=masks)
                        else:
                                result = self.model.core(images, text_inputs)
                                image_features = result['image_features']
                                text_features = result['text_features']
                                image_global = result.get('image_global', None)
                                image_local = result.get('image_local', None)
                                attention_map = result.get('attention_map', None)
                        else:
                        # 否则使用原来的方式
                                model_output = self.model(images, text_inputs, masks=masks, debug=debug_mode)
                        # 处理可能返回3个值的情况（包含attention_map）
                        if isinstance(model_output, tuple) and len(model_output) == 3:
                                image_features, text_features, attention_map = model_output
                        else:
                                image_features, text_features = model_output
        
                        # if debug_mode:
                        # print(f"  图像特征形状: {image_features.shape}")
                        # print(f"  文本特征形状: {text_features.shape}")

                        # 多视角损失组合
                                loss = 0.0
                        # 融合视角
                                loss = loss + self.mv_weight_fused * self.criterion(image_features, text_features)
                        # 全局视角
                        if image_global is not None:
                                loss = loss + self.mv_weight_global * self.criterion(image_global, text_features)
                        # 局部视角
                        if image_local is not None:
                                loss = loss + self.mv_weight_local * self.criterion(image_local, text_features)
                
                        # 添加mask监督损失（与混合精度训练相同）
                            if attention_map is not None and masks is not None and self.mask_supervision_enabled:
                                mask_loss = self.mask_supervision_loss(attention_map, masks)
                                loss = loss + self.mask_supervision_weight * mask_loss
                
                        # 添加attention正则化损失
                        if attention_map is not None and self.attention_regularization_enabled:
                        # 熵正则化
                                entropy_loss = self.attention_entropy_loss(attention_map)
                                loss = loss + entropy_loss
                    
                        # Inside/Outside对比损失
                            if masks is not None and self.inside_outside_enabled:
                                io_loss = self.attention_io_loss(attention_map, masks)
                                loss = loss + self.inside_outside_weight * io_loss
                
                        if not torch.isfinite(loss):
                            print("  警告: loss 非有限 (NaN/Inf)，跳过该batch")
                        # 清理梯度状态
                            self.optimizer.zero_grad()
                            continue
                        # 归一化损失（用于梯度累计）
                                loss = loss / grad_accum_steps
                
                        # 反向传播
                            loss.backward()
            
                        # 梯度累计逻辑
                        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                        if self.use_mixed_precision:
                        # 混合精度训练的优化器步骤 - 修复unscale错误
                        try:
                        if grad_clip_val > 0:
                        # 检查是否已经unscale过
                        if not hasattr(self, '_unscaled_this_step') or not self._unscaled_this_step:
                            self.scaler.unscale_(self.optimizer)
                            self._unscaled_this_step = True
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_val)

                        # 检查梯度是否有效
                        if self._check_gradients_finite():
                            self.scaler.step(self.optimizer)
                        else:
                            print("  警告: 检测到无效梯度，跳过优化器步骤")

                            self.scaler.update()
                            self.optimizer.zero_grad()

                        # 重置unscale标志
                            self._unscaled_this_step = False

                        except RuntimeError as e:
                        if "unscale_() has already been called" in str(e):
                            print(f"  混合精度错误，重置scaler: {e}")
                        # 强制重置scaler状态
                            self.scaler.update()
                            self.optimizer.zero_grad()
                            self._unscaled_this_step = False
                        # 跳过这次更新，继续下一个batch
                            continue
                        else:
                            raise e
                else:
                # 标准训练的优化器步骤
                if grad_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_val)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # 更新学习率
                if hasattr(self.scheduler, 'step'):
                    self.scheduler.step()
                
                # 更新EMA
                    self._update_ema()
                
                    self.global_step += 1
            
                # 如果到达这里说明batch处理成功
                # 更新统计
                    total_loss += loss.item() * grad_accum_steps
                    successful_batches += 1
                    avg_loss = total_loss / successful_batches
                
                # 重置连续失败计数器
                    consecutive_failures = 0
                
                # 计算batch处理时间
                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time
                
                # 统计正常速度的batch
                if batch_time < slow_batch_threshold:
                    fast_batch_count += 1
                elif batch_time > slow_batch_threshold * 2:  # 超过60秒的batch
                    print(f"  警告: batch {batch_idx} 处理时间异常长: {batch_time:.1f}s")
                
                    progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'success': f'{successful_batches}/{batch_idx+1}',
                    'skipped': skipped_batches,
                    'time': f'{batch_time:.1f}s'
                    })
                
                # 内存清理
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                # 捕获所有其他异常
                consecutive_failures += 1
                skipped_batches += 1
                
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                
                print(f"  错误: batch {batch_idx} 处理失败 ({batch_time:.1f}s): {str(e)[:100]}")
                
                # 清理状态
                if hasattr(self, 'optimizer'):
                    self.optimizer.zero_grad()
                if self.use_mixed_precision and hasattr(self, '_unscaled_this_step'):
                    self._unscaled_this_step = False
                
                # 强制内存清理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 更新进度条显示错误信息
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}' if successful_batches > 0 else 'N/A',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'success': f'{successful_batches}/{batch_idx+1}',
                    'skipped': skipped_batches,
                    'errors': consecutive_failures
                })
                
                # 如果是严重错误或者连续失败过多，考虑提前结束
                if consecutive_failures >= max_consecutive_failures // 2:
                    print(f"  警告: 连续失败 {consecutive_failures} 个batch，请检查数据或模型配置")
                
                continue  # 继续处理下一个batch
                
        
        # Epoch完成后的统计报告
        epoch_end_time = time.time()
        total_epoch_time = epoch_end_time - epoch_start_time
        
        # 确保至少有一个成功的batch
        if successful_batches == 0:
            print("警告: 所有batch都处理失败，返回默认损失值")
            print(f"  总共处理了 {num_batches} 个batch，成功处理 {successful_batches} 个，跳过 {skipped_batches} 个")
            print(f"  Epoch总时间: {total_epoch_time:.1f}s")
            return {'train_loss': float('inf')}
        
        avg_loss = total_loss / successful_batches
        avg_batch_time = total_epoch_time / num_batches
        success_rate = successful_batches / num_batches
        
        # 详细的性能报告
        print(f"Epoch训练完成:")
        print(f"  ✓ 成功处理: {successful_batches}/{num_batches} 个batch ({success_rate:.1%})")
        print(f"  ✗ 跳过/失败: {skipped_batches} 个batch")
        print(f"  📊 平均损失: {avg_loss:.4f}")
        print(f"  ⏱️  总时间: {total_epoch_time:.1f}s")
        print(f"  ⚡ 平均batch时间: {avg_batch_time:.1f}s")
        print(f"  🚀 正常速度batch: {fast_batch_count}/{successful_batches}")
        
        # 性能警告
        if success_rate < 0.8:
            print(f"  ⚠️  警告: 成功率较低 ({success_rate:.1%})，建议检查数据质量")
        if avg_batch_time > slow_batch_threshold:
            print(f"  ⚠️  警告: 平均batch时间过长 ({avg_batch_time:.1f}s)，可能存在性能问题")
        
        return {
            'train_loss': avg_loss,
            'successful_batches': successful_batches,
            'total_batches': num_batches,
            'skipped_batches': skipped_batches,
            'success_rate': success_rate,
            'epoch_time': total_epoch_time,
            'avg_batch_time': avg_batch_time
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        # 获取验证用的模型（可能是EMA模型）
        validation_model = self._get_ema_model_for_validation()
        validation_model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # 收集所有特征用于全量检索计算（R@K 等）
        all_image_features = []
        all_text_features = []
        
        try:
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc="Validation"):
                    try:
                        # 数据移到设备
                        # 支持不同的键名格式
                        if 'images' in batch:
                            images = batch['images'].to(self.device)
                        elif 'image' in batch:
                            images = batch['image'].to(self.device)
                        else:
                            raise KeyError(f"Batch中找不到图像数据，可用键: {list(batch.keys())}")
                        
                        # 对于Qwen2.5-VL CLIP模型，使用原始文本而不是tokenized的结果
                        if hasattr(validation_model, 'config') and 'qwen' in str(type(validation_model)).lower():
                            # 使用原始文本
                            if 'texts' in batch:
                                texts = batch['texts']  # 这是原始文本列表
                            elif 'text' in batch:
                                text_data = batch['text']
                                # 如果是字符串，转换为列表；如果已经是列表，直接使用
                                texts = [text_data] if isinstance(text_data, str) else text_data
                            else:
                                raise KeyError(f"Batch中找不到文本数据，可用键: {list(batch.keys())}")
                            text_inputs = texts
                        else:
                            # 使用tokenized的结果
                            text_tokens = batch['text_tokens']
                            if isinstance(text_tokens, dict):
                                text_inputs = {k: v.to(self.device) for k, v in text_tokens.items()}
                            else:
                                text_inputs = text_tokens.to(self.device)
                        
                        # 前向传播
                        if hasattr(validation_model, 'core'):
                            # 如果有 core 属性，直接调用核心模型
                            # 兼容 'mask' 与 'masks' 两种键名
                            masks = None
                            if isinstance(batch, dict):
                                masks = batch.get('mask', batch.get('masks'))
                            if masks is not None:
                                result = validation_model.core(images, text_inputs, masks=masks)
                            else:
                                result = validation_model.core(images, text_inputs)
                            image_features = result['image_features']
                            text_features = result['text_features']
                        else:
                            # 否则使用原来的方式
                            masks = None
                            if isinstance(batch, dict):
                                masks = batch.get('mask', batch.get('masks'))
                            if masks is not None:
                                image_features, text_features = validation_model(images, text_inputs, masks=masks)
                            else:
                                image_features, text_features = validation_model(images, text_inputs)
            
                        # 计算损失
                        loss = self.criterion(image_features, text_features)
                        total_loss += loss.item()
                        
                        # 计算准确率
                        similarity = torch.matmul(image_features, text_features.T)
                        predictions = torch.argmax(similarity, dim=1)
                        labels = torch.arange(len(images), device=self.device)
                        correct_predictions += (predictions == labels).sum().item()
                        total_predictions += len(images)
                        
                        # 计算Top-5准确率（对于细粒度识别更有意义）
                        if not hasattr(self, 'top5_correct'):
                            self.top5_correct = 0
                            self.top5_total = 0
                        
                        _, top5_predictions = torch.topk(similarity, k=min(5, similarity.size(1)), dim=1)
                        top5_matches = (top5_predictions == labels.unsqueeze(1)).any(dim=1)
                        self.top5_correct += top5_matches.sum().item()
                        self.top5_total += len(images)
                        
                        # 收集特征用于R@1计算
                        all_image_features.append(image_features.cpu())
                        all_text_features.append(text_features.cpu())
                        
                    except Exception as e:
                        print(f"验证batch时出错: {e}")
                        continue
        finally:
            # 恢复原始模型权重（如果使用了EMA）
            self._restore_model_from_ema()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        top5_accuracy = self.top5_correct / self.top5_total if hasattr(self, 'top5_total') and self.top5_total > 0 else 0.0

        # 计算全量检索指标：i2t/t2i 的 R@1/5/10
        metrics = {
            'val_loss': avg_loss,
            # batch级分类指标保留在metrics中但不打印（对检索不稳定）
            'val_acc': accuracy,
            'val_top5_acc': top5_accuracy,
            # 检索指标
            'val_r1': 0.0,
            'i2t_r1': 0.0,
            't2i_r1': 0.0,
            'i2t_r5': 0.0,
            't2i_r5': 0.0,
            'i2t_r10': 0.0,
            't2i_r10': 0.0,
            'i2t_mean_rank': 0.0,
            't2i_mean_rank': 0.0,
            'i2t_median_rank': 0.0,
            't2i_median_rank': 0.0,
            'i2t_mrr': 0.0,
            't2i_mrr': 0.0,
            'i2t_map10': 0.0,
            't2i_map10': 0.0,
        }

        if all_image_features and all_text_features:
            # 拼接所有特征
            image_features = torch.cat(all_image_features, dim=0)
            text_features = torch.cat(all_text_features, dim=0)

            # 归一化
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)

            # 计算相似度矩阵
            similarity_matrix = torch.matmul(image_features, text_features.T)
            n_samples = similarity_matrix.size(0)

            # 计算 ranks
            i2t_ranks = []
            t2i_ranks = []
            for i in range(n_samples):
                # i->t
                scores_i = similarity_matrix[i]
                sorted_i = torch.argsort(scores_i, descending=True)
                rank_i = (sorted_i == i).nonzero(as_tuple=True)[0].item() + 1
                i2t_ranks.append(rank_i)
                # t->i
                scores_t = similarity_matrix[:, i]
                sorted_t = torch.argsort(scores_t, descending=True)
                rank_t = (sorted_t == i).nonzero(as_tuple=True)[0].item() + 1
                t2i_ranks.append(rank_t)

            def recall_at_k(ranks, k):
                return sum(1 for r in ranks if r <= k) / len(ranks) if ranks else 0.0
            def mean_rank(ranks):
                return float(sum(ranks) / len(ranks)) if ranks else 0.0
            def median_rank(ranks):
                import math
                if not ranks:
                    return 0.0
                sr = sorted(ranks)
                n = len(sr)
                mid = n // 2
                if n % 2 == 1:
                    return float(sr[mid])
                return float((sr[mid - 1] + sr[mid]) / 2.0)
            def mrr(ranks):
                return float(sum(1.0 / r for r in ranks) / len(ranks)) if ranks else 0.0
            def map_at_k(ranks, k):
                # 单一相关项情况下，AP@K=1/rank 若 rank<=K，否则为0
                if not ranks:
                    return 0.0
                vals = [(1.0 / r) if r <= k else 0.0 for r in ranks]
                return float(sum(vals) / len(vals))

            i2t_r1 = recall_at_k(i2t_ranks, 1)
            t2i_r1 = recall_at_k(t2i_ranks, 1)
            i2t_r5 = recall_at_k(i2t_ranks, 5)
            t2i_r5 = recall_at_k(t2i_ranks, 5)
            i2t_r10 = recall_at_k(i2t_ranks, 10)
            t2i_r10 = recall_at_k(t2i_ranks, 10)
            i2t_mean = mean_rank(i2t_ranks)
            t2i_mean = mean_rank(t2i_ranks)
            i2t_med = median_rank(i2t_ranks)
            t2i_med = median_rank(t2i_ranks)
            i2t_mrr = mrr(i2t_ranks)
            t2i_mrr = mrr(t2i_ranks)
            i2t_map10 = map_at_k(i2t_ranks, 10)
            t2i_map10 = map_at_k(t2i_ranks, 10)

            metrics.update({
                'i2t_r1': i2t_r1,
                't2i_r1': t2i_r1,
                'i2t_r5': i2t_r5,
                't2i_r5': t2i_r5,
                'i2t_r10': i2t_r10,
                't2i_r10': t2i_r10,
                'val_r1': (i2t_r1 + t2i_r1) / 2.0,
                'i2t_mean_rank': i2t_mean,
                't2i_mean_rank': t2i_mean,
                'i2t_median_rank': i2t_med,
                't2i_median_rank': t2i_med,
                'i2t_mrr': i2t_mrr,
                't2i_mrr': t2i_mrr,
                'i2t_map10': i2t_map10,
                't2i_map10': t2i_map10,
            })

        # 重置Top-5计数器
        if hasattr(self, 'top5_correct'):
            self.top5_correct = 0
            self.top5_total = 0

        return metrics
    
    def train(self, max_epochs: int, save_dir: str):
        """执行完整训练过程"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取监控参数
        save_top_k = self.config['save_top_k']
        monitor = self.config['monitor']
        mode = self.config['mode']
        
        # 初始化最佳指标跟踪
        if mode == 'min':
            self.best_val_loss = float('inf')
            best_metric = float('inf')
        else:
            self.best_val_loss = float('-inf')
            best_metric = float('-inf')
        
        # 保存的模型列表
        saved_models = []
        
        print(f"开始训练，最大epoch数: {max_epochs}")
        print(f"监控指标: {monitor}, 模式: {mode}, 保存前{save_top_k}个模型")
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            
            try:
                # 训练
                train_metrics = self.train_epoch()
                
                # 验证
                val_metrics = self.validate_epoch()
                
                # 打印结果
                print(f"Epoch {epoch+1}/{max_epochs}:")
                print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"  Retrieval R@1 (i2t/t2i/avg): {val_metrics['i2t_r1']:.4f} / {val_metrics['t2i_r1']:.4f} / {val_metrics['val_r1']:.4f}")
                print(f"  Retrieval R@5 (i2t/t2i): {val_metrics['i2t_r5']:.4f} / {val_metrics['t2i_r5']:.4f}")
                print(f"  Retrieval R@10(i2t/t2i): {val_metrics['i2t_r10']:.4f} / {val_metrics['t2i_r10']:.4f}")
                print(f"  Mean/Median Rank (i2t/t2i): {val_metrics['i2t_mean_rank']:.2f}/{val_metrics['t2i_mean_rank']:.2f}  |  {val_metrics['i2t_median_rank']:.0f}/{val_metrics['t2i_median_rank']:.0f}")
                print(f"  MRR (i2t/t2i): {val_metrics['i2t_mrr']:.4f} / {val_metrics['t2i_mrr']:.4f}")
                print(f"  mAP@10 (i2t/t2i): {val_metrics['i2t_map10']:.4f} / {val_metrics['t2i_map10']:.4f}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # 获取监控指标 - 优先使用配置的monitor，否则使用R@1
                if monitor in val_metrics:
                    current_metric = val_metrics[monitor]
                elif 'val_r1' in val_metrics:
                    current_metric = val_metrics['val_r1']  # 默认使用R@1作为主要指标
                    print(f"  使用R@1作为监控指标: {current_metric:.4f}")
                else:
                    current_metric = val_metrics['val_loss']  # 兜底使用val_loss
                    print(f"  使用val_loss作为监控指标: {current_metric:.4f}")
                # 检查是否是最佳模型
                is_best = False
                if mode == 'min':
                    if current_metric < best_metric:
                        best_metric = current_metric
                        is_best = True
                else:
                    if current_metric > best_metric:
                        best_metric = current_metric
                        is_best = True
                
                # 优化的保存策略：只保留最新和最佳两个checkpoint
                
                # 1. 始终保存最新的checkpoint（用于断点续训）
                latest_checkpoint_path = os.path.join(save_dir, 'latest_checkpoint.pth')
                self._save_checkpoint_safely(latest_checkpoint_path, epoch + 1, val_metrics)
                print(f"  保存最新checkpoint: latest_checkpoint.pth")
                
                # 2. 只有当性能更好时才保存最佳模型（用于使用）
                if is_best:
                    self.best_val_loss = val_metrics['val_loss']
                    best_model_path = os.path.join(save_dir, 'best_model.pth')
                    self._save_checkpoint_safely(best_model_path, epoch + 1, val_metrics)
                    print(f"  保存最佳模型: {monitor} = {best_metric:.4f}")
                
                # 3. 清理旧的epoch checkpoint文件（如果存在）
                self._cleanup_old_epoch_checkpoints(save_dir)
                
                # 内存清理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"Epoch {epoch+1} 训练出错: {e}")
                import traceback
                traceback.print_exc()
                
                # 紧急保存到最新checkpoint（覆盖原有的）
                emergency_save_path = os.path.join(save_dir, 'latest_checkpoint.pth')
                try:
                    self._save_checkpoint_safely(emergency_save_path, epoch + 1, {'error': str(e)})
                    print(f"紧急保存到: {emergency_save_path}")
                except Exception as save_error:
                    print(f"紧急保存失败: {save_error}")
                    # 创建临时紧急文件
                    temp_emergency = os.path.join(save_dir, f'emergency_temp_{epoch+1}.pth')
                    try:
                        self._save_minimal_checkpoint(temp_emergency, epoch + 1, {'error': str(e)})
                        print(f"临时紧急保存到: {temp_emergency}")
                    except:
                        print("所有保存方法都失败，跳过保存")
                
                # 继续下一个epoch
                continue
        
        print("训练完成!")
        print(f"最佳{monitor}: {best_metric:.4f}")
    
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        """保存模型checkpoint，支持lora权重保存"""
        if self.use_lora:
            # 保存lora权重
            checkpoint = {
                'epoch': epoch,
                'global_step': self.global_step,
                'lora_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'metrics': metrics,
                'config': self.config,
                'best_val_loss': self.best_val_loss,
                'lora_config': self.lora_config
            }
        else: 
            checkpoint = {
                'epoch': epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'metrics': metrics,
                'config': self.config,
                'best_val_loss': self.best_val_loss
            }
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """加载模型checkpoint,支持lira加载"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 判断是全量/lora的权重加载
        if self.use_lora and 'lora_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['lora_state_dict'])
            print(f"  - 加载lora权重:{filepath}")
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  - 加载全量权重:{filepath}")
        # 尝试加载优化器和调度器状态（精简模式可能没有）
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("警告: checkpoint中没有优化器状态，使用默认状态")
            
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("警告: checkpoint中没有调度器状态，使用默认状态")
            
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"从checkpoint恢复训练: epoch {self.current_epoch}, global_step {self.global_step}")
        return checkpoint['metrics']
    
    def _save_checkpoint_safely(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        """安全保存checkpoint - 带磁盘空间检查和错误处理"""
        import shutil
        
        # 检查磁盘空间
        try:
            total, used, free = shutil.disk_usage(os.path.dirname(filepath))
            required_space = 12 * 1024 * 1024 * 1024  # 预留12GB空间
            
            if free < required_space:
                print(f"警告: 磁盘空间不足！可用: {free/1024**3:.1f}GB, 需要: {required_space/1024**3:.1f}GB")
                # 尝试清理旧文件
                self._cleanup_old_epoch_checkpoints(os.path.dirname(filepath))
                
                # 再次检查空间
                total, used, free = shutil.disk_usage(os.path.dirname(filepath))
                if free < required_space:
                    print(f"清理后仍空间不足，使用精简保存模式")
                    self._save_minimal_checkpoint(filepath, epoch, metrics)
                    return
        
        except Exception as e:
            print(f"磁盘空间检查失败: {e}")
        
        # 创建checkpoint
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        # 安全保存策略
        temp_filepath = filepath + '.tmp'
        try:
            # 先保存到临时文件
            torch.save(checkpoint, temp_filepath)
            # 成功后重命名
            os.replace(temp_filepath, filepath)
            
        except Exception as e:
            print(f"Checkpoint保存失败: {e}")
            # 清理临时文件
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            
            # 尝试精简保存
            try:
                print("尝试精简保存模式...")
                self._save_minimal_checkpoint(filepath, epoch, metrics)
            except Exception as e2:
                print(f"精简保存也失败: {e2}")
                raise e
    
    def _save_minimal_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        """精简保存模式，只保存必要信息"""
        try:
            # 只保存模型权重和基本信息
            minimal_checkpoint = {
                'epoch': epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'metrics': metrics,
                'best_val_loss': self.best_val_loss,
                'config': {
                    'learning_rate': self.config['learning_rate'],
                    'batch_size': self.config['batch_size']
                }
            }
            
            # 压缩保存
            torch.save(minimal_checkpoint, filepath, _use_new_zipfile_serialization=False)
            print(f"精简checkpoint保存成功: {os.path.basename(filepath)}")
            
        except Exception as e:
            print(f"精简保存失败: {e}")
            raise e
    
    def _cleanup_old_epoch_checkpoints(self, save_dir: str):
        """清理旧的epoch checkpoint文件，只保留latest_checkpoint.pth和best_model.pth"""
        try:
            if not os.path.exists(save_dir):
                return
            
            # 要保留的文件
            keep_files = {'latest_checkpoint.pth', 'best_model.pth'}
            
            # 清理旧的checkpoint文件
            deleted_count = 0
            for file in os.listdir(save_dir):
                if file.endswith('.pth') and file not in keep_files:
                    # 删除旧的epoch checkpoint和emergency文件
                    if ('checkpoint_epoch_' in file or 
                        'emergency' in file or 
                        file.endswith('.tmp')):
                        filepath = os.path.join(save_dir, file)
                        try:
                            os.remove(filepath)
                            deleted_count += 1
                            print(f"清理旧文件: {file}")
                        except Exception as e:
                            print(f"删除文件失败 {file}: {e}")
            
            if deleted_count > 0:
                print(f"总共清理了 {deleted_count} 个旧checkpoint文件")
                
        except Exception as e:
            print(f"清理checkpoint失败: {e}")
