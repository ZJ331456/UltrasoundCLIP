"""
Qwen2.5-VL视觉编码器+文本编码器模型模块 - 修复批量处理问题
"""

from ast import Pass
import torch
import torch.nn as nn
import json
import os
from typing import Optional, Tuple, Dict, Any, Union
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModel, AutoTokenizer
import torch.nn.functional as F
from pathlib import Path


class SpatialAttentionModule(nn.Module):
    """空间注意力模块 - 专为超声图像设计"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim // 4, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim // 4, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: [seq_len, hidden_dim]
        # 数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            return x  # 如果输入有问题，直接返回
        
        x_transposed = x.transpose(0, 1).unsqueeze(0)  # [1, hidden_dim, seq_len]
        
        # 空间注意力权重计算（添加数值稳定性）
        attention = F.relu(self.conv1(x_transposed))
        attention = self.dropout(attention)
        attention = torch.sigmoid(self.conv2(attention))  # [1, 1, seq_len]
        
        # 检查注意力权重
        if torch.isnan(attention).any() or torch.isinf(attention).any():
            # 如果注意力权重有问题，使用均匀权重
            attention = torch.ones_like(attention) / attention.size(-1)
        
        attention = attention.squeeze(0).transpose(0, 1)  # [seq_len, 1]
        
        # 应用注意力权重
        attended_features = x * attention
        
        # 最终检查
        if torch.isnan(attended_features).any() or torch.isinf(attended_features).any():
            return x  # 回退到原始输入
        
        return attended_features


class ChannelAttentionModule(nn.Module):
    """通道注意力模块 - 突出重要特征通道"""
    def __init__(self, hidden_dim, reduction=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // reduction, hidden_dim, bias=False)
        )
        
    def forward(self, x):
        # x: [seq_len, hidden_dim]
        # 数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            return x  # 如果输入有问题，直接返回
            
        x_transposed = x.transpose(0, 1).unsqueeze(0)  # [1, hidden_dim, seq_len]
        
        # 全局平均池化和最大池化
        avg_out = self.avg_pool(x_transposed).squeeze(-1).squeeze(0)  # [hidden_dim]
        max_out = self.max_pool(x_transposed).squeeze(-1).squeeze(0)  # [hidden_dim]
        
        # 检查池化结果
        if torch.isnan(avg_out).any() or torch.isinf(avg_out).any():
            avg_out = torch.zeros_like(avg_out)
        if torch.isnan(max_out).any() or torch.isinf(max_out).any():
            max_out = torch.zeros_like(max_out)
        
        # 通过全连接层
        avg_attention = self.fc(avg_out)
        max_attention = self.fc(max_out)
        
        # 检查全连接层输出
        if torch.isnan(avg_attention).any() or torch.isinf(avg_attention).any():
            avg_attention = torch.zeros_like(avg_attention)
        if torch.isnan(max_attention).any() or torch.isinf(max_attention).any():
            max_attention = torch.zeros_like(max_attention)
        
        # 合并注意力权重
        attention = torch.sigmoid(avg_attention + max_attention)  # [hidden_dim]
        
        # 检查注意力权重
        if torch.isnan(attention).any() or torch.isinf(attention).any():
            # 使用均匀权重
            attention = torch.ones_like(attention)
        
        # 应用通道注意力
        attended_features = x * attention.unsqueeze(0)  # [seq_len, hidden_dim]
        
        # 最终检查
        if torch.isnan(attended_features).any() or torch.isinf(attended_features).any():
            return x  # 回退到原始输入
            
        return attended_features


class UltrasoundFeatureEnhancer(nn.Module):
    """超声图像特征增强模块"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 多尺度特征提取
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        
        # 特征融合
        self.fusion_conv = nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: [seq_len, hidden_dim]
        # 数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            return x  # 如果输入有问题，直接返回
            
        x_transposed = x.transpose(0, 1).unsqueeze(0)  # [1, hidden_dim, seq_len]
        
        # 多尺度卷积特征
        multi_scale_features = []
        for conv in self.conv_layers:
            try:
                feature = F.relu(conv(x_transposed))
                # 检查每个卷积层的输出
                if torch.isnan(feature).any() or torch.isinf(feature).any():
                    # 如果有问题，使用零特征
                    feature = torch.zeros_like(feature)
                multi_scale_features.append(feature)
            except Exception as e:
                # 如果卷积失败，添加零特征
                feature = torch.zeros_like(x_transposed)
                multi_scale_features.append(feature)
        
        # 特征融合
        try:
            concatenated = torch.cat(multi_scale_features, dim=1)  # [1, hidden_dim*3, seq_len]
            fused = self.fusion_conv(concatenated)  # [1, hidden_dim, seq_len]
            
            # 检查融合结果
            if torch.isnan(fused).any() or torch.isinf(fused).any():
                fused = torch.zeros_like(fused)
        except Exception as e:
            # 如果融合失败，使用零特征
            fused = torch.zeros_like(x_transposed)
        
        # 转回原始格式
        fused = fused.squeeze(0).transpose(0, 1)  # [seq_len, hidden_dim]
        
        # 残差连接和归一化（增强数值稳定性）
        try:
            residual = x + self.dropout(fused)
            # 检查残差连接结果
            if torch.isnan(residual).any() or torch.isinf(residual).any():
                residual = x  # 回退到原始输入
            
            output = self.layer_norm(residual)
            # 检查归一化结果
            if torch.isnan(output).any() or torch.isinf(output).any():
                output = x  # 回退到原始输入
        except Exception as e:
            output = x  # 如果所有操作都失败，直接返回输入
        
        return output

class QwenVLCLIPModel(nn.Module):
    """Qwen2.5-VL CLIP模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Qwen2.5-VL CLIP模型
        Args:
            config: 模型配置
        """
        super().__init__()
        
        self.config = config
        # self.model_path = config['model_path']
        self.model_path = config['vision_encoder']['model_path']
        self.embed_dim = config['embed_dim']
        
        # 初始化组件
        self._init_models()
        self._init_projections()
        
        print(f"Qwen2.5-VL CLIP模型初始化完成")
        print(f"  模型路径: {self.model_path}")
        print(f"  嵌入维度: {self.embed_dim}")
    
    def _init_models(self):
        """初始化视觉和文本模型"""
        print("正在加载Qwen2.5-VL模型...")
        
        # 加载完整模型并指定设备和数据类型
        self.full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,  # 优化内存使用
            trust_remote_code=True   # 可能需要信任远程代码
        )

        # 提取视觉编码器
        self.vision_model = self.full_model.model.visual
        
        # 加载文本编码器（BERT）
        text_config = self.config['text_encoder']
        text_model_name = text_config['model_name']
        self.text_encoder = AutoModel.from_pretrained(
            text_model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        # print(f"打印一下图像处理器得配置")
        # print(self.processor.image_processor)
        """
        打印一下图像处理器得配置
        Qwen2VLImageProcessor {
        "do_convert_rgb": true,
        "do_normalize": true,
        "do_rescale": true,
        "do_resize": true,
        "image_mean": [
            0.48145466,
            0.4578275,
            0.40821073
        ],
        "image_processor_type": "Qwen2VLImageProcessor",
        "image_std": [
            0.26862954,
            0.26130258,
            0.27577711
        ],
        "max_pixels": 12845056,
        "merge_size": 2,
        "min_pixels": 3136,
        "patch_size": 14,
        "processor_class": "Qwen2_5_VLProcessor",
        "resample": 3,
        "rescale_factor": 0.00392156862745098,
        "size": {
            "longest_edge": 12845056,
            "shortest_edge": 3136
        },
        "temporal_patch_size": 2
        }
        """
        # 初始化tokenizer缓存
        self._tokenizer = None
        
        print(f"  视觉编码器加载完成 (设备: {next(self.vision_model.parameters()).device})")
        print(f"  文本编码器: {text_model_name} (设备: {next(self.text_encoder.parameters()).device})")
    
    def _init_projections(self):
        """初始化投影层"""
        # 获取视觉特征维度-这应该与Qwen2.5-VL的视觉编码器输出维度匹配
        vision_dim = self.config['vision_encoder']['vision_dim']
        # 获取文本特征维度
        text_dim = self.text_encoder.config.hidden_size
        
        # 获取设备和数据类型信息
        vision_device = next(self.vision_model.parameters()).device
        vision_dtype = next(self.vision_model.parameters()).dtype
        text_device = next(self.text_encoder.parameters()).device
        
        # 增强的投影层 - 多层MLP增强特征表达
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_dim, vision_dim // 2),
            nn.LayerNorm(vision_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(vision_dim // 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        ).to(device=vision_device, dtype=vision_dtype)
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.LayerNorm(text_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(text_dim // 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        ).to(device=text_device, dtype=vision_dtype)
        
        # 注意力增强模块
        self.spatial_attention = SpatialAttentionModule(vision_dim).to(device=vision_device, dtype=vision_dtype)
        self.channel_attention = ChannelAttentionModule(vision_dim).to(device=vision_device, dtype=vision_dtype)
        self.feature_enhancer = UltrasoundFeatureEnhancer(vision_dim).to(device=vision_device, dtype=vision_dtype)
        
        # 温度参数 - 确保设备和数据类型匹配，增强数值稳定性
        # 使用更保守的初始化值，避免极端情况
        initial_temp = 0.07
        logit_scale_value = torch.log(torch.tensor(1.0 / initial_temp))
        # 确保数值在合理范围内
        logit_scale_value = torch.clamp(logit_scale_value, min=1.0, max=4.0)  # 对应温度范围[0.018, 0.368]
        logit_scale_value = logit_scale_value.to(device=vision_device, dtype=vision_dtype)
        self.logit_scale = nn.Parameter(logit_scale_value)
        
        # 注册hook来监控和约束logit_scale
        self.logit_scale.register_hook(self._logit_scale_hook)
        
        print(f"  投影层初始化完成: 视觉{vision_dim}→{self.embed_dim} (设备: {vision_device}, 类型: {vision_dtype}), 文本{text_dim}→{self.embed_dim} (设备: {text_device}, 类型: {vision_dtype})")
    
    def _logit_scale_hook(self, grad):
        """logit_scale参数的梯度hook，确保数值稳定性"""
        if grad is not None:
            # 梯度裁剪，防止梯度爆炸
            grad = torch.clamp(grad, min=-1.0, max=1.0)
            # 检查梯度是否有异常值
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                print("WARNING: logit_scale梯度包含NaN/Inf，设置为零")
                grad = torch.zeros_like(grad)
        return grad
    
    def _apply_logit_scale_constraints(self):
        """应用logit_scale约束"""
        with torch.no_grad():
            # 限制logit_scale在合理范围内，对应温度范围[0.01, 0.5]
            device = self.logit_scale.device
            dtype = self.logit_scale.dtype
            min_val = torch.log(torch.tensor(2.0, device=device, dtype=dtype))   # min temp = 0.5
            max_val = torch.log(torch.tensor(100.0, device=device, dtype=dtype))   # max temp = 0.01
            self.logit_scale.clamp_(min=min_val, max=max_val)
    
    def encode_image(self, images: Union[torch.Tensor, str, Image.Image, list], debug: bool = False) -> torch.Tensor:
        """
        编码图像
        Args:
            images: 图像输入，可以是tensor、路径、PIL图像或列表
            debug: 是否输出调试信息
        Returns:
            image_features: 归一化的图像特征
        """
        # 预处理输入图像为PIL图像列表
        pil_images = self._preprocess_images(images)
        # print(f"pil_images是什么样：{pil_images}")
        # 使用Qwen的processor处理图像
        # 这一步很关键！
        # 因为图像的尺寸和分割尺寸需要qwen处理器自适应
        dummy_text = "<|vision_start|><|image_pad|><|vision_end|>"
        inputs = self.processor(
            text=[dummy_text] * len(pil_images), 
            images=pil_images, 
            return_tensors="pt"
        )
        
        # 移动到正确设备
        device = next(self.vision_model.parameters()).device
        pixel_values = inputs["pixel_values"].to(device)
        image_grid_thw = inputs["image_grid_thw"].to(device)
        
        # if debug:
        #     print(f"DEBUG: pixel_values shape: {pixel_values.shape}")
        #     print(f"DEBUG: image_grid_thw: {image_grid_thw}")
        #     print(f"DEBUG: batch size: {len(pil_images)}")
        
        # 提取视觉特征 - 使用正确的批量处理方式
        with torch.no_grad():
            # 1. 拿到整批的patch特征
            all_feats = self.vision_model(pixel_values, grid_thw=image_grid_thw)
            # 2. 计算每张图的 patch 数量并拆分（使用正确的公式）
            patch_nums = (
                image_grid_thw.prod(dim=1) // (self.vision_model.spatial_merge_size ** 2)
            ).tolist()
            # 3. 按照每张图片的patch数量分组
            image_embeddings_list = torch.split(all_feats, patch_nums, dim=0)  # list[Tensor]
        
        # if debug:
        #     print(f"DEBUG: patch_nums: {patch_nums}")
        #     print(f"DEBUG: total patches: {sum(patch_nums)}, all_feats shape: {all_feats.shape}")
        #     print(f"DEBUG: spatial_merge_size: {self.vision_model.spatial_merge_size}")
        
        # 处理特征维度和分组 - 添加安全检查
        batch_size = len(pil_images)
        
        # 验证特征是否为空
        # if debug:
        #     print(f"DEBUG: image_embeddings_list长度: {len(image_embeddings_list)}")
        #     for i, emb in enumerate(image_embeddings_list):
        #         print(f"  图像 {i}: 特征形状 {emb.shape}")
        
        # 检查是否有空的特征
        valid_embeddings = []
        for i, emb in enumerate(image_embeddings_list):
            if emb.numel() < 10:  # 如果特征元素太少
                print(f"警告: 图像 {i} 特征过少 ({emb.numel()} 元素)，使用全零特征替代")
                # 创建一个默认的特征
                dummy_feature = torch.zeros(1, 2048, device=emb.device, dtype=emb.dtype)
                valid_embeddings.append(dummy_feature)
            else:
                valid_embeddings.append(emb)
        
        image_embeddings_list = valid_embeddings
        
        process_type = self.config['vision_encoder']['process_type']
        try:
            if process_type == 'default':
                image_features = self._process_vision_features_from_list(
                    image_embeddings_list, batch_size, debug
                )
            elif process_type == 'projection':
                image_features = self._process_vision_features_projection_from_list(
                    image_embeddings_list, batch_size, debug
                )
            elif process_type == 'attention_pooling':
                image_features = self._process_vision_features_attention_pooling_from_list(
                    image_embeddings_list, batch_size, debug
                )
            elif process_type == 'top_k':
                image_features = self._process_vision_features_top_k_from_list(
                    image_embeddings_list, batch_size, debug
                )
            elif process_type == 'cls_token':
                image_features = self._process_vision_features_cls_token_from_list(
                    image_embeddings_list, batch_size, debug
                )
            else:
                # 默认处理方式
                image_features = self._process_vision_features_from_list(
                    image_embeddings_list, batch_size, debug
                )
        except Exception as e:
            print(f"特征处理失败: {e}")
            if debug:
                print(f"使用简单平均池化作为备用方案")
            # 备用方案：简单平均池化
            batch_features = []
            for emb in image_embeddings_list:
                if emb.numel() > 0:
                    avg_feature = emb.mean(dim=0, keepdim=True)
                    if avg_feature.shape[-1] != 2048:
                        # 如果维度不对，使用线性层调整
                        if not hasattr(self, '_emergency_projection'):
                            self._emergency_projection = torch.nn.Linear(
                                avg_feature.shape[-1], 2048
                            ).to(avg_feature.device)
                        avg_feature = self._emergency_projection(avg_feature)
                else:
                    avg_feature = torch.zeros(1, 2048, device=next(self.parameters()).device)
                batch_features.append(avg_feature)
            image_features = torch.cat(batch_features, dim=0)
            
        # 投影到统一维度前进行数值检查
        if torch.isnan(image_features).any() or torch.isinf(image_features).any():
            print("WARNING: 投影前图像特征包含NaN/Inf，进行清理")
            image_features = torch.where(torch.isnan(image_features) | torch.isinf(image_features),
                                       torch.zeros_like(image_features), image_features)
            # 重新归一化
            image_features = F.normalize(image_features, p=2, dim=-1)
        
        vision_device = next(self.vision_projection.parameters()).device
        if image_features.device != vision_device:
            image_features = image_features.to(vision_device)
        
        # 应用投影层前添加数值限制
        image_features = torch.clamp(image_features, min=-10.0, max=10.0)
        image_features = self.vision_projection(image_features)
        
        # 投影后再次检查
        if torch.isnan(image_features).any() or torch.isinf(image_features).any():
            print("WARNING: 投影后图像特征包含NaN/Inf，使用安全归一化")
            image_features = torch.where(torch.isnan(image_features) | torch.isinf(image_features),
                                       torch.zeros_like(image_features), image_features)
        
        # 更保守的归一化策略
        # 1. 带epsilon的L2归一化
        norm = torch.norm(image_features, p=2, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-8)  # 避免除零
        normalized_features = image_features / norm
        
        # 2. 训练时采用更稳定的特征缩放策略
        if self.training:
            # 使用极其保守的特征增强，优先稳定性
            scale_factor = 1.02  # 进一步降低从1.05到1.02
            enhanced_features = normalized_features * scale_factor
            
            # 重新归一化（更严格的数值稳定性）
            norm = torch.norm(enhanced_features, p=2, dim=-1, keepdim=True)
            norm = torch.clamp(norm, min=1e-6, max=1e6)  # 双边裁剪避免极值
            enhanced_features = enhanced_features / norm
            
            # 特征范围限制，防止极值
            enhanced_features = torch.clamp(enhanced_features, min=-10.0, max=10.0)
            
            # 最终数值检查
            if torch.isnan(enhanced_features).any() or torch.isinf(enhanced_features).any():
                print("WARNING: 最终特征仍包含NaN/Inf，回退到简单归一化")
                return F.normalize(image_features, p=2, dim=-1, eps=1e-6)
            
            return enhanced_features
        else:
            return normalized_features
    
    def _preprocess_images(self, images: Union[torch.Tensor, str, Image.Image, list]) -> list:
        """预处理图像输入为PIL图像列表，增强错误处理和尺寸验证"""
        try:
            if isinstance(images, str):
                # 图像路径
                if not os.path.exists(images):
                    raise FileNotFoundError(f"图像文件不存在: {images}")
                img = Image.open(images)
                # 验证图像有效性
                img.verify()
                img = Image.open(images)  # 重新打开，因为verify()会关闭文件
                return [self._validate_and_convert_image(img)]
                
            elif isinstance(images, Image.Image):
                return [self._validate_and_convert_image(images)]
                
            elif isinstance(images, torch.Tensor):
                # tensor转换为PIL图像
                if len(images.shape) < 3:
                    raise ValueError(f"图像tensor维度不足: {images.shape}")
                
                pil_images = []
                if len(images.shape) == 4:  # Batch tensor [B, C, H, W]
                    for i in range(images.shape[0]):
                        img_tensor = images[i]
                        pil_image = self._tensor_to_pil(img_tensor)
                        pil_images.append(self._validate_and_convert_image(pil_image))
                elif len(images.shape) == 3:  # Single image [C, H, W]
                    pil_image = self._tensor_to_pil(images)
                    pil_images.append(self._validate_and_convert_image(pil_image))
                else:
                    raise ValueError(f"不支持的tensor形状: {images.shape}")
                
                return pil_images
                
            elif isinstance(images, list):
                # 处理列表输入，支持混合类型
                if len(images) == 0:
                    raise ValueError("图像列表不能为空")
                
                pil_images = []
                for i, img in enumerate(images):
                    try:
                        if isinstance(img, str):
                            if not os.path.exists(img):
                                raise FileNotFoundError(f"图像文件不存在: {img}")
                            pil_img = Image.open(img)
                            pil_img.verify()
                            pil_img = Image.open(img)  # 重新打开
                            pil_images.append(self._validate_and_convert_image(pil_img))
                        elif isinstance(img, Image.Image):
                            pil_images.append(self._validate_and_convert_image(img))
                        elif isinstance(img, torch.Tensor):
                            pil_img = self._tensor_to_pil(img)
                            pil_images.append(self._validate_and_convert_image(pil_img))
                        else:
                            raise ValueError(f"列表中包含不支持的图像类型: {type(img)}")
                    except Exception as e:
                        print(f"警告: 处理第 {i} 张图像时出错: {e}")
                        continue
                
                if len(pil_images) == 0:
                    raise ValueError("没有成功处理的图像")
                
                return pil_images
            else:
                raise ValueError(f"不支持的图像输入类型: {type(images)}")
                
        except Exception as e:
            print(f"图像预处理失败: {e}")
            raise e
    
    def _validate_and_convert_image(self, img: Image.Image) -> Image.Image:
        """验证和转换图像格式"""
        try:
            # 检查图像模式
            if img.mode not in ['RGB', 'RGBA', 'L', 'P']:
                print(f"警告: 不常见的图像模式 {img.mode}，尝试转换为RGB")
            
            # 统一转换为RGB模式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 检查图像尺寸
            width, height = img.size
            if width < 32 or height < 32:
                raise ValueError(f"图像尺寸过小: {width}x{height}")
            if width > 4096 or height > 4096:
                print(f"警告: 图像尺寸过大 {width}x{height}，可能影响处理速度")
            
            # 检查图像是否损坏
            try:
                img.load()
            except Exception as e:
                raise ValueError(f"图像数据损坏: {e}")
            
            return img
            
        except Exception as e:
            print(f"图像验证失败: {e}")
            raise e
    
    def _tensor_to_pil(self, img_tensor: torch.Tensor) -> Image.Image:
        """将tensor转换为PIL图像"""
        try:
            if len(img_tensor.shape) != 3:
                raise ValueError(f"期望3维tensor [C, H, W]，但得到 {img_tensor.shape}")
            
            channels, height, width = img_tensor.shape
            
            # 检查通道数
            if channels not in [1, 3, 4]:
                raise ValueError(f"不支持的通道数: {channels}")
            
            # 如果是单通道，复制为三通道
            if channels == 1:
                img_tensor = img_tensor.repeat(3, 1, 1)
            elif channels == 4:  # RGBA，丢弃alpha通道
                img_tensor = img_tensor[:3]
            
            # 反归一化（使用CLIP标准化参数）
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(img_tensor.device)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(img_tensor.device)
            
            # 检查tensor值范围，判断是否已经归一化
            if img_tensor.min() >= -3 and img_tensor.max() <= 3:
                # 看起来像是已归一化的数据，进行反归一化
                img_tensor = img_tensor * std + mean
            
            # 确保值在[0, 1]范围内
            img_tensor = torch.clamp(img_tensor, 0, 1)
            
            # 转换为PIL图像
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            pil_image = Image.fromarray(img_np)
            
            return pil_image
            
        except Exception as e:
            print(f"Tensor转PIL失败: {e}")
            raise e
    
    def _process_vision_features_from_list(self, image_embeddings_list: list, 
                                         batch_size: int, 
                                         debug: bool = False) -> torch.Tensor:
        """从已经分组的特征列表中处理视觉特征"""
        batch_features = []
        for i, img_patches in enumerate(image_embeddings_list):
            # 全局平均池化
            img_feature = img_patches.mean(dim=0, keepdim=True)  # [1, hidden_dim]
            batch_features.append(img_feature)
            
            if debug:
                print(f"DEBUG: 图像 {i} 的patch数量: {img_patches.shape[0]}, 特征维度: {img_patches.shape[1]}")
        
        # 堆叠为批次
        image_features = torch.cat(batch_features, dim=0)  # [batch_size, hidden_dim]
        
        if debug:
            print(f"DEBUG: final image_features shape: {image_features.shape}")
        
        return image_features
    
    def _process_vision_features_projection_from_list(self, image_embeddings_list: list, batch_size: int, debug: bool = False) -> torch.Tensor:
        """方案1: 可学习线性投影（Projection Head） + L2 归一化 - 从list处理"""
        batch_features = []
        for img_patches in image_embeddings_list:
            img_feature = img_patches.mean(dim=0, keepdim=True)
            batch_features.append(img_feature)

        image_features = torch.cat(batch_features, dim=0)  # [batch_size, hidden_dim]
        image_features = self.vision_projection(image_features)
        
        # 使用相同的增强归一化策略
        normalized_features = F.normalize(image_features, p=2, dim=-1)
        
        if self.training:
            enhanced_features = normalized_features * 1.1
            enhanced_features = F.normalize(enhanced_features, p=2, dim=-1)
            return enhanced_features
        else:
            return normalized_features
    
    def _process_vision_features_attention_pooling_from_list(self, image_embeddings_list: list, batch_size: int, debug: bool = False) -> torch.Tensor:
        """数值稳定的GeM池化 + 稳健注意力机制 - 防止NaN/Inf"""
        batch_features = []
        
        # 安全获取hidden_dim
        if len(image_embeddings_list) == 0:
            raise ValueError("image_embeddings_list为空")
        
        # 检查第一个embeddings的形状
        first_emb = image_embeddings_list[0]
        if first_emb.numel() == 0:
            raise ValueError("第一个embedding为空")
        
        hidden_dim = first_emb.shape[-1]
        # if debug:
        #     print(f"DEBUG: 注意力池化 - hidden_dim: {hidden_dim}")
        #     print(f"DEBUG: 第一个embedding形状: {first_emb.shape}")
        
        # 为超声图像设计的稳健多尺度注意力池化
        for i, img_patches in enumerate(image_embeddings_list):
            # if debug:
            #     print(f"DEBUG: 处理图像 {i}, patches形状: {img_patches.shape}")
            
            # 安全检查
            if img_patches.numel() == 0:
                print(f"警告: 图像 {i} 的patches为空，使用默认特征")
                dummy_feature = torch.zeros(1, hidden_dim, device=img_patches.device, dtype=img_patches.dtype)
                batch_features.append(dummy_feature)
                continue
            
            seq_len = img_patches.shape[0]
            if seq_len == 0:
                print(f"警告: 图像 {i} 的序列长度为0，使用默认特征")
                dummy_feature = torch.zeros(1, hidden_dim, device=img_patches.device, dtype=img_patches.dtype)
                batch_features.append(dummy_feature)
                continue
            
            # 0. 数值稳定性预处理
            # 检查输入特征
            if torch.isnan(img_patches).any() or torch.isinf(img_patches).any():
                if debug:
                    print(f"  检测到输入NaN/Inf，进行修复")
                img_patches = self._fix_nan_inf_features(img_patches)
            
            # 输入特征裁剪（更保守的范围）
            img_patches = torch.clamp(img_patches, min=-5.0, max=5.0)
            
            # 1. 稳健的特征预处理
            try:
                # 使用稳健的归一化策略
                processed_patches = self._robust_feature_normalization(img_patches, debug)
                
                # 只有当patches足够多且维度正确时才应用注意力增强
                if seq_len > 3 and img_patches.shape[-1] == hidden_dim:
                    # 应用稳健的注意力模块
                    enhanced_patches = self._apply_robust_attention_modules(processed_patches, debug)
                else:
                    enhanced_patches = processed_patches
                    if debug:
                        print(f"  跳过注意力增强 (seq_len={seq_len})")
                        
            except Exception as e:
                if debug:
                    print(f"  特征预处理失败: {e}")
                enhanced_patches = img_patches
            
            # 2. 稳健的GeM池化（Generalized Mean Pooling）
            try:
                final_feature = self._gem_pooling_with_fallback(enhanced_patches, debug)
                
                # if debug:
                #     pass
                    # print(f"  最终特征形状: {final_feature.shape}")
                
                batch_features.append(final_feature)
                
            except Exception as e:
                # if debug:
                #     print(f"  GeM池化失败: {e}")
                # 最安全的备用方案
                safe_feature = self._safe_average_pooling(img_patches)
                batch_features.append(safe_feature)
        
        # 3. 批次特征聚合和最终检查
        try:
            batch_result = torch.cat(batch_features, dim=0)
            
            # 最终数值稳定性检查
            if torch.isnan(batch_result).any() or torch.isinf(batch_result).any():
                if debug:
                    print("  最终结果包含NaN/Inf，应用紧急修复")
                batch_result = self._fix_nan_inf_features(batch_result)
            
            return batch_result
            
        except Exception as e:
            if debug:
                print(f"  批次聚合失败: {e}")
            # 创建安全的零特征
            safe_batch = torch.zeros(batch_size, hidden_dim, 
                                   device=first_emb.device, dtype=first_emb.dtype)
            return safe_batch
    
    def _fix_nan_inf_features(self, features: torch.Tensor) -> torch.Tensor:
        """修复包含NaN/Inf的特征"""
        # 将NaN和Inf替换为零
        features = torch.where(torch.isnan(features) | torch.isinf(features), 
                             torch.zeros_like(features), features)
        # 重新归一化
        norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-8)
        return features / norm
    
    def _robust_feature_normalization(self, features: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """稳健的特征归一化"""
        # 使用稳健的统计量进行归一化
        # 计算中位数和稳健的标准差（MAD - Median Absolute Deviation）
        median = torch.median(features, dim=0, keepdim=True)[0]
        mad = torch.median(torch.abs(features - median), dim=0, keepdim=True)[0]
        
        # 避免除零
        mad = torch.clamp(mad, min=1e-6)
        
        # 稳健的z-score归一化
        robust_normalized = (features - median) / (1.4826 * mad)  # 1.4826是正态分布的MAD常数
        
        # 限制在合理范围内
        robust_normalized = torch.clamp(robust_normalized, min=-3.0, max=3.0)
        
        return robust_normalized
    
    def _apply_robust_attention_modules(self, features: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """应用稳健的注意力模块"""
        enhanced_features = features
        
        try:
            # 空间注意力（带数值检查）
            spatial_enhanced = self.spatial_attention(enhanced_features)
            if not (torch.isnan(spatial_enhanced).any() or torch.isinf(spatial_enhanced).any()):
                enhanced_features = spatial_enhanced
            elif debug:
                print("  空间注意力产生NaN/Inf，使用原始特征")
            
            # 通道注意力（带数值检查）
            channel_enhanced = self.channel_attention(enhanced_features)
            if not (torch.isnan(channel_enhanced).any() or torch.isinf(channel_enhanced).any()):
                enhanced_features = channel_enhanced
            elif debug:
                print("  通道注意力产生NaN/Inf，使用原始特征")
            
            # 特征增强（带数值检查）
            feature_enhanced = self.feature_enhancer(enhanced_features)
            if not (torch.isnan(feature_enhanced).any() or torch.isinf(feature_enhanced).any()):
                enhanced_features = feature_enhanced
            elif debug:
                print("  特征增强产生NaN/Inf，使用原始特征")
                
        except Exception as e:
            if debug:
                print(f"  注意力模块失败: {e}")
        
        return enhanced_features
    
    def _gem_pooling_with_fallback(self, features: torch.Tensor, debug: bool = False, p: float = 3.0) -> torch.Tensor:
        """GeM池化（Generalized Mean Pooling）带安全回退"""
        try:
            # GeM池化：(1/n * sum(x_i^p))^(1/p)
            # 数值稳定的实现
            
            # 首先确保输入为正值（通过shift和scale）
            shifted_features = features - features.min(dim=0, keepdim=True)[0] + 1e-6
            
            # 使用log-空间计算避免数值溢出
            # log(GeM) = (1/p) * log(mean(exp(p * log(x))))
            log_features = torch.log(torch.clamp(shifted_features, min=1e-8))
            
            # 计算加权对数平均
            scaled_log = p * log_features
            
            # 使用logsumexp技巧保证数值稳定性
            max_val = scaled_log.max(dim=0, keepdim=True)[0]
            stable_exp = torch.exp(scaled_log - max_val)
            mean_exp = stable_exp.mean(dim=0, keepdim=True)
            
            gem_log = (max_val + torch.log(torch.clamp(mean_exp, min=1e-8))) / p
            gem_result = torch.exp(gem_log)
            
            # 检查结果
            if torch.isnan(gem_result).any() or torch.isinf(gem_result).any():
                if debug:
                    print("  GeM池化产生NaN/Inf，使用平均池化")
                return self._safe_average_pooling(features)
            
            # 最终归一化
            norm = torch.norm(gem_result, p=2, dim=-1, keepdim=True)
            norm = torch.clamp(norm, min=1e-8)
            return gem_result / norm
            
        except Exception as e:
            # if debug:
            #     print(f"  GeM池化失败: {e}")
            return self._safe_average_pooling(features)
    
    def _safe_average_pooling(self, features: torch.Tensor) -> torch.Tensor:
        """安全的平均池化"""
        # 简单但稳健的平均池化
        avg_feature = features.mean(dim=0, keepdim=True)
        
        # 检查和修复
        if torch.isnan(avg_feature).any() or torch.isinf(avg_feature).any():
            avg_feature = torch.zeros_like(avg_feature)
        
        # 归一化
        norm = torch.norm(avg_feature, p=2, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-8)
        return avg_feature / norm

    def _process_vision_features_top_k_from_list(self, image_embeddings_list: list, batch_size: int, debug: bool = False) -> torch.Tensor:
        """方案3: Top-K Token 选择 + 投影 - 从list处理"""
        k = 64  # 选取的token数量
        batch_features = []
        for img_patches in image_embeddings_list:
            actual_k = min(k, img_patches.shape[0])  # 确保k不超过实际patch数量
            norms = torch.norm(img_patches, dim=-1)
            top_k_indices = torch.topk(norms, k=actual_k, dim=0).indices
            top_k_features = img_patches[top_k_indices]
            projected_features = self.vision_projection(top_k_features)
            pooled_feature = projected_features.mean(dim=0, keepdim=True)
            batch_features.append(pooled_feature)
        
        return torch.cat(batch_features, dim=0)

    def _process_vision_features_cls_token_from_list(self, image_embeddings_list: list, batch_size: int, debug: bool = False) -> torch.Tensor:
        """方案4: CLS Token 或聚合 Token 表示 - 从list处理"""
        batch_features = []
        for img_patches in image_embeddings_list:
            cls_token = img_patches[0]  # 假设CLS token在第一个位置
            batch_features.append(cls_token.unsqueeze(0))
        
        return torch.cat(batch_features, dim=0)
    
    # 保留原来的方法作为备用
    def _process_vision_features(self, image_embeddings: torch.Tensor, 
                               image_grid_thw: torch.Tensor, 
                               batch_size: int, 
                               debug: bool = False) -> torch.Tensor:
        """处理视觉特征，正确分组和池化（保留原方法作为备用）"""
        hidden_dim = image_embeddings.shape[-1]
        total_patches = image_embeddings.shape[0]
        
        if debug:
            print(f"DEBUG: image_embeddings shape: {image_embeddings.shape}")
            print(f"DEBUG: total_patches: {total_patches}, batch_size: {batch_size}")
        
        # 计算每张图像的patch数量
        patches_per_image = []
        for i in range(batch_size):
            t, h, w = image_grid_thw[i].tolist()
            num_patches = t * h * w
            patches_per_image.append(num_patches)
            
            if debug:
                print(f"DEBUG: 图像 {i} 的patch数量: {num_patches} (T={t}, H={h}, W={w})")
        
        expected_total = sum(patches_per_image)
        
        if debug:
            print(f"DEBUG: expected_total: {expected_total}, actual_total: {total_patches}")
        
        # 如果总数不匹配，使用平均分配策略
        if expected_total != total_patches:
            if debug:
                print(f"WARNING: patches总数不匹配，使用平均分配")
            patches_per_img = total_patches // batch_size
            patches_per_image = [patches_per_img] * batch_size
            # 处理余数
            remainder = total_patches % batch_size
            for i in range(remainder):
                patches_per_image[i] += 1
            
            if debug:
                print(f"DEBUG: 调整后patches_per_image: {patches_per_image}")
        
        # 重新组织特征
        start_idx = 0
        batch_features = []
        for num_patches in patches_per_image:
            # 提取这张图像的所有patch特征
            img_patches = image_embeddings[start_idx:start_idx + num_patches]
            # 全局平均池化
            img_feature = img_patches.mean(dim=0, keepdim=True)
            batch_features.append(img_feature)
            start_idx += num_patches
        
        # 堆叠为批次
        image_features = torch.cat(batch_features, dim=0)  # [batch_size, hidden_dim]
        
        if debug:
            print(f"DEBUG: final image_features shape: {image_features.shape}")
        
        return image_features
    
    def encode_text(self, text_inputs: Union[Dict[str, torch.Tensor], list, str, torch.Tensor]) -> torch.Tensor:
        """
        编码文本
        Args:
            text_inputs: 文本输入，可以是tokenized的字典、文本列表、单个文本或tensor
        Returns:
            text_features: 归一化的文本特征
        """
        # 处理不同类型的输入
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]
        
        if isinstance(text_inputs, torch.Tensor):
            # 如果是tensor，说明是CLIP tokenizer的结果，我们需要重新处理
            # 这种情况下我们无法恢复原始文本，只能跳过或报错
            raise ValueError("暂不支持tensor类型的文本输入，请传入原始文本")
        
        if isinstance(text_inputs, list):
            # 文本列表，需要用BERT tokenizer处理
            # text_model_name = self.config.get('text_encoder', {}).get('model_name', 'bert-base-chinese')
            text_model_name = self.config['text_encoder']['model_name']
            # 检查是否已经初始化了tokenizer
            if not hasattr(self, '_tokenizer') or self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            
            text_inputs = self._tokenizer(
                text_inputs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # 移动到正确设备
            device = next(self.text_encoder.parameters()).device
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        elif isinstance(text_inputs, dict):
            # 已经tokenized的输入，确保在正确设备上
            device = next(self.text_encoder.parameters()).device
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        else:
            raise ValueError(f"不支持的文本输入类型: {type(text_inputs)}")
        
        # BERT文本编码
        with torch.no_grad():
            outputs = self.text_encoder(**text_inputs)
        
        # 使用[CLS] token的表示
        text_features = outputs.last_hidden_state[:, 0]  # [B, hidden_size]
        
        # 投影到统一维度
        text_features = self.text_projection(text_features)
        
        # 与图像特征使用相同的增强归一化策略
        normalized_features = F.normalize(text_features, p=2, dim=-1)
        
        if self.training:
            enhanced_features = normalized_features * 1.1
            enhanced_features = F.normalize(enhanced_features, p=2, dim=-1)
            return enhanced_features
        else:
            return normalized_features
    
    def forward(self, images, texts, debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            images: 图像输入
            texts: 文本输入
            debug: 是否输出调试信息
        Returns:
            image_features, text_features
        """
        # 应用logit_scale约束
        self._apply_logit_scale_constraints()
        
        # 编码图像和文本
        image_features = self.encode_image(images, debug=debug)
        text_features = self.encode_text(texts)
        
        # 验证特征有效性
        if torch.isnan(image_features).any() or torch.isinf(image_features).any():
            print("WARNING: 图像特征包含NaN/Inf值")
            image_features = torch.where(torch.isnan(image_features) | torch.isinf(image_features),
                                       torch.zeros_like(image_features), image_features)
            image_features = F.normalize(image_features, p=2, dim=-1)
        
        if torch.isnan(text_features).any() or torch.isinf(text_features).any():
            print("WARNING: 文本特征包含NaN/Inf值")
            text_features = torch.where(torch.isnan(text_features) | torch.isinf(text_features),
                                      torch.zeros_like(text_features), text_features)
            text_features = F.normalize(text_features, p=2, dim=-1)
        
        return image_features, text_features
    
    @property
    def tokenizer(self):
        """获取tokenizer"""
        text_model_name = self.config['text_encoder']['model_name']
        # text_model_name = self.config.get('text_encoder', {}).get('model_name', 'bert-base-chinese')
        return AutoTokenizer.from_pretrained(text_model_name)
