import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Union
import math
from transformers import (
    AutoImageProcessor,
    ConvNextModel,
    RobertaTokenizer,
    RobertaModel,
)


class ConvNeXtCLIPModel(nn.Module):
    """基于 ConvNeXt(图像) + RoBERTa(文本) 的轻量 CLIP 模型。

    约定：
    - config 结构参考 qwen_vl_clip_opt_6.json，但将 vision/text encoder 改为 ConvNeXt + RoBERTa。
    - 暴露 encode_image / encode_text / forward 接口；forward 返回相似度矩阵与特征。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        model_cfg: Dict[str, Any] = config.get("model", config)
        
        self.embed_dim: int = model_cfg.get("embed_dim", 768)
        self.device_name: str = model_cfg.get("device", "auto")
        # 设备放置
        if self.device_name == "auto":
            self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        # self.to(self.device_name)

        # 图像全局特征聚合策略：auto | pooler | gap
        self.vision_pooling: str = model_cfg.get("vision_pooling", "auto")

        # ------------ Vision encoder (ConvNeXt) ------------
        vision_cfg: Dict[str, Any] = model_cfg.get("vision_encoder", {})
        self.vision_model_path: str = vision_cfg.get("model_path")
        vision_dim: int = vision_cfg.get("vision_dim", 768)

        if not self.vision_model_path:
            raise ValueError("vision_encoder.model_path 不能为空")

        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_model_path)
        self.vision_encoder = ConvNextModel.from_pretrained(self.vision_model_path)
        self.image_projection = nn.Linear(vision_dim, self.embed_dim, bias=False)
        # 多视角输出：为全局与局部表征分别提供投影头（与融合头并存）
        self.image_projection_global = nn.Linear(vision_dim, self.embed_dim, bias=False)
        self.image_projection_local = nn.Linear(vision_dim, self.embed_dim, bias=False)
        # 当使用 GAP 作为兜底或强制策略时，添加 LN 以更贴近 pooler 的数值风格
        self._gap = nn.AdaptiveAvgPool2d((1, 1))
        self._gap_ln = nn.LayerNorm(vision_dim, eps=1e-6)
        
        # ------------ 新增：实验方案配置 ------------
        self.experiment_config = model_cfg.get("experiment_config", {})
        # 掩码相关可调超参
        self.mask_min_area_ratio: float = float(self.experiment_config.get("mask_min_area_ratio", 0.001))
        self.mask_soften_kernel: int = int(self.experiment_config.get("mask_soften_kernel", 3))
        self.gate_min: float = float(self.experiment_config.get("gate_min", 0.1))
        self.gate_max: float = float(self.experiment_config.get("gate_max", 0.9))
        self.local_global_alpha: float = float(self.experiment_config.get("local_global_alpha", 0.5))
        
        # 版本1：Attention Side-Head for Mask Supervision
        self.use_attention_head = self.experiment_config.get("use_attention_head", False)
        if self.use_attention_head:
            # 注意力输出分支：将特征映射转换为attention map
            self.attention_head = nn.Sequential(
                nn.Conv2d(vision_dim, vision_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(vision_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(vision_dim // 2, 1, kernel_size=1),
                nn.Sigmoid()  # 输出0-1的attention权重
            )
            # print("  - 启用Attention Side-Head进行mask监督")
        
        # 版本2：Gating机制
        self.use_gating = self.experiment_config.get("use_gating", False)
        if self.use_gating:
            # Gate模块：决定local vs global的融合权重
            self.fusion_gate = nn.Sequential(
                nn.Linear(vision_dim * 2, vision_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(vision_dim, 1),
                nn.Sigmoid()
            )
            # print("  - 启用Gating机制进行自适应融合")
        
        # 版本3：多尺度/局部patch
        self.use_multi_scale = self.experiment_config.get("use_multi_scale", False)
        if self.use_multi_scale:
            # 多尺度卷积分支
            self.multi_scale_branch = nn.ModuleList([
                nn.Conv2d(vision_dim, vision_dim, kernel_size=1, stride=1),  # 细粒度
                nn.Conv2d(vision_dim, vision_dim, kernel_size=3, stride=1, padding=1),  # 中等
                nn.Conv2d(vision_dim, vision_dim, kernel_size=5, stride=1, padding=2),  # 粗粒度
            ])
            self.scale_fusion = nn.Conv2d(vision_dim * 3, vision_dim, kernel_size=1)
            # print("  - 启用多尺度特征提取")
        
        # 版本4：稀疏注意力正则化
        self.use_sparse_attention = self.experiment_config.get("use_sparse_attention", False)
        self.attention_sparsity_weight = self.experiment_config.get("attention_sparsity_weight", 0.01)
        if self.use_sparse_attention:
            # print(f"  - 启用稀疏注意力正则化，权重: {self.attention_sparsity_weight}")
            pass

        # 优化方法
        # self.attn_layer = nn.MultiheadAttention(embed_dim=vision_dim, num_heads=8, dropout=0.1, batch_first=True)
        # # self.attn_layer = nn.MultiheadAttention(embed_dim=vision_dim, num_heads=12, dropout=0.1, batch_first=True)
        # self.fusion_ln = nn.LayerNorm(vision_dim) # 额外LN for fused
        # self.dropout = nn.Dropout(0.1) # 全局dropout防止过拟合
        # 优化方法 - 防过拟合增强
        self.attn_layer = nn.MultiheadAttention(embed_dim=vision_dim, num_heads=8, dropout=0.15, batch_first=True)
        # 原始代码：self.attn_layer = nn.MultiheadAttention(embed_dim=vision_dim, num_heads=8, dropout=0.1, batch_first=True)
        # 优化：增加dropout从0.1到0.15，增强正则化防止过拟合
        
        self.fusion_ln = nn.LayerNorm(vision_dim) # 额外LN for fused
        self.dropout = nn.Dropout(0.15) # 全局dropout防止过拟合
        # 原始代码：self.dropout = nn.Dropout(0.1)
        # 优化：增加dropout从0.1到0.15，增强正则化
        
        # 新增防过拟合层
        self.feature_dropout = nn.Dropout(0.1)
        self.feature_ln = nn.LayerNorm(self.embed_dim)
        self.attention_dropout = nn.Dropout(0.1)

        # 位置编码(用于硬编码位置)
        self.pe_layer = self._positional_encoding(vision_dim) # # [1, HW_max, C]，假设H*W固定或动态生成
        # ------------ Text encoder (RoBERTa) ------------
        text_cfg: Dict[str, Any] = model_cfg.get("text_encoder", {})
        self.text_model_name: str = text_cfg.get("model_name")
        self.max_text_length: int = text_cfg.get("max_length", 128)

        if not self.text_model_name:
            raise ValueError("text_encoder.model_name 不能为空")

        self.tokenizer = RobertaTokenizer.from_pretrained(self.text_model_name)
        self.text_encoder = RobertaModel.from_pretrained(self.text_model_name)
        text_hidden: int = self.text_encoder.config.hidden_size
        self.text_projection = nn.Linear(text_hidden, self.embed_dim, bias=False)

        # 温度参数（可选，与标准 CLIP 一致）
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        # 设备放置 - 在所有组件创建完成后执行
        self.to(self.device_name)


    # -----------------------------------------------------
    # 编码与前向
    # -----------------------------------------------------
    @torch.no_grad()
    def preprocess_images(self, images):
        """将图像(PIL 或 PIL 列表)批处理为 pixel_values 张量。"""
        inputs = self.image_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device_name)
        return pixel_values
    
    def _positional_encoding(self, dim: int, max_len: int = 1024):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, dim]
        return nn.Parameter(pe, requires_grad=False).to(self.device_name)
    # 原始可用的图像编码器---只有结构表征特征
    # def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
    #     """输入 pixel_values，输出 L2 归一化后的图像特征。"""
    #     outputs = self.vision_encoder(pixel_values=pixel_values)
    #     # 三种策略：
    #     # - pooler: 强制使用 pooler_output，不存在则回退 GAP+LN
    #     # - gap:    强制使用 GAP+LN
    #     # - auto:   有 pooler 用 pooler，否则 GAP+LN
    #     use_pooler = (
    #         self.vision_pooling == "pooler" or
    #         (self.vision_pooling == "auto" and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None)
    #     )

    #     if use_pooler and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
    #         pooled: torch.Tensor = outputs.pooler_output  # [B, vision_dim]
    #     else:
    #         last: torch.Tensor = outputs.last_hidden_state  # [B, C, H, W]
    #         pooled_hw = self._gap(last).squeeze(-1).squeeze(-1)  # [B, C]
    #         pooled = self._gap_ln(pooled_hw)
    #     image_features = self.image_projection(pooled)
    #     image_features = F.normalize(image_features, dim=-1)
    #     return image_features

    def encode_image(self, pixel_values: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """输入pixel_values，输出L2归一化之后的图像特征(病理表征+结构表征融合)
        可选参数 masks: [B,1,H,W] 取值{0,1}，用于局部加权。
        若提供，将以mask为权重对空间特征做加权池化，并与全局表征做注意力融合。
        
        返回:
            如果use_attention_head=True: (image_features, attention_map)
            否则: image_features
        """
        outputs = self.vision_encoder(pixel_values=pixel_values)
        last = outputs.last_hidden_state #[B, C, H, W]

        B, C, H, W = last.shape
        seq_len = H * W
        
        # 版本1：生成attention map用于mask监督
        attention_map = None
        if self.use_attention_head:
            attention_map = self.attention_head(last)  # [B, 1, H, W]
        
        # 版本3：多尺度特征提取
        if self.use_multi_scale:
            multi_scale_features = []
            for scale_conv in self.multi_scale_branch:
                scale_feat = scale_conv(last)
                multi_scale_features.append(scale_feat)
            # 拼接并融合多尺度特征
            multi_scale_concat = torch.cat(multi_scale_features, dim=1)  # [B, C*3, H, W]
            last = self.scale_fusion(multi_scale_concat)  # [B, C, H, W]
        
        # 结构表征(局部空间结构信息)
        struct_features = last.flatten(2).transpose(1, 2) #[B, C, H*W]?[B, HW, C]?
        # 添加位置编码：截取到seq_len
        pe = self.pe_layer[:, :seq_len, :] # [1, HW, C]
        struct_features = struct_features + pe.expand(B, -1, -1)
        struct_features = self.dropout(struct_features) # 防止过拟合

        # 病理表征(全局embedding)
        pooled_hw = self._gap(last).squeeze(-1).squeeze(-1)  # [B, C]

        # 若提供掩码或使用attention map：局部表征(局部mask加权聚合)
        effective_mask = masks  # 默认使用输入的mask
        if self.use_attention_head and attention_map is not None and masks is None:
            # 如果没有输入mask但有attention map，使用attention map作为soft mask
            effective_mask = attention_map
        
        # 添加mask有效性检查
        mask_valid = False
        if effective_mask is not None and isinstance(effective_mask, torch.Tensor) and effective_mask.numel() > 0:
            # print(f"DEBUG: 开始处理mask，原始shape: {effective_mask.shape}")
            try:
                # 统一设备与数据类型
                if effective_mask.device != last.device or effective_mask.dtype != last.dtype:
                    effective_mask = effective_mask.to(device=last.device, dtype=last.dtype)

                # 规范维度到 [B,1,H,W]
                if effective_mask.dim() == 4:
                    pass
                elif effective_mask.dim() == 3:
                    effective_mask = effective_mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
                elif effective_mask.dim() == 2:
                    effective_mask = effective_mask.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
                else:
                    raise ValueError(f"不支持的mask维度: {effective_mask.dim()}")

                # 批大小对齐
                if effective_mask.shape[0] != B:
                    if effective_mask.shape[0] == 1:
                        effective_mask = effective_mask.expand(B, -1, -1, -1)
                    elif effective_mask.shape[0] > B:
                        effective_mask = effective_mask[:B]
                    else:
                        repeat_times = B - effective_mask.shape[0]
                        last_mask = effective_mask[-1:].repeat(repeat_times, 1, 1, 1)
                        effective_mask = torch.cat([effective_mask, last_mask], dim=0)

                # 无条件重采样到 (H, W)，避免上游尺寸不一致导致的隐式错误
                effective_mask = torch.nn.functional.interpolate(effective_mask, size=(H, W), mode='nearest')

                effective_mask = effective_mask.clamp(0, 1)

                # 掩码软化（盒式平滑）
                if self.mask_soften_kernel and self.mask_soften_kernel >= 3:
                    k = int(self.mask_soften_kernel)
                    if k % 2 == 0:
                        k += 1
                    pad = k // 2
                    kernel = torch.ones((1, 1, k, k), device=effective_mask.device, dtype=effective_mask.dtype) / (k * k)
                    eff_pad = torch.nn.functional.pad(effective_mask, (pad, pad, pad, pad), mode='replicate')
                    eff_blur = torch.nn.functional.conv2d(eff_pad, kernel)
                    effective_mask = eff_blur[:, :, pad:-pad, pad:-pad].clamp(0, 1)

                # 再次确保与特征图一致（防御性编程）
                if effective_mask.shape[-2:] != (H, W):
                    effective_mask = torch.nn.functional.interpolate(effective_mask, size=(H, W), mode='nearest')

                # 加权平均: sum(Feat * mask) / (sum(mask)+eps)
                # 在乘法前做一次形状断言与一次性自动修复
                if effective_mask.shape != (B, 1, H, W):
                    # print(f"Warning: 掩码形状不规范，尝试自动修复: {effective_mask.shape} -> ({B},1,{H},{W})")
                    effective_mask = effective_mask.reshape(B, 1, H, W)

                masked_sum = (last * effective_mask).sum(dim=(-1, -2))  # [B,C]
                mask_area = effective_mask.sum(dim=(-1, -2)).clamp(min=1e-6)  # [B,1]
                local_feat = masked_sum / mask_area  # [B,C]

                # 小面积掩码降权
                mask_area_keepdim = effective_mask.sum(dim=(-1, -2), keepdim=True)
                total_area = torch.tensor(float(H * W), device=effective_mask.device, dtype=effective_mask.dtype)
                area_ratio = mask_area_keepdim / total_area
                small_mask_flag = (area_ratio < self.mask_min_area_ratio).to(effective_mask.dtype)
                safe_den = max(self.mask_min_area_ratio, 1e-6)
                scale = torch.minimum(area_ratio / safe_den, torch.ones_like(area_ratio))
                effective_mask = effective_mask * (small_mask_flag * scale + (1 - small_mask_flag))

                # 版本2：使用gating机制融合
                if self.use_gating:
                    concat_feat = torch.cat([local_feat, pooled_hw], dim=1)  # [B, C*2]
                    gate_weight = self.fusion_gate(concat_feat)  # [B, 1]
                    gate_weight = gate_weight.clamp(min=self.gate_min, max=self.gate_max)
                    pooled_hw = gate_weight * local_feat + (1 - gate_weight) * pooled_hw
                else:
                    alpha = self.local_global_alpha
                    pooled_hw = (1 - alpha) * pooled_hw + alpha * local_feat

                mask_valid = True
                # print(f"DEBUG: mask处理成功，最终shape: {effective_mask.shape}")
            except Exception as e:
                # print(f"Warning: mask processing failed: {e}")
                # print(f"  - effective_mask shape: {effective_mask.shape if 'effective_mask' in locals() else 'N/A'}")
                # print(f"  - last shape: {last.shape}")
                # print(f"  - B, C, H, W: {B}, {C}, {H}, {W}")
                effective_mask = None
                mask_valid = False
                # print(f"  - 回退到全局特征处理")
        else:
            # 未提供有效mask时，构造一个温和的局部候选：使用结构序列简单平均作为局部代表
            # 这样可以在无mask数据时仍然给出local embedding以参与多视角损失
            local_feat = struct_features.mean(dim=1)  # [B, C]
                
        patho_features = self._gap_ln(pooled_hw).unsqueeze(1)  # [B, 1, C]

        # ===== Cross-Attention 融合 =====
        fused, attn_weights = self.attn_layer(patho_features, struct_features, struct_features) #[B,1,C]
        fused = fused.squeeze(1)  # [B, C]

        # 残差连接+线性层
        fused = self.fusion_ln(fused + pooled_hw)
        fused = F.gelu(fused) # 添加非线性激活，感觉区分度应该会大一些
        fused = self.dropout(fused)

        # 新增防过拟合处理
        fused = self.feature_dropout(fused)
        fused = self.feature_ln(fused)

        # ===== 投影到共享空间 =====
        fused = self.image_projection(fused)  # [B, D]
        fused = F.normalize(fused, dim=-1)
        image_features = fused

        # 全局与局部多视角嵌入
        global_token = self._gap_ln(pooled_hw)
        image_global = self.image_projection_global(global_token)
        image_global = F.normalize(image_global, dim=-1)

        local_token = self._gap_ln(local_feat) if 'local_feat' in locals() else self._gap_ln(pooled_hw)
        image_local = self.image_projection_local(local_token)
        image_local = F.normalize(image_local, dim=-1)
        
        # 版本1：返回attention map用于监督
        if self.use_attention_head and self.training:
            return image_features, attention_map
        else:
            return image_features
    def encode_text(self, texts_or_tokens) -> torch.Tensor:
        """编码文本或已分词的 tokens 映射，输出 L2 归一化文本特征。
        接受：
          - List[str]
          - Dict[str, Tensor]
          - transformers.BatchEncoding 或任意拥有 .keys()/.items() 的映射类型
        """
        if isinstance(texts_or_tokens, list) and (len(texts_or_tokens) == 0 or isinstance(texts_or_tokens[0], str)):
            encoded = self.tokenizer(
                texts_or_tokens,
                padding=True,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device_name) for k, v in encoded.items()}
        elif torch.is_tensor(texts_or_tokens):
            # 直接把 [B, L] 的 input_ids 张量作为输入，自动生成 attention_mask
            input_ids = texts_or_tokens.to(self.device_name)
            attention_mask = (input_ids != 0).long()
            encoded = {"input_ids": input_ids, "attention_mask": attention_mask}
        elif hasattr(texts_or_tokens, 'input_ids'):
            # 兼容具有属性的对象（如 BatchEncoding 子类）
            input_ids = getattr(texts_or_tokens, 'input_ids')
            attention_mask = getattr(texts_or_tokens, 'attention_mask', None)
            token_type_ids = getattr(texts_or_tokens, 'token_type_ids', None)
            mapping = {"input_ids": input_ids}
            if attention_mask is not None:
                mapping["attention_mask"] = attention_mask
            if token_type_ids is not None:
                mapping["token_type_ids"] = token_type_ids
            encoded = {k: (v.to(self.device_name) if torch.is_tensor(v) else torch.as_tensor(v).to(self.device_name)) for k, v in mapping.items()}
        else:
            # 兼容 transformers.BatchEncoding 或通用映射类型
            mapping = None
            if hasattr(texts_or_tokens, 'to_dict'):
                try:
                    mapping = texts_or_tokens.to_dict()
                except Exception:
                    mapping = None
            if mapping is None and hasattr(texts_or_tokens, 'data') and isinstance(texts_or_tokens.data, dict):
                mapping = texts_or_tokens.data
            if mapping is None and hasattr(texts_or_tokens, 'keys') and hasattr(texts_or_tokens, 'items'):
                mapping = {k: v for k, v in texts_or_tokens.items()}

            if mapping is not None:
                encoded = {k: (v.to(self.device_name) if torch.is_tensor(v) else torch.as_tensor(v).to(self.device_name)) for k, v in mapping.items()}
            else:
                raise TypeError("encode_text 需要 List[str] 或 token 映射({input_ids, attention_mask,...})")

        outputs = self.text_encoder(**encoded)
        # 优先使用 pooler_output；若不可用则取 CLS 表示
        if outputs.pooler_output is not None:
            pooled: torch.Tensor = outputs.pooler_output  # [B, hidden]
        else:
            pooled = outputs.last_hidden_state[:, 0]  # [B, hidden]

        text_features = self.text_projection(pooled)
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    def forward(
        self,
        pixel_values: torch.Tensor,
        texts: Optional[Union[List[str], Dict[str, torch.Tensor]]] = None,
        text_features: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """前向：计算图像特征、文本特征与相似度。

        - 若提供 texts，将内部编码文本；否则使用传入的 text_features。
        - 返回：
          - image_features: [B, D]
          - text_features:  [N, D]
          - logits_per_image: [B, N]
          - logits_per_text:  [N, B]
          - attention_map: [B, 1, H, W] (仅在use_attention_head=True且training时)
        """
        # 编码图像，可能返回attention map
        encode_result = self.encode_image(pixel_values, masks=masks)
        
        # 处理返回值
        if isinstance(encode_result, tuple):
            image_features, attention_map = encode_result
        else:
            image_features = encode_result
            attention_map = None

        if text_features is None:
            if texts is None:
                raise ValueError("forward 需要提供 texts 或已编码的 text_features 之一")
            text_features = self.encode_text(texts)

        #######################防止过拟合方法1.添加一个温度钳制看看能不能防止过拟合#########################
        with torch.no_grad():
            self.logit_scale.clamp_(min=-4.6,max=4.6)
        #########################    
        # 相似度（温度缩放）
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # 额外返回全局与局部嵌入，供多视角损失使用
        # 为避免重复计算，这里简单地再次调用 encode_image 的内部缓存不可行，因此在上面构造时一并产生
        # 为了最小改动，这里通过再次计算得到 global/local（开销可接受）；若需进一步优化，可重构 encode_image 返回多视角
        # 复用上一步中的 pooled_hw/local_feat 需要调整 encode_image 的返回签名，暂不改动，转而在此重算
        with torch.no_grad():
            outputs_tmp = self.vision_encoder(pixel_values=pixel_values)
            last_tmp = outputs_tmp.last_hidden_state
            pooled_tmp = self._gap(last_tmp).squeeze(-1).squeeze(-1)
        image_global = F.normalize(self.image_projection_global(self._gap_ln(pooled_tmp)), dim=-1)

        # 计算一个稳健的local（无mask则用结构平均）
        Btmp, Ctmp, Htmp, Wtmp = last_tmp.shape
        struct_tmp = last_tmp.flatten(2).transpose(1, 2)
        if masks is not None and isinstance(masks, torch.Tensor) and masks.numel() > 0:
            try:
                m = masks
                if m.device != last_tmp.device or m.dtype != last_tmp.dtype:
                    m = m.to(device=last_tmp.device, dtype=last_tmp.dtype)
                if m.dim() == 3:
                    m = m.unsqueeze(1)
                if m.shape[0] != Btmp:
                    if m.shape[0] == 1:
                        m = m.expand(Btmp, -1, -1, -1)
                    elif m.shape[0] > Btmp:
                        m = m[:Btmp]
                    else:
                        m = torch.cat([m, m[-1:].repeat(Btmp - m.shape[0], 1, 1, 1)], dim=0)
                m = torch.nn.functional.interpolate(m, size=(Htmp, Wtmp), mode='nearest').clamp(0, 1)
                local_tmp = (last_tmp * m).sum(dim=(-1, -2)) / m.sum(dim=(-1, -2)).clamp(min=1e-6)
            except Exception:
                local_tmp = struct_tmp.mean(dim=1)
        else:
            local_tmp = struct_tmp.mean(dim=1)
        image_local = F.normalize(self.image_projection_local(self._gap_ln(local_tmp)), dim=-1)

        result = {
            "image_features": image_features,
            "text_features": text_features,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "image_global": image_global,
            "image_local": image_local,
        }
        
        # 如果有attention map，添加到返回结果中
        if attention_map is not None:
            result["attention_map"] = attention_map
            
        return result


def build_convnext_clip_from_config(config: Dict[str, Any]) -> ConvNeXtCLIPModel:
    """工厂函数：从配置构建 ConvNeXtCLIPModel。"""
    return ConvNeXtCLIPModel(config)


class ConvNeXtCLIPTrainingWrapper(nn.Module):
    """适配 Trainer 的训练包装器。

    训练期前向接口: forward(images, text_inputs, debug=False) -> (image_features, text_features)
    - images: 支持 torch.Tensor[B,C,H,W] 或 PIL(PIL列表)；若为PIL则走processor；若为Tensor则直接作为 pixel_values 使用。
    - text_inputs: 支持 List[str] 或 token 字典。
    """

    def __init__(self, core: ConvNeXtCLIPModel):
        super().__init__()
        self.core = core
        # 透出 tokenizer 供外部访问
        self.tokenizer = core.tokenizer
    # **kwargs用于兼容peft的调用方式！
    # def forward(self, images, text_inputs, debug: bool = False, **kwargs):
    #     # 兼容peft的调用方式！
    #     # 如果没有通过关键字参数去传递，尝试从kwargs中获取
    #     if images is None and 'images' in kwargs:
    #         images = kwargs['images']
    #     if text_inputs is None and 'text_inputs' in kwargs:
    #         text_inputs = kwargs['text_inputs']
    #     # 确保必要参数存在
    #     if images is None:
    #         raise ValueError("images 参数不能为空")
    #     if text_inputs is None:
    #         raise ValueError("text_inputs 参数不能为空")

    #     # 图像处理
    #     if isinstance(images, torch.Tensor):
    #         pixel_values = images.to(self.core.device_name)
    #     else:
    #         pixel_values = self.core.preprocess_images(images)

    #     # 文本编码
    #     text_features = self.core.encode_text(text_inputs)
    #     # 图像编码
    #     image_features = self.core.encode_image(pixel_values)

    #     return image_features, text_features
    def forward(self, *args, **kwargs):
        # 兼容peft的调用方式！
        
        # 添加更详细的调试信息
        # print(f"DEBUG: args 长度 = {len(args)}")
        # print(f"DEBUG: args 类型 = {[type(arg) for arg in args]}")
        # print(f"DEBUG: kwargs 键 = {list(kwargs.keys())}")
        
        # 检查是否有图像/掩码相关的参数
        image_keys = ['images', 'pixel_values', 'image', 'input_images']
        text_keys = ['text_inputs', 'input_ids', 'texts', 'text', 'text_tokens']
        mask_keys = ['masks', 'mask']
        
        # print(f"DEBUG: 查找图像参数...")
        for key in image_keys:
            if key in kwargs:
                # print(f"DEBUG: 找到图像参数 '{key}': {type(kwargs[key])}")
                pass
        
        # print(f"DEBUG: 查找文本参数...")
        for key in text_keys:
            if key in kwargs:
                # print(f"DEBUG: 找到文本参数 '{key}': {type(kwargs[key])}")
                pass
        
        # 处理位置参数
        if len(args) >= 1:
            images = args[0]
            # print(f"DEBUG: 从 args[0] 获取 images: {type(images)}")
        elif 'images' in kwargs:
            images = kwargs['images']
            # print(f"DEBUG: 从 kwargs['images'] 获取 images: {type(images)}")
        elif 'pixel_values' in kwargs:
            images = kwargs['pixel_values']
            # print(f"DEBUG: 从 kwargs['pixel_values'] 获取 images: {type(images)}")
        else:
            images = None
            # print("DEBUG: images 为 None")
            
        if len(args) >= 2:
            text_inputs = args[1]
            # print(f"DEBUG: 从 args[1] 获取 text_inputs: {type(text_inputs)}")
        elif 'text_inputs' in kwargs:
            text_inputs = kwargs['text_inputs']
            # print(f"DEBUG: 从 kwargs['text_inputs'] 获取 text_inputs: {type(text_inputs)}")
        elif 'input_ids' in kwargs:
            text_inputs = kwargs['input_ids']
            # print(f"DEBUG: 从 kwargs['input_ids'] 获取 text_inputs: {type(text_inputs)}")
        else:
            text_inputs = None
            # print("DEBUG: text_inputs 为 None")
            
        # 从 kwargs 中获取 debug 参数
        debug = kwargs.get('debug', False)
        
        # 可选掩码
        masks = None
        for key in mask_keys:
            if key in kwargs:
                masks = kwargs[key]
                break
        
        # 确保必要参数存在
        if images is None:
            # print(f"DEBUG: 所有可用的参数:")
            # print(f"  - args: {args}")
            # print(f"  - kwargs: {kwargs}")
            raise ValueError("images 参数不能为空")
        if text_inputs is None:
            raise ValueError("text_inputs 参数不能为空")
        
        # 图像处理
        if isinstance(images, torch.Tensor):
            pixel_values = images.to(self.core.device_name)
        else:
            pixel_values = self.core.preprocess_images(images)

        # 文本编码
        text_features = self.core.encode_text(text_inputs)
        # 图像编码
        if masks is not None and isinstance(masks, torch.Tensor):
            masks = masks.to(self.core.device_name)
        
        # 调用core的forward方法获取完整结果
        result = self.core.forward(pixel_values, text_inputs, masks=masks)
        
        # 提取特征
        image_features = result['image_features']
        text_features = result['text_features']
        
        # 返回格式兼容旧代码，但支持扩展返回attention map
        if 'attention_map' in result and self.core.training:
            # 训练时返回额外的attention map用于监督
            return image_features, text_features, result['attention_map']
        else:
            return image_features, text_features