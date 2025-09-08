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
        # 当使用 GAP 作为兜底或强制策略时，添加 LN 以更贴近 pooler 的数值风格
        self._gap = nn.AdaptiveAvgPool2d((1, 1))
        self._gap_ln = nn.LayerNorm(vision_dim, eps=1e-6)

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

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """输入pixel_values，输出L2归一化之后的图像特征(病理表征+结构表征融合)"""
        outputs = self.vision_encoder(pixel_values=pixel_values)
        last = outputs.last_hidden_state #[B, C, H, W]

        B, C, H, W = last.shape
        seq_len = H * W
        # 结构表征(局部空间结构信息)
        struct_features = last.flatten(2).transpose(1, 2) #[B, C, H*W]?[B, HW, C]?
        # 添加位置编码：截取到seq_len
        pe = self.pe_layer[:, :seq_len, :] # [1, HW, C]
        struct_features = struct_features + pe.expand(B, -1, -1)
        struct_features = self.dropout(struct_features) # 防止过拟合

        # 病理表征(全局embedding)
        pooled_hw = self._gap(last).squeeze(-1).squeeze(-1)  # [B, C]
        patho_features = self._gap_ln(pooled_hw).unsqueeze(1)  # [B, 1, C]

        # ===== Cross-Attention 融合 =====
        # 以病理表征为 Query，结构表征为 Key/Value
        # nn.MultiheadAttention 输入格式是 [L, B, E]，所以要转置
        # query = patho_features.transpose(0, 1)      # [1, B, C]
        # key_value = struct_features.transpose(0, 1) # [HW, B, C]

        # attn_layer = nn.MultiheadAttention(embed_dim=C, num_heads=8, batch_first=False).to(self.device_name)
        # fused, _ = attn_layer(query, key_value, key_value)  # [1, B, C]
        # fused = fused.squeeze(0)  # [B, C]
        fused, _ = self.attn_layer(patho_features, struct_features, struct_features) #[B,1,C]
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
    ) -> Dict[str, torch.Tensor]:
        """前向：计算图像特征、文本特征与相似度。

        - 若提供 texts，将内部编码文本；否则使用传入的 text_features。
        - 返回：
          - image_features: [B, D]
          - text_features:  [N, D]
          - logits_per_image: [B, N]
          - logits_per_text:  [N, B]
        """
        image_features = self.encode_image(pixel_values)

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

        return {
            "image_features": image_features,
            "text_features": text_features,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
        }


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
        print(f"DEBUG: args 长度 = {len(args)}")
        print(f"DEBUG: args 类型 = {[type(arg) for arg in args]}")
        print(f"DEBUG: kwargs 键 = {list(kwargs.keys())}")
        
        # 检查是否有图像相关的参数
        image_keys = ['images', 'pixel_values', 'image', 'input_images']
        text_keys = ['text_inputs', 'input_ids', 'texts', 'text', 'text_tokens']
        
        print(f"DEBUG: 查找图像参数...")
        for key in image_keys:
            if key in kwargs:
                print(f"DEBUG: 找到图像参数 '{key}': {type(kwargs[key])}")
        
        print(f"DEBUG: 查找文本参数...")
        for key in text_keys:
            if key in kwargs:
                print(f"DEBUG: 找到文本参数 '{key}': {type(kwargs[key])}")
        
        # 处理位置参数
        if len(args) >= 1:
            images = args[0]
            print(f"DEBUG: 从 args[0] 获取 images: {type(images)}")
        elif 'images' in kwargs:
            images = kwargs['images']
            print(f"DEBUG: 从 kwargs['images'] 获取 images: {type(images)}")
        elif 'pixel_values' in kwargs:
            images = kwargs['pixel_values']
            print(f"DEBUG: 从 kwargs['pixel_values'] 获取 images: {type(images)}")
        else:
            images = None
            print("DEBUG: images 为 None")
            
        if len(args) >= 2:
            text_inputs = args[1]
            print(f"DEBUG: 从 args[1] 获取 text_inputs: {type(text_inputs)}")
        elif 'text_inputs' in kwargs:
            text_inputs = kwargs['text_inputs']
            print(f"DEBUG: 从 kwargs['text_inputs'] 获取 text_inputs: {type(text_inputs)}")
        elif 'input_ids' in kwargs:
            text_inputs = kwargs['input_ids']
            print(f"DEBUG: 从 kwargs['input_ids'] 获取 text_inputs: {type(text_inputs)}")
        else:
            text_inputs = None
            print("DEBUG: text_inputs 为 None")
            
        # 从 kwargs 中获取 debug 参数
        debug = kwargs.get('debug', False)
        
        # 确保必要参数存在
        if images is None:
            print(f"DEBUG: 所有可用的参数:")
            print(f"  - args: {args}")
            print(f"  - kwargs: {kwargs}")
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
        image_features = self.core.encode_image(pixel_values)

        return image_features, text_features