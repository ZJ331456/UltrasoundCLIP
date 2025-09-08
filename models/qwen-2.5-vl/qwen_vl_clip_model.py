"""
Qwen2.5-VL视觉编码器+文本编码器模型模块
"""

import torch
import torch.nn as nn
import json
import os
from typing import Optional, Tuple, Dict, Any, Union
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModel, AutoTokenizer
import torch.nn.functional as F
from pathlib import Path

# 加载配置文件进行测试，后面可以删除！
def load_config(path: str = "qwen_clip_config.json"):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()

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
        self.model_path = config['model_path']
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
        
        # 加载完整模型
        self.full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # 提取视觉编码器
        self.vision_model = self.full_model.model.visual
        
        # 加载文本编码器（BERT）
        text_config = self.config['text_encoder']
        text_model_name = text_config['model_name']
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        # 将文本编码器移动到合适的设备并转换数据类型
        vision_device = next(self.vision_model.parameters()).device
        vision_dtype = next(self.vision_model.parameters()).dtype
        if torch.cuda.is_available() and vision_device.type == 'cuda':
            # 如果有GPU且视觉模型在GPU上，将文本编码器也移动到GPU并转换数据类型
            self.text_encoder = self.text_encoder.to(device=vision_device, dtype=vision_dtype)
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        print(f"  视觉编码器加载完成 (设备: {next(self.vision_model.parameters()).device})")
        print(f"  文本编码器: {text_model_name} (设备: {next(self.text_encoder.parameters()).device})")
    
    def _init_projections(self):
        """初始化投影层"""
        # 获取视觉特征维度（实际是2048维）
        # 这应该与Qwen2.5-VL的视觉编码器输出维度匹配
        vision_dim = 2048
        
        # 获取文本特征维度
        text_dim = self.text_encoder.config.hidden_size
        
        # 获取设备和数据类型信息
        vision_device = next(self.vision_model.parameters()).device
        vision_dtype = next(self.vision_model.parameters()).dtype
        text_device = next(self.text_encoder.parameters()).device
        
        # 投影层 - 确保数据类型匹配
        self.vision_projection = nn.Linear(vision_dim, self.embed_dim).to(device=vision_device, dtype=vision_dtype)
        self.text_projection = nn.Linear(text_dim, self.embed_dim).to(device=text_device, dtype=vision_dtype)
        
        # 温度参数 - 确保设备和数据类型匹配
        logit_scale_value = torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        logit_scale_value = logit_scale_value.to(device=vision_device, dtype=vision_dtype)
        self.logit_scale = nn.Parameter(logit_scale_value)
        
        print(f"  投影层初始化完成: 视觉{vision_dim}→{self.embed_dim} (设备: {vision_device}, 类型: {vision_dtype}), 文本{text_dim}→{self.embed_dim} (设备: {text_device}, 类型: {vision_dtype})")
    
    def encode_image(self, images: Union[torch.Tensor, str, Image.Image, list]) -> torch.Tensor:
        """
        编码图像
        Args:
            images: 图像输入，可以是tensor、路径、PIL图像或列表
        Returns:
            image_features: 归一化的图像特征
        """
        # 处理不同类型的输入
        if isinstance(images, str):
            # 图像路径
            image = Image.open(images)
            images = [image]
        elif isinstance(images, Image.Image):
            images = [images]
        elif isinstance(images, torch.Tensor):
            # 假设已经是处理过的pixel_values
            pixel_values = images
        else:
            # 列表形式的图像
            pass
        
        if not isinstance(images, torch.Tensor):
            # 处理图像
            dummy_text = "<|vision_start|><|image_pad|><|vision_end|>"
            inputs = self.processor(text=[dummy_text] * len(images), images=images, return_tensors="pt")
            
            # 移动到正确设备
            device = next(self.vision_model.parameters()).device
            pixel_values = inputs["pixel_values"].to(device)
            image_grid_thw = inputs["image_grid_thw"].to(device)
        
        # 提取视觉特征
        with torch.no_grad():
            image_embeddings = self.vision_model(pixel_values, grid_thw=image_grid_thw)
        
        # 调试信息：打印视觉编码器输出形状和图像网格信息
        print(f"视觉编码器输出形状: {image_embeddings.shape}")
        print(f"图像网格信息: {image_grid_thw}")
        
        # # 全局平均池化（如果需要）
        # if len(image_embeddings.shape) > 2:
        #     image_features = image_embeddings.mean(dim=1)  # [B, hidden_size]
        # else:
        #     image_features = image_embeddings
        # 强制全局平均池化
        image_features = image_embeddings.mean(dim=0, keepdim=True)  # [1, 2048]


        # 投影到统一维度
        # 确保特征在正确的设备上（处理多GPU情况）
        vision_device = next(self.vision_projection.parameters()).device
        if image_features.device != vision_device:
            image_features = image_features.to(vision_device)
            
        image_features = self.vision_projection(image_features)
        
        return F.normalize(image_features, dim=-1)
    
    def encode_text(self, text_inputs: Union[Dict[str, torch.Tensor], list, str]) -> torch.Tensor:
        """
        编码文本
        Args:
            text_inputs: 文本输入，可以是tokenized的字典、文本列表或单个文本
        Returns:
            text_features: 归一化的文本特征
        """
        # 处理不同类型的输入
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]
        
        if isinstance(text_inputs, list):
            # 文本列表，需要tokenize
            tokenizer = AutoTokenizer.from_pretrained(self.config.get('text_encoder', {}).get('model_name', 'bert-base-chinese'))
            text_inputs = tokenizer(
                text_inputs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # 移动到正确设备
            device = next(self.text_encoder.parameters()).device
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        # BERT文本编码
        outputs = self.text_encoder(**text_inputs)
        # 使用[CLS] token的表示
        text_features = outputs.last_hidden_state[:, 0]  # [B, hidden_size]
        
        # 投影到统一维度
        text_features = self.text_projection(text_features)
        
        return F.normalize(text_features, dim=-1)
    
    def forward(self, images, texts) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            images: 图像输入
            texts: 文本输入
        Returns:
            image_features, text_features
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        return image_features, text_features
    
    @property
    def tokenizer(self):
        """获取tokenizer"""
        text_model_name = self.config.get('text_encoder', {}).get('model_name', 'bert-base-chinese')
        return AutoTokenizer.from_pretrained(text_model_name)

# QwenVLCLIPPostModel =  QwenVLCLIPModel()

# class QwenVLCLIPPostModel:
#     """Qwen2.5-VL CLIP专用加载器"""
    
#     def __init__(self):
#         self.model_type = "Qwen2.5-VL CLIP"
    
#     def load_model(self, 
#                    model_path: str,
#                    config_path: Optional[str] = None,
#                    device: str = 'auto') -> Tuple[QwenVLCLIPModel, Any]:
#         """
#         加载Qwen2.5-VL CLIP模型
#         Args:
#             model_path: Qwen2.5-VL模型路径
#             config_path: 配置文件路径（可选）
#             device: 设备
#         Returns:
#             model, tokenizer
#         """
#         print(f"正在加载{self.model_type}模型...")
#         print(f"  模型路径: {model_path}")
#         print(f"  设备: {device}")
        
#         # 1. 加载或创建配置
#         if config_path and os.path.exists(config_path):
#             config = self._load_config(config_path)
#         else:
#             config = self._create_default_config(model_path)
        
#         # 2. 创建模型
#         model = QwenVLCLIPModel(config)
        
#         # 3. 获取tokenizer
#         tokenizer = model.tokenizer
        
#         # 4. 设置评估模式
#         model.eval()
        
#         print(f"{self.model_type}模型加载完成!")
#         return model, tokenizer
    
#     def _load_config(self, config_path: str) -> Dict[str, Any]:
#         """加载配置文件"""
#         with open(config_path, 'r', encoding='utf-8') as f:
#             config = json.load(f)
#         print(f"  配置文件加载成功: {config_path}")
#         return config
    
#     def _create_default_config(self, model_path: str) -> Dict[str, Any]:
#         """创建默认配置"""
#         config = {
#             'model_path': model_path,
#             'embed_dim': 768,
#             'text_encoder': {
#                 'model_name': 'bert-base-chinese'
#             }
#         }
#         print(f"  使用默认配置")
#         return config


# def load_qwen_vl_clip_model(model_path: str,
#                            config_path: Optional[str] = None,
#                            device: str = 'auto') -> Tuple[QwenVLCLIPModel, Any]:
#     """
#     便捷函数：加载Qwen2.5-VL CLIP模型
#     Args:
#         model_path: Qwen2.5-VL模型路径
#         config_path: 配置文件路径（可选）
#         device: 设备
#     Returns:
#         model, tokenizer
#     """
#     loader = QwenVLCLIPPostModel()
#     return loader.load_model(model_path, config_path, device)


# # 便捷函数，兼容原始vision_encoder功能
# def extract_vision_features(model_path: str, image_path: str):
#     """
#     提取视觉特征（兼容原始函数）
#     Args:
#         model_path: 模型路径
#         image_path: 图像路径
#     Returns:
#         image_embeddings, image_grid_thw
#     """
#     model, _ = load_qwen_vl_clip_model(model_path)
    
#     # 加载图像
#     image = Image.open(image_path)
    
#     # 提取特征
#     with torch.no_grad():
#         image_features = model.encode_image(image)
    
#     print(f"图像特征提取完成")
#     print(f"特征形状: {image_features.shape}")
    
#     return image_features, None  # 返回格式兼容原始函数