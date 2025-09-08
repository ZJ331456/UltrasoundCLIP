"""
MedCLIP模型实现 - 用于超声CLIP训练的验证实验
基于原始MedCLIP架构，适配现有训练框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Union, Tuple, List
from PIL import Image
import os
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as transforms
import numpy as np

# 导入MedCLIP组件
import sys
sys.path.append('/media/ps/data-ssd/UltrasoundRAG/CLIP/models/MedCLIP')
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
from medclip import MedCLIPProcessor


class MedCLIPWrapper(nn.Module):
    """
    MedCLIP模型包装器，适配现有训练框架
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.embed_dim = config['embed_dim']
        
        # 初始化组件
        self._init_medclip_model()
        self._init_text_encoder()
        self._init_processor()
        
        print(f"MedCLIP包装器初始化完成")
        print(f"  视觉编码器: MedCLIP-ViT")
        print(f"  嵌入维度: {self.embed_dim}")
    
    def _init_medclip_model(self):
        """初始化MedCLIP模型"""
        vision_config = self.config['vision_encoder']
        
        # 使用MedCLIP-ViT作为视觉编码器
        if vision_config.get('architecture') == 'resnet50':
            vision_cls = MedCLIPVisionModel
            print("使用MedCLIP-ResNet50作为视觉编码器")
        else:
            vision_cls = MedCLIPVisionModelViT
            print("使用MedCLIP-ViT作为视觉编码器")
        
        # 加载预训练权重
        checkpoint_path = vision_config.get('checkpoint_path')
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"从本地加载MedCLIP预训练权重: {checkpoint_path}")
            # 直接创建视觉模型并加载权重，避免网络下载
            self.vision_model = vision_cls(medclip_checkpoint=checkpoint_path)
            
            # 动态获取特征维度
            if isinstance(self.vision_model, MedCLIPVisionModelViT):
                # ViT模型的特征维度通常是768
                self.vision_feature_dim = 768
            else:
                # ResNet模型的特征维度是512  
                self.vision_feature_dim = 512
        else:
            print("警告: 未找到本地预训练权重，将尝试在线下载")
            self.medclip_model = MedCLIPModel(vision_cls=vision_cls)
            self.medclip_model.from_pretrained()
            self.vision_model = self.medclip_model.vision_model
            
            # 动态获取特征维度
            if isinstance(self.vision_model, MedCLIPVisionModelViT):
                self.vision_feature_dim = 768
            else:
                self.vision_feature_dim = 512
        
        print(f"MedCLIP视觉编码器初始化完成，特征维度: {self.vision_feature_dim}")
    
    def _init_text_encoder(self):
        """初始化文本编码器（复用qwen配置中的BERT）"""
        text_config = self.config['text_encoder']
        text_model_name = text_config['model_name']
        
        # 使用与qwen_vl_clip相同的文本编码器（移除device_map以兼容BERT）
        self.text_encoder = AutoModel.from_pretrained(
            text_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        # 获取文本特征维度
        self.text_feature_dim = self.text_encoder.config.hidden_size
        
        print(f"  文本编码器: {text_model_name}")
        print(f"  文本特征维度: {self.text_feature_dim}")
    
    def _init_processor(self):
        """初始化图像处理器"""
        # 使用MedCLIP的处理器
        self.processor = MedCLIPProcessor()
        
        # 为了保持一致性，也可以手动定义transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 投影层，将MedCLIP特征投影到目标维度
        self._init_projections()
    
    def _init_projections(self):
        """初始化投影层"""
        # 视觉投影层
        self.vision_projection = nn.Sequential(
            nn.Linear(self.vision_feature_dim, self.vision_feature_dim // 2),
            nn.LayerNorm(self.vision_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.vision_feature_dim // 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        )
        
        # 文本投影层
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_feature_dim, self.text_feature_dim // 2),
            nn.LayerNorm(self.text_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.text_feature_dim // 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        )
        
        # 温度参数
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))
        
        print(f"  投影层: 视觉{self.vision_feature_dim}→{self.embed_dim}, 文本{self.text_feature_dim}→{self.embed_dim}")
    
    def encode_image(self, images: Union[torch.Tensor, str, Image.Image, list], debug: bool = False) -> torch.Tensor:
        """
        编码图像
        Args:
            images: 图像输入
            debug: 调试模式
        Returns:
            归一化的图像特征
        """
        # 如果输入已经是tensor，直接使用MedCLIP处理器的transform
        if isinstance(images, torch.Tensor):
            # 训练时数据加载器传入的已经是预处理好的tensor
            device = next(self.vision_model.parameters()).device
            pixel_values = images.to(device)
            
            if debug:
                print(f"DEBUG: 直接使用tensor输入，形状: {pixel_values.shape}")
        else:
            # 如果是PIL图像或路径，使用MedCLIP处理器
            pil_images = self._preprocess_images(images)
            
            # 使用MedCLIP处理器
            if len(pil_images) == 1:
                # 单张图像
                inputs = self.processor(images=pil_images[0], return_tensors="pt")
            else:
                # 批量图像 - 逐个处理然后堆叠
                batch_inputs = []
                for img in pil_images:
                    img_input = self.processor(images=img, return_tensors="pt")
                    batch_inputs.append(img_input['pixel_values'])
                
                inputs = {'pixel_values': torch.cat(batch_inputs, dim=0)}
            
            # 移动到正确设备
            device = next(self.vision_model.parameters()).device
            pixel_values = inputs['pixel_values'].to(device)
            
            if debug:
                print(f"DEBUG: 图像输入形状: {pixel_values.shape}")
        
        # 提取视觉特征
        if hasattr(self.vision_model, 'forward') and 'project' in self.vision_model.forward.__code__.co_varnames:
            # 对于ViT模型，获取未投影的特征
            image_features = self.vision_model(pixel_values, project=False)
        else:
            # 对于ResNet模型，直接获取特征
            image_features = self.vision_model(pixel_values)
        
        if debug:
            print(f"DEBUG: MedCLIP视觉特征形状: {image_features.shape}")
            print(f"DEBUG: 期望的投影层输入维度: {self.vision_feature_dim}")
        
        # 使用我们的投影层
        image_features = self.vision_projection(image_features)
        
        # 归一化
        normalized_features = F.normalize(image_features, p=2, dim=-1)
        
        if debug:
            print(f"DEBUG: 最终图像特征形状: {normalized_features.shape}")
        
        return normalized_features
    
    def _preprocess_images(self, images: Union[torch.Tensor, str, Image.Image, list]) -> List[Image.Image]:
        """预处理图像输入"""
        if isinstance(images, str):
            return [Image.open(images).convert('RGB')]
        elif isinstance(images, Image.Image):
            return [images.convert('RGB')]
        elif isinstance(images, torch.Tensor):
            # 将tensor转换为PIL图像
            pil_images = []
            if len(images.shape) == 4:  # Batch
                for i in range(images.shape[0]):
                    img_tensor = images[i]
                    # 反归一化（确保设备匹配）
                    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
                    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
                    img_tensor = img_tensor * std + mean
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                    pil_images.append(Image.fromarray(img_np))
            elif len(images.shape) == 3:  # Single image
                # 反归一化（确保设备匹配）
                std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(3, 1, 1)
                mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(3, 1, 1)
                img_tensor = images * std + mean
                img_tensor = torch.clamp(img_tensor, 0, 1)
                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                pil_images.append(Image.fromarray(img_np))
            return pil_images
        elif isinstance(images, list):
            pil_images = []
            for img in images:
                if isinstance(img, str):
                    pil_images.append(Image.open(img).convert('RGB'))
                elif isinstance(img, Image.Image):
                    pil_images.append(img.convert('RGB'))
                elif isinstance(img, torch.Tensor):
                    # 处理tensor（确保设备匹配）
                    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
                    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
                    img_tensor = img * std + mean
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                    pil_images.append(Image.fromarray(img_np))
            return pil_images
        else:
            raise ValueError(f"不支持的图像输入类型: {type(images)}")
    
    def encode_text(self, text_inputs: Union[Dict[str, torch.Tensor], list, str, torch.Tensor]) -> torch.Tensor:
        """
        编码文本
        Args:
            text_inputs: 文本输入
        Returns:
            归一化的文本特征
        """
        # 处理不同类型的输入
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]
        
        if isinstance(text_inputs, torch.Tensor):
            # 训练时传入的tokenized tensor (input_ids)
            device = next(self.text_encoder.parameters()).device
            input_ids = text_inputs.to(device)
            
            # 创建attention_mask (假设非pad token的位置为1)
            attention_mask = (input_ids != 0).long().to(device)  # 假设pad_token_id为0
            
            text_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        elif isinstance(text_inputs, list):
            # 使用BERT tokenizer
            text_inputs = self.text_tokenizer(
                text_inputs,
                padding=True,
                truncation=True,
                max_length=self.config['text_encoder']['max_length'],
                return_tensors="pt"
            )
            
            # 移动到正确设备
            device = next(self.text_encoder.parameters()).device
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        elif isinstance(text_inputs, dict):
            # 已经tokenized
            device = next(self.text_encoder.parameters()).device
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        else:
            raise ValueError(f"不支持的文本输入类型: {type(text_inputs)}")
        
        # BERT文本编码
        outputs = self.text_encoder(**text_inputs)
        
        # 使用[CLS] token
        text_features = outputs.last_hidden_state[:, 0]
        
        # 投影到统一维度
        text_features = self.text_projection(text_features)
        
        # 归一化
        normalized_features = F.normalize(text_features, p=2, dim=-1)
        
        return normalized_features
    
    def forward(self, images, texts, debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            images: 图像输入
            texts: 文本输入
            debug: 调试模式
        Returns:
            image_features, text_features
        """
        # 编码图像和文本
        image_features = self.encode_image(images, debug=debug)
        text_features = self.encode_text(texts)
        
        return image_features, text_features
    
    @property
    def tokenizer(self):
        """获取tokenizer"""
        return self.text_tokenizer
