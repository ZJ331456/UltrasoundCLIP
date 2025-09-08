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
        self.model_path = config['model']['vision_encoder']['model_path']
        self.embed_dim = config['model']['embed_dim']
        
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
        text_config = self.config['model']['text_encoder']
        text_model_name = text_config['model_name']
        self.text_encoder = AutoModel.from_pretrained(
            text_model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        # 初始化tokenizer缓存
        self._tokenizer = None
        
        print(f"  视觉编码器加载完成 (设备: {next(self.vision_model.parameters()).device})")
        print(f"  文本编码器: {text_model_name} (设备: {next(self.text_encoder.parameters()).device})")
    
    def _init_projections(self):
        """初始化投影层"""
        # 获取视觉特征维度-这应该与Qwen2.5-VL的视觉编码器输出维度匹配
        vision_dim = self.config['model']['vision_encoder']['vision_dim']
        # 获取文本特征维度
        text_dim = self.text_encoder.config.hidden_size
        
        # 获取设备和数据类型信息
        vision_device = next(self.vision_model.parameters()).device
        vision_dtype = next(self.vision_model.parameters()).dtype
        text_device = next(self.text_encoder.parameters()).device
        
        # 投影层 - 确保数据类型匹配
        # 使用视觉数据类型对齐是因为文本数据相对轻量并且保证了视觉特征处理的效率
        self.vision_projection = nn.Linear(vision_dim, self.embed_dim).to(device=vision_device, dtype=vision_dtype)
        self.text_projection = nn.Linear(text_dim, self.embed_dim).to(device=text_device, dtype=vision_dtype)
        
        # 温度参数 - 确保设备和数据类型匹配
        # 主要是不加会出现nan这种损失值
        logit_scale_value = torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        logit_scale_value = logit_scale_value.to(device=vision_device, dtype=vision_dtype)
        self.logit_scale = nn.Parameter(logit_scale_value)
        
        print(f"  投影层初始化完成: 视觉{vision_dim}→{self.embed_dim} (设备: {vision_device}, 类型: {vision_dtype}), 文本{text_dim}→{self.embed_dim} (设备: {text_device}, 类型: {vision_dtype})")
    
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
        
        if debug:
            print(f"DEBUG: pixel_values shape: {pixel_values.shape}")
            print(f"DEBUG: image_grid_thw: {image_grid_thw}")
            print(f"DEBUG: batch size: {len(pil_images)}")
        
        # 提取视觉特征
        with torch.no_grad():
            image_embeddings = self.vision_model(pixel_values, grid_thw=image_grid_thw)
        
        # 处理特征维度和分组
        batch_size = len(pil_images)
        process_type = self.config['model']['vision_encoder']['process_type']
        if process_type == 'default':
            image_features = self._process_vision_features(
                image_embeddings, image_grid_thw, batch_size, debug
            )
        elif process_type == 'projection':
            image_features = self._process_vision_features_projection(
                image_embeddings, image_grid_thw, batch_size, debug
            )
        elif process_type == 'attention_pooling':
            image_features = self._process_vision_features_attention_pooling(
                image_embeddings, image_grid_thw, batch_size, debug
            )
        elif process_type == 'top_k':
            image_features = self._process_vision_features_top_k(
                image_embeddings, image_grid_thw, batch_size, debug
            )
        elif process_type == 'cls_token':
            image_features = self._process_vision_features_cls_token(
                image_embeddings, image_grid_thw, batch_size, debug
            )
            
        # 投影到统一维度
        vision_device = next(self.vision_projection.parameters()).device
        if image_features.device != vision_device:
            image_features = image_features.to(vision_device)
            
        image_features = self.vision_projection(image_features)
        
        return F.normalize(image_features, dim=-1)
    
    def _preprocess_images(self, images: Union[torch.Tensor, str, Image.Image, list]) -> list:
        """预处理图像输入为PIL图像列表"""
        if isinstance(images, str):
            # 图像路径
            return [Image.open(images)]
        elif isinstance(images, Image.Image):
            return [images]
        elif isinstance(images, torch.Tensor):
            # tensor转换为PIL图像
            pil_images = []
            for i in range(images.shape[0]):
                img_tensor = images[i]
                
                # 反归一化（假设使用了标准的ImageNet归一化）
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
                
                img_tensor = img_tensor * std + mean
                img_tensor = torch.clamp(img_tensor, 0, 1)
                
                # 转换为PIL图像
                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                pil_image = Image.fromarray(img_np)
                pil_images.append(pil_image)
            return pil_images
        elif isinstance(images, list):
            # 处理列表输入，支持混合类型
            pil_images = []
            for img in images:
                if isinstance(img, str):
                    pil_images.append(Image.open(img))
                elif isinstance(img, Image.Image):
                    pil_images.append(img)
                else:
                    raise ValueError(f"列表中包含不支持的图像类型: {type(img)}")
            return pil_images
        else:
            raise ValueError(f"不支持的图像输入类型: {type(images)}")
    
    def _process_vision_features(self, image_embeddings: torch.Tensor, 
                               image_grid_thw: torch.Tensor, 
                               batch_size: int, 
                               debug: bool = False) -> torch.Tensor:
        """处理视觉特征，正确分组和池化"""
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
    
    def _process_vision_features_projection(self, image_embeddings: torch.Tensor, image_grid_thw: torch.Tensor, batch_size: int, debug: bool = False) -> torch.Tensor:
        """方案1: 可学习线性投影（Projection Head） + L2 归一化"""
        hidden_dim = image_embeddings.shape[-1]
        total_patches = image_embeddings.shape[0]

        patches_per_image = []
        for i in range(batch_size):
            t, h, w = image_grid_thw[i].tolist()
            num_patches = t * h * w
            patches_per_image.append(num_patches)

        start_idx = 0
        batch_features = []
        for num_patches in patches_per_image:
            img_patches = image_embeddings[start_idx:start_idx + num_patches]
            img_feature = img_patches.mean(dim=0, keepdim=True)
            batch_features.append(img_feature)
            start_idx += num_patches

        image_features = torch.cat(batch_features, dim=0)  # [batch_size, hidden_dim]
        image_features = self.vision_projection(image_features)
        return F.normalize(image_features, dim=-1)

    def _process_vision_features_attention_pooling(self, image_embeddings: torch.Tensor, image_grid_thw: torch.Tensor, batch_size: int, debug: bool = False) -> torch.Tensor:
        """方案2: Attention Pooling（加权聚合）"""
        query = nn.Parameter(torch.randn(1, image_embeddings.shape[-1]))
        attention_weights = torch.matmul(image_embeddings, query.T)
        attention_weights = F.softmax(attention_weights, dim=0)
        weighted_features = image_embeddings * attention_weights
        pooled_feature = weighted_features.sum(dim=0, keepdim=True)
        return pooled_feature

    def _process_vision_features_top_k(self, image_embeddings: torch.Tensor, image_grid_thw: torch.Tensor, batch_size: int, debug: bool = False) -> torch.Tensor:
        """方案3: Top-K Token 选择 + 投影"""
        k = 64  # 选取的token数量
        norms = torch.norm(image_embeddings, dim=-1)
        top_k_indices = torch.topk(norms, k=k, dim=0).indices
        top_k_features = image_embeddings[top_k_indices]
        projected_features = self.vision_projection(top_k_features)
        return projected_features.mean(dim=0, keepdim=True)

    def _process_vision_features_cls_token(self, image_embeddings: torch.Tensor, image_grid_thw: torch.Tensor, batch_size: int, debug: bool = False) -> torch.Tensor:
        """方案4: CLS Token 或聚合 Token 表示"""
        cls_token = image_embeddings[0]  # 假设CLS token在第一个位置
        return cls_token.unsqueeze(0)
    
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
            text_model_name = self.config.get('text_encoder', {}).get('model_name', 'bert-base-chinese')
            
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
        text_model_name = self.config['text_encoder']['model_name']
        # text_model_name = self.config.get('text_encoder', {}).get('model_name', 'bert-base-chinese')
        return AutoTokenizer.from_pretrained(text_model_name)

