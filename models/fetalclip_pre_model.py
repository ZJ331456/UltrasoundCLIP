"""
FetalCLIP模型模块
"""

import torch
import torch.nn as nn
import open_clip
import json
import os
from typing import Optional, Tuple


class FetalCLIPPreModel(nn.Module):
    """FetalCLIP模型"""
    
    def __init__(self, config_path: str, pretrained_path: Optional[str] = None):
        """
        初始化FetalCLIP模型
        Args:
            config_path: 配置文件路径
            pretrained_path: 预训练权重路径
        """
        super().__init__()
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 创建模型
        self.model = self._create_model()
        
        # 加载预训练权重
        if pretrained_path:
            self._load_pretrained(pretrained_path)
        
        # 获取tokenizer
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
    
    def _load_config(self, config_path: str) -> dict:
        """加载FetalCLIP配置"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"FetalCLIP配置文件加载成功")
        return config
    
    def _create_model(self):
        """创建FetalCLIP模型"""
        # 注册配置
        arch_name = 'ViT-L-14'
        open_clip.factory._MODEL_CONFIGS[arch_name] = self.config
        
        # 创建模型
        model, _, _ = open_clip.create_model_and_transforms(arch_name)
        return model
    
    def _load_pretrained(self, pretrained_path: str):
        """加载预训练权重"""
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"预训练权重文件不存在: {pretrained_path}")
            
        print(f"加载FetalCLIP预训练权重: {pretrained_path}")
        
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 处理不同的checkpoint格式
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 移除module.前缀
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # 加载权重
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"  缺失的权重键 ({len(missing_keys)}): {missing_keys[:3]}...")
        if unexpected_keys:
            print(f"  多余的权重键 ({len(unexpected_keys)}): {unexpected_keys[:3]}...")
            
        print("FetalCLIP权重加载完成")
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """编码图像"""
        return self.model.encode_image(images)
    
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """编码文本"""
        return self.model.encode_text(text_tokens)
    
    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens)
        return image_features, text_features
