"""
SAM+BERT模型模块
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from segment_anything import sam_model_registry
from typing import Dict, Any, Tuple


class TextModel(nn.Module):
    """SAM+BERT组合模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化SAM+BERT模型
        Args:
            config: 模型配置
        """
        super().__init__()
        
        self.config = config
        text_config = config['text_encoder']
        self.embed_dim = config['embed_dim']
        
        # 创建文本编码器（BERT）
        self.text_encoder = self._create_text_encoder(text_config)
        
        # 投影层
        bert_dim = self.text_encoder.config.hidden_size

        self.text_projection = nn.Linear(bert_dim, self.embed_dim)
        
        # 温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        
        print(f"  - 文本编码器: BERT {text_config['model_name']}")
    def _create_text_encoder(self, config: Dict[str, Any]):
        """创建BERT文本编码器"""
        model_name = config['model_name']
        return AutoModel.from_pretrained(model_name)
    
    def encode_text(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """编码文本"""
        # BERT文本编码
        outputs = self.text_encoder(**text_inputs)
        # 使用[CLS] token的表示
        features = outputs.last_hidden_state[:, 0]  # [B, hidden_size]
        # 投影到统一维度
        features = self.text_projection(features)
        return nn.functional.normalize(features, dim=-1)
    
    def forward(self, images: torch.Tensor, text_inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_inputs)
        return image_features, text_features
    
    @property
    def tokenizer(self):
        """获取tokenizer"""
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.config['text_encoder']['model_name'])
