"""
SAM+BERT模型模块
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from segment_anything import sam_model_registry
from typing import Dict, Any, Tuple


class SAMBERTPreModel(nn.Module):
    """SAM+BERT组合模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化SAM+BERT模型
        Args:
            config: 模型配置
        """
        super().__init__()
        
        self.config = config
        image_config = config['image_encoder']
        text_config = config['text_encoder']
        self.embed_dim = config['embed_dim']
        
        # 创建图像编码器（SAM）
        self.image_encoder = self._create_image_encoder(image_config)
        
        # 创建文本编码器（BERT）
        self.text_encoder = self._create_text_encoder(text_config)
        
        # 投影层
        sam_dim = self._get_sam_feature_dim(image_config['model_name'])
        bert_dim = self.text_encoder.config.hidden_size
        
        self.image_projection = nn.Linear(sam_dim, self.embed_dim)
        self.text_projection = nn.Linear(bert_dim, self.embed_dim)
        
        # 温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        
        print(f"SAM+BERT模型初始化完成:")
        print(f"  - 图像编码器: SAM {image_config['model_name']}")
        print(f"  - 文本编码器: BERT {text_config['model_name']}")
        print(f"  - 嵌入维度: {self.embed_dim}")
    
    def _create_image_encoder(self, config: Dict[str, Any]):
        """创建SAM图像编码器"""
        model_name = config['model_name']
        checkpoint_path = config['checkpoint_path']
        
        # 先创建没有checkpoint的SAM模型
        if model_name == 'vit_l':
            sam = sam_model_registry['vit_l'](checkpoint=None)
        elif model_name == 'vit_b':
            sam = sam_model_registry['vit_b'](checkpoint=None)
        elif model_name == 'vit_h':
            sam = sam_model_registry['vit_h'](checkpoint=None)
        else:
            raise ValueError(f"不支持的SAM模型: {model_name}")
        
        # 如果有checkpoint路径，手动加载权重
        if checkpoint_path:
            print(f"正在加载SAM权重: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 处理不同的checkpoint格式
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
                
            # 移除不需要的键
            if 'meta' in state_dict:
                del state_dict['meta']
            
            # 尝试加载权重，使用strict=False允许部分加载
            try:
                sam.load_state_dict(state_dict, strict=False)
                print("SAM权重加载成功")
            except Exception as e:
                print(f"SAM权重加载失败，使用随机初始化: {e}")

        return sam.image_encoder
    
    def _create_text_encoder(self, config: Dict[str, Any]):
        """创建BERT文本编码器"""
        model_name = config['model_name']
        return AutoModel.from_pretrained(model_name)
    
    def _get_sam_feature_dim(self, model_name: str) -> int:
        """获取SAM特征维度"""
        if model_name in ['vit_l', 'vit_h']:
            return 1024
        elif model_name == 'vit_b':
            return 256  # SAM ViT-B的实际输出维度是256
        else:
            return 1024  # 默认值
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """编码图像"""
        # SAM图像编码
        features = self.image_encoder(images)
        # Global average pooling
        features = features.mean(dim=[2, 3])  # [B, C, H, W] -> [B, C]
        # 投影到统一维度
        features = self.image_projection(features)
        return nn.functional.normalize(features, dim=-1)
    
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
        # 不添加词汇表就直接返回
        # return AutoTokenizer.from_pretrained(self.config['text_encoder']['model_name'])
        #  添加词汇表！
        tokenizer = AutoTokenizer.from_pretrained(self.config['text_encoder']['model_name'])
        
        # 读取新增词汇表
        vocab_path = "/media/ps/data-ssd/UltrasoundRAG/clip_caption/caption_breast/breast_vocab/vocab.txt"
        with open(vocab_path, "r", encoding="utf-8") as f:
            new_tokens = [line.strip() for line in f.readlines()]
        
        # 新增词汇到分词器
        tokenizer.add_tokens(new_tokens)
        print(f"新增词汇数量: {len(new_tokens)}")
        
        # 调整模型嵌入层大小
        self.text_encoder.resize_token_embeddings(len(tokenizer))
        
        return tokenizer
