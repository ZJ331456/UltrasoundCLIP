"""
数据整理器模块
负责批处理数据的整理和tokenization
"""

import torch
from typing import List, Dict, Any, Union
from transformers import AutoTokenizer
import open_clip


class TextTokenizer:
    """文本tokenizer封装"""
    
    def __init__(self, tokenizer_type: str = 'bert', max_length: int = 256, model_path: str = None):
        self.tokenizer_type = tokenizer_type
        self.max_length = max_length
        self.model_path = model_path
        self.tokenizer = self._create_tokenizer()
    
    def _create_tokenizer(self):
        """创建tokenizer"""
        if self.tokenizer_type == 'bert':
            return AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
        elif self.tokenizer_type == 'local_bert':
            if self.model_path:
                return AutoTokenizer.from_pretrained(self.model_path)
            else:
                raise ValueError("local_bert类型需要指定model_path参数")
        elif self.tokenizer_type == 'clip':
            return open_clip.get_tokenizer('ViT-L-14')
        else:
            raise ValueError(f"不支持的tokenizer类型: {self.tokenizer_type}")
    
    def tokenize(self, texts: List[str]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """文本tokenization"""
        if self.tokenizer_type == 'bert':
            encoded = self.tokenizer.batch_encode_plus(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            return encoded
        else:
            # CLIP tokenizer
            tokens = []
            for text in texts:
                token_ids = self.tokenizer.encode(text)
                # 截断和填充
                if len(token_ids) > self.max_length:
                    token_ids = token_ids[:self.max_length]
                while len(token_ids) < self.max_length:
                    token_ids.append(0)  # padding token
                tokens.append(token_ids)
            
            return torch.tensor(tokens, dtype=torch.long)


class DataCollator:
    """通用数据整理器"""
    
    def __init__(self, 
                 max_text_length: int = 256, 
                 tokenizer_type: str = 'bert',
                 model_path: str = None):
        """
        初始化数据整理器
        Args:
            max_text_length: 文本最大长度
            tokenizer_type: tokenizer类型
            model_path: 本地模型路径（当tokenizer_type为local_bert时需要）
        """
        self.text_tokenizer = TextTokenizer(tokenizer_type, max_text_length, model_path)
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        整理批数据
        Args:
            batch: 批数据列表
        Returns:
            整理后的批数据
        """
        # 整理图像 - 增强形状验证和修复
        try:
            processed_images = []
            target_channels = 3  # 强制所有图像为3通道
            
            for i, item in enumerate(batch):
                # 支持两种键名：'image' 和 'images'
                if 'image' in item:
                    img_tensor = item['image']
                elif 'images' in item:
                    img_tensor = item['images']
                else:
                    available_keys = list(item.keys())
                    raise KeyError(f"图像 {i} 缺少图像数据，可用键: {available_keys}")
                
                # 确保是3维tensor [C, H, W]
                if len(img_tensor.shape) != 3:
                    raise ValueError(f"图像 {i} 维度错误: {img_tensor.shape}，期望3维 [C, H, W]")
                
                channels, height, width = img_tensor.shape
                
                # 处理通道数问题
                if channels == target_channels:
                    processed_images.append(img_tensor)
                elif channels > target_channels:
                    # 截取前3个通道
                    processed_images.append(img_tensor[:target_channels])
                    print(f"警告: 图像 {i} 通道数 {channels} > {target_channels}，截取前{target_channels}个通道")
                elif channels < target_channels:
                    # 复制通道到3个
                    if channels == 1:
                        # 灰度图转RGB
                        rgb_tensor = img_tensor.repeat(3, 1, 1)
                    else:
                        # 其他情况，重复最后一个通道
                        missing_channels = target_channels - channels
                        extra_channels = img_tensor[-1:].repeat(missing_channels, 1, 1)
                        rgb_tensor = torch.cat([img_tensor, extra_channels], dim=0)
                    processed_images.append(rgb_tensor)
                    print(f"警告: 图像 {i} 通道数 {channels} < {target_channels}，扩展到{target_channels}通道")
            
            # 验证所有图像形状一致后再堆叠
            shapes = [img.shape for img in processed_images]
            if len(set(shapes)) > 1:
                print(f"错误: 处理后图像形状仍不一致: {shapes}")
                # 统一到最小尺寸
                min_h = min(img.shape[1] for img in processed_images)
                min_w = min(img.shape[2] for img in processed_images)
                processed_images = [img[:, :min_h, :min_w] for img in processed_images]
                print(f"统一图像尺寸到: [3, {min_h}, {min_w}]")
            
            images = torch.stack(processed_images)
            
        except Exception as e:
            print(f"图像堆叠错误: {e}")
            print(f"批次大小: {len(batch)}")
            for i, item in enumerate(batch):
                if 'image' in item:
                    print(f"图像 {i}: 形状 {item['image'].shape}, 类型 {item['image'].dtype}")
                elif 'images' in item:
                    print(f"图像 {i}: 形状 {item['images'].shape}, 类型 {item['images'].dtype}")
                else:
                    print(f"图像 {i}: 缺少图像数据，可用键: {list(item.keys())}")
            raise e
        
        # 整理文本
        texts = [item['text'] for item in batch]
        text_tokens = self.text_tokenizer.tokenize(texts)
        
        # 整理其他信息
        image_paths = [item['image_path'] for item in batch]
        
        return {
            'images': images,
            'text_tokens': text_tokens,
            'texts': texts,
            'image_paths': image_paths
        }
