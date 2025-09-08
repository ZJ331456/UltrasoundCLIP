"""
JSONL格式数据集实现
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class JSONLDataset(BaseDataset):
    """JSONL格式数据集"""
    
    def __init__(self, 
                 nodes_jsonl: str,
                 texts_jsonl: str,
                 image_root: str,
                 transform=None,
                 max_samples=None,
                 cache_size: int = 500):
        """
        初始化JSONL数据集
        Args:
            nodes_jsonl: 图像节点文件
            texts_jsonl: 文本文件
            image_root: 图像根目录
            transform: 图像变换
            max_samples: 最大样本数量
            cache_size: 图像缓存大小
        """
        self.nodes_jsonl = nodes_jsonl
        self.texts_jsonl = texts_jsonl
        self.image_root = Path(image_root)
        
        super().__init__(transform, max_samples, cache_size)
        
        # 加载数据
        self.samples = self._load_samples()
        logger.info(f"JSONL数据集加载完成，共 {len(self.samples)} 个样本")
    
    def _load_samples(self) -> List[Tuple[str, str]]:
        """加载JSONL格式的样本数据"""
        # 加载图像ID到路径的映射
        id2path = self._load_image_mapping()
        
        # 加载文本描述并匹配图像
        samples = []
        with open(self.texts_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                obj = json.loads(line)
                text = obj['text']
                
                for image_id in obj['image_ids']:
                    image_path = id2path['image_id']
                    if image_path:
                        full_path = self.image_root / image_path
                        if full_path.exists():
                            samples.append((str(full_path), text))
                    
                    # 限制样本数量
                    if self.max_samples and len(samples) >= self.max_samples:
                        return samples
        
        return samples
    
    def _load_image_mapping(self) -> dict:
        """加载图像ID到路径的映射"""
        id2path = {}
        with open(self.nodes_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    id2path[obj['image_id']] = obj['image_path']
        return id2path
