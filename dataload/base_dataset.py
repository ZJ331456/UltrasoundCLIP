"""
基础数据集类
提供通用的数据集接口和功能
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from functools import lru_cache

# 允许加载截断的图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):
    """基础数据集抽象类"""
    
    def __init__(self, 
                 transform=None, 
                 max_samples: Optional[int] = None,
                 cache_size: int = 500):
        """
        初始化基础数据集
        Args:
            transform: 图像变换
            max_samples: 最大样本数量
            cache_size: 图像缓存大小
        """
        self.transform = transform
        self.max_samples = max_samples
        self.cache_size = cache_size
        
        # 初始化样本数据
        self.samples = []
        self.stats = {}
        
        # 设置图像缓存
        if cache_size > 0:
            self._cached_load_image = lru_cache(maxsize=cache_size)(self._load_image)
        else:
            self._cached_load_image = self._load_image
    
    @abstractmethod
    def _load_samples(self) -> List[Tuple[str, str]]:
        """加载样本数据 - 子类必须实现"""
        pass
    
    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """加载单张图像"""
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            logger.warning(f"图像加载失败 {image_path}: {e}")
            return None
    
    def _get_placeholder_image(self) -> Image.Image:
        """获取占位符图像"""
        return Image.new('RGB', (64, 64), color='gray')
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path, text = self.samples[idx]
        
        # 加载图像
        image = self._cached_load_image(image_path)
        
        if image is None:
            image = self._get_placeholder_image()
            logger.warning(f"使用占位符图像替代: {image_path}")
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'text': text,
            'image_path': image_path
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return self.stats.copy()
