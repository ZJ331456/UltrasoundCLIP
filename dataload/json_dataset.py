"""
JSON格式数据集实现
"""

import os
import json
import logging
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class JSONDataset(BaseDataset):
    """JSON格式数据集"""
    
    def __init__(self, 
                 json_file: str,
                 transform=None,
                 max_samples=None,
                 validate_files: bool = True,
                 cache_size: int = 500):
        """
        初始化JSON数据集
        Args:
            json_file: JSON数据文件路径
            transform: 图像变换
            max_samples: 最大样本数量
            validate_files: 是否验证文件存在性
            cache_size: 图像缓存大小
        """
        self.json_file = json_file
        self.validate_files = validate_files
        
        super().__init__(transform, max_samples, cache_size)
        
        # 加载数据
        self.samples, self.stats = self._load_samples()
        self._log_stats()
    
    def _load_samples(self) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
        """加载JSON格式的样本数据"""
        logger.info(f"开始加载JSON数据集: {self.json_file}")
        
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        stats = {
            'total': 0,
            'valid': 0,
            'missing_image': 0,
            'missing_caption': 0,
            'file_not_exists': 0
        }
        
        data_info = data['DataInfo']
        
        # 批量检查文件存在性（如果需要）
        existing_files = set()
        if self.validate_files:
            image_paths = [
                sample_data['data_path'] 
                for sample_data in data_info.values() 
                if sample_data['data_path']
            ]
            existing_files = self._batch_check_files(image_paths)
        
        # 处理每个样本
        for sample_id, sample_data in data_info.items():
            stats['total'] += 1
            
            # 获取图像路径和文本描述
            image_path = sample_data['data_path']
            # caption = sample_data['CLIPcaption']
            caption = sample_data['refined_caption']
            # 数据验证
            if not image_path:
                stats['missing_image'] += 1
                continue
                
            if not caption:
                stats['missing_caption'] += 1
                continue
            
            if self.validate_files and image_path not in existing_files:
                stats['file_not_exists'] += 1
                continue
            
            samples.append((image_path, caption))
            stats['valid'] += 1
            
            # 限制样本数量
            if self.max_samples and len(samples) >= self.max_samples:
                break
        
        return samples, stats
    
    def _batch_check_files(self, file_paths: List[str], max_workers: int = 8) -> set:
        """批量检查文件存在性"""
        existing_files = set()
        
        def check_file(path):
            return path if os.path.exists(path) else None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(check_file, path): path for path in file_paths}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    existing_files.add(result)
        
        return existing_files
    
    def _log_stats(self):
        """记录统计信息"""
        if self.stats['total'] > 0:
            valid_rate = self.stats['valid'] / self.stats['total'] * 100
        else:
            valid_rate = 0
            
        logger.info(f"JSON数据集加载完成:")
        logger.info(f"  总样本数: {self.stats['total']}")
        logger.info(f"  有效样本数: {self.stats['valid']}")
        logger.info(f"  缺失图像: {self.stats['missing_image']}")
        logger.info(f"  缺失描述: {self.stats['missing_caption']}")
        logger.info(f"  文件不存在: {self.stats['file_not_exists']}")
        logger.info(f"  有效率: {valid_rate:.2f}%")
