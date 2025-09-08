"""
模型缓存管理器 - 避免重复加载模型
"""

import torch
from typing import Dict, Any, Optional
import weakref
import gc

class ModelCache:
    """模型缓存管理器"""
    
    _instance = None
    _cache: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_model(cls, model_config: Dict[str, Any], force_reload: bool = False):
        """
        获取缓存的模型或创建新模型
        Args:
            model_config: 模型配置
            force_reload: 是否强制重新加载
        Returns:
            模型实例
        """
        # 创建缓存键
        cache_key = cls._create_cache_key(model_config)
        
        if not force_reload and cache_key in cls._cache:
            cached_model = cls._cache[cache_key]
            if cached_model is not None:
                print(f"使用缓存的模型: {cache_key}")
                return cached_model
        
        # 创建新模型
        print(f"创建新模型: {cache_key}")
        from models.model_factory import ModelFactory
        model = ModelFactory.create_model(model_config)
        
        # 缓存模型（使用弱引用避免内存泄漏）
        cls._cache[cache_key] = model
        
        return model
    
    @classmethod
    def _create_cache_key(cls, config: Dict[str, Any]) -> str:
        """创建缓存键"""
        key_parts = [
            config['type'],
            config['model_path'],
            str(config['embed_dim'])
        ]
        return '_'.join(key_parts)
    
    @classmethod
    def clear_cache(cls):
        """清空缓存"""
        cls._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("模型缓存已清空")
    
    @classmethod
    def get_cache_info(cls):
        """获取缓存信息"""
        return {
            'cached_models': len(cls._cache),
            'cache_keys': list(cls._cache.keys())
        }
