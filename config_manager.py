"""
配置管理模块
负责加载和管理训练配置
"""

import json
import os
from typing import Dict, Any


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # 设置默认值
        self._set_defaults(config)
        return config
    
    def _set_defaults(self, config: Dict[str, Any]):
        """设置默认配置值"""
        # 数据配置默认值
        config.setdefault('data', {})
        config['data'].setdefault('image_size', 224)
        config['data'].setdefault('batch_size', 16)
        config['data'].setdefault('num_workers', 4)
        config['data'].setdefault('max_text_length', 256)
        
        # 训练配置默认值
        config.setdefault('training', {})
        config['training'].setdefault('learning_rate', 1e-5)
        config['training'].setdefault('weight_decay', 0.01)
        config['training'].setdefault('warmup_steps', 1000)
        config['training'].setdefault('max_epochs', 100)
        config['training'].setdefault('gradient_clip_val', 1.0)
        
        # 模型配置默认值
        config.setdefault('model', {})
        config['model'].setdefault('embed_dim', 768)
        config['model'].setdefault('temperature', 0.07)
        
    def get(self, key: str, default=None):
        """获取配置值，支持点分割的嵌套key"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value):
        """设置配置值"""
        keys = key.split('.')
        current = self.config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
