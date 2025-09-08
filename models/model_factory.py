"""
模型工厂模块
负责根据配置创建不同类型的模型
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
# 延迟导入避免不必要的依赖加载

class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> nn.Module:
        """
        根据配置创建模型
        Args:
            config: 模型配置
        Returns:
            创建的模型
        """
        model_type = config['type']
        
        if model_type == 'fetalclip':
            return ModelFactory._create_fetalclip(config)
        elif model_type == 'sam_bert':
            return ModelFactory._create_sam_bert(config)
        elif model_type == 'qwen_vl_clip':
            return ModelFactory._create_qwen_vl_clip(config)
        elif model_type == 'medclip':
            return ModelFactory._create_medclip(config)
        elif model_type == 'convnext_clip':
            return ModelFactory._create_convnext_clip(config)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    @staticmethod
    def _create_fetalclip(config: Dict[str, Any]):
        """创建FetalCLIP模型"""
        from .fetalclip_pre_model import FetalCLIPPreModel
        
        config_path = config['config_path']
        pretrained_path = config['pretrained_path']
        
        return FetalCLIPPreModel(
            config_path=config_path,
            pretrained_path=pretrained_path
        )
    
    @staticmethod
    def _create_sam_bert(config: Dict[str, Any]):
        """创建SAM+BERT模型"""
        from .ultrasam_bert_pre_model import SAMBERTPreModel
        return SAMBERTPreModel(config)
    
    @staticmethod
    def _create_qwen_vl_clip(config: Dict[str, Any]):
        """创建Qwen2.5-VL CLIP模型"""
        from .qwen_vl_clip_model import QwenVLCLIPModel
        return QwenVLCLIPModel(config)
    
    @staticmethod
    def _create_medclip(config: Dict[str, Any]):
        """创建MedCLIP包装器模型 - 完全独立，不依赖Qwen"""
        from .medclip_model import MedCLIPWrapper
        return MedCLIPWrapper(config)
    
    @staticmethod
    def _create_convnext_clip(config: Dict[str, Any]):
        """创建 ConvNeXt + RoBERTa 的 CLIP 模型"""
        from .convnext_model import build_convnext_clip_from_config, ConvNeXtCLIPTrainingWrapper
        core = build_convnext_clip_from_config(config)
        return ConvNeXtCLIPTrainingWrapper(core)
    
    @staticmethod
    def load_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
        """
        加载模型权重
        Args:
            model: 模型实例
            checkpoint_path: 权重文件路径
        Returns:
            加载权重后的模型
        """
        print(f"加载模型权重: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 移除可能的前缀
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # 加载权重
        model.load_state_dict(state_dict, strict=False)
        print("模型权重加载完成")
        
        return model
    
    @staticmethod
    def load_trained_model(weight_path: str, 
                           config_path: str, 
                           model_type: str = 'auto',
                           device: str = 'cpu') -> Tuple[nn.Module, Any]:
        """
        加载训练后的模型（使用专用加载器）
        Args:
            weight_path: 权重文件路径
            config_path: 配置文件路径
            model_type: 模型类型 ('fetalclip', 'sam_bert', 'auto')
            device: 设备
        Returns:
            model, tokenizer
        """
        # 自动检测模型类型
        if model_type == 'auto':
            model_type = ModelFactory._detect_model_type(weight_path, config_path)
            print(f"自动检测模型类型: {model_type}")

        # 使用专用加载器
        if model_type == 'fetalclip':
            # 使用 FetalCLIPPreModel 直接加载
            from .fetalclip_pre_model import FetalCLIPPreModel
            model = FetalCLIPPreModel(config_path=config_path, pretrained_path=weight_path)
            return model, model.tokenizer
        elif model_type == 'sam_bert':
            # 使用 SAMBERTPreModel 直接加载
            from .ultrasam_bert_pre_model import SAMBERTPreModel
            model = SAMBERTPreModel(config={"image_encoder": {}, "text_encoder": {}, "embed_dim": 512})
            return model, model.tokenizer
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    @staticmethod
    def _detect_model_type(weight_path: str, config_path: str = None) -> str:
        """自动检测模型类型"""
        checkpoint = torch.load(weight_path, map_location='cpu')
        # 检查权重结构
        if isinstance(checkpoint, dict):
            state_dict = checkpoint['model_state_dict']
            # state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint

        keys = list(state_dict.keys())
        has_sam_encoder = any('image_encoder' in key for key in keys)
        has_bert_encoder = any('text_encoder' in key for key in keys)
        has_projection = any('projection' in key for key in keys)

        if has_sam_encoder and has_bert_encoder:
            return 'sam_bert'

        has_visual = any('visual' in key for key in keys)
        has_transformer = any('transformer' in key for key in keys)
        has_token_embedding = any('token_embedding' in key for key in keys)

        if has_visual and has_transformer and has_token_embedding:
            return 'fetalclip'

        return 'fetalclip'
