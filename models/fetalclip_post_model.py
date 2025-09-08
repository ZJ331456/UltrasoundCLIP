"""
FetalCLIP专用模型加载器
专门处理FetalCLIP训练后权重的加载
"""

import torch
import torch.nn as nn
import open_clip
import json
import os
from typing import Optional, Tuple, Dict, Any


class FetalCLIPPostModel:
    """FetalCLIP专用加载器"""
    
    def __init__(self):
        self.model_type = "FetalCLIP"
    
    def load_model(self, 
                   weight_path: str, 
                   config_path: str, 
                   device: str = 'cpu') -> Tuple[nn.Module, Any]:
        """
        加载FetalCLIP模型
        Args:
            weight_path: 权重文件路径
            config_path: 配置文件路径
            device: 设备
        Returns:
            model, tokenizer
        """
        print(f"正在加载{self.model_type}模型...")
        print(f"  权重文件: {weight_path}")
        print(f"  配置文件: {config_path}")
        print(f"  设备: {device}")
        
        # 1. 加载配置
        config = self._load_config(config_path)
        
        # 2. 创建模型架构
        model = self._create_model_architecture(config)
        
        # 3. 加载权重
        self._load_weights(model, weight_path)
        
        # 4. 获取tokenizer
        tokenizer = self._get_tokenizer()
        
        # 5. 移动到设备并设置评估模式
        model = model.to(device)
        model.eval()
        
        print(f"{self.model_type}模型加载完成!")
        return model, tokenizer
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载FetalCLIP配置"""
        if not os.path.exists(config_path):
            print(f"  配置文件不存在: {config_path}")
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print(f"  配置文件加载成功")
        return config
    
    def _create_model_architecture(self, config: Dict[str, Any]) -> nn.Module:
        """创建FetalCLIP模型架构"""
        # 注册配置到OpenCLIP
        arch_name = 'ViT-L-14'
        open_clip.factory._MODEL_CONFIGS[arch_name] = config
        
        # 创建模型
        model, _, _ = open_clip.create_model_and_transforms(arch_name)
        
        print(f"  模型架构创建完成: {arch_name}")
        return model
    
    def _load_weights(self, model: nn.Module, weight_path: str):
        """加载FetalCLIP权重"""
        print(f"  加载权重文件: {weight_path}")
        
        # 加载checkpoint
        checkpoint = torch.load(weight_path, map_location='cpu')
        
        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"    检测到训练checkpoint格式")
                self._print_checkpoint_info(checkpoint)
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"    检测到Lightning checkpoint格式")
            else:
                state_dict = checkpoint
                print(f"    检测到直接权重格式")
        else:
            state_dict = checkpoint
            print(f"    检测到tensor格式权重")
        
        # 清理权重键名
        clean_state_dict = self._clean_state_dict(state_dict)
        
        # 处理位置编码插值
        self._interpolate_position_embedding(model, clean_state_dict)
        
        # 加载权重到模型
        missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)
        
        if missing_keys:
            print(f"    缺失的权重键 ({len(missing_keys)}): {missing_keys[:3]}...")
        if unexpected_keys:
            print(f"    多余的权重键 ({len(unexpected_keys)}): {unexpected_keys[:3]}...")
        
        print(f"  权重加载完成")
    
    def _print_checkpoint_info(self, checkpoint: Dict[str, Any]):
        """打印checkpoint信息"""
        if 'epoch' in checkpoint:
            print(f"    Epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            val_loss = checkpoint['val_loss']
            if val_loss != float('inf'):
                print(f"    Val Loss: {val_loss:.4f}")
            else:
                print(f"    Val Loss: inf")
        if 'val_acc' in checkpoint:
            print(f"    Val Acc: {checkpoint['val_acc']:.4f}")
    
    def _clean_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """清理state_dict中的键名"""
        clean_state_dict = {}
        
        for key, value in state_dict.items():
            # 移除module.前缀
            if key.startswith('module.'):
                new_key = key[7:]
            # 移除model.前缀（Lightning包装）
            elif key.startswith('model.'):
                new_key = key[6:]
            else:
                new_key = key
            
            clean_state_dict[new_key] = value
        
        return clean_state_dict
    
    def _interpolate_position_embedding(self, model: nn.Module, state_dict: Dict[str, torch.Tensor]):
        """插值位置编码（用于不同尺寸的适配）"""
        if 'visual.positional_embedding' not in state_dict:
            return
        
        try:
            pretrained_pe = state_dict['visual.positional_embedding']
            model_pe = model.visual.positional_embedding
            
            if pretrained_pe.shape != model_pe.shape:
                print(f" 位置编码形状不匹配，进行插值: {pretrained_pe.shape} -> {model_pe.shape}")
                
                # 分离class token和空间位置编码
                num_cls_token = 1
                pretrained_spatial = pretrained_pe[num_cls_token:]
                model_spatial = model_pe[num_cls_token:]
                
                # 计算网格尺寸
                N_pre = pretrained_spatial.shape[0]
                N_model = model_spatial.shape[0]
                D = pretrained_spatial.shape[1]
                
                S_pre = int(N_pre ** 0.5)
                S_model = int(N_model ** 0.5)
                
                if S_pre * S_pre == N_pre and S_model * S_model == N_model:
                    # 重塑为网格形状并进行双三次插值
                    pretrained_grid = pretrained_spatial.reshape(S_pre, S_pre, D).permute(2, 0, 1).unsqueeze(0)
                    
                    model_grid = torch.nn.functional.interpolate(
                        pretrained_grid,
                        size=(S_model, S_model),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0).reshape(N_model, D)
                    
                    # 拼接class token和插值后的空间编码
                    new_pe = torch.cat([pretrained_pe[:num_cls_token], model_grid], dim=0)
                    state_dict['visual.positional_embedding'] = new_pe
                    
                    print(f"    位置编码插值完成: {S_pre}x{S_pre} -> {S_model}x{S_model}")
                else:
                    print(f"    位置编码不是正方形网格，跳过插值")
        except Exception as e:
            print(f"    位置编码插值失败: {e}")
    
    def _get_tokenizer(self) -> Any:
        """获取FetalCLIP tokenizer"""
        arch_name = 'ViT-L-14'
        tokenizer = open_clip.get_tokenizer(arch_name)
        print(f"  Tokenizer创建完成: {arch_name}")
        return tokenizer

def load_fetalclip_post_model(weight_path: str, config_path: str, device: str) -> Tuple[nn.Module, Any]:
    """
    便捷函数：加载FetalCLIP模型
    Args:
        weight_path: 权重文件路径
        config_path: 配置文件路径
        device: 设备
    Returns:
        model, tokenizer
    """
    loader = FetalCLIPPostModel()
    return loader.load_model(weight_path, config_path, device)
