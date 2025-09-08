"""
超声图像数据集类
专门针对超声图像优化的数据集实现
整合了CLIP_mini中的超声图像预处理优化
"""

import json
import logging
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import os

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class UltrasoundDataset(BaseDataset):
    """
    超声图像专用数据集
    整合了CLIP_mini中的超声图像优化特性：
    - 等比例缩放+填充，避免图像变形
    - 超声图像特定的数据增强
    - 斑点噪声模拟
    - Gamma校正等
    """
    
    def __init__(self, 
                 json_path: str,
                 transform=None,
                 max_samples: Optional[int] = None,
                 validate_files: bool = True,
                 cache_size: int = 500,
                 target_image_size: int = 224,
                 augment: bool = True,
                 ultrasound_specific: bool = True):
        """
        初始化超声图像数据集
        Args:
            json_path: JSON标注文件路径
            transform: 图像变换
            max_samples: 最大样本数量
            validate_files: 是否验证文件存在性
            cache_size: 图像缓存大小
            target_image_size: 目标图像尺寸
            augment: 是否启用数据增强
            ultrasound_specific: 是否启用超声图像特定优化
        """
        self.json_path = json_path
        self.target_image_size = target_image_size
        self.augment = augment
        self.ultrasound_specific = ultrasound_specific
        
        # 调用父类初始化
        super().__init__(transform=transform, max_samples=max_samples, cache_size=cache_size)
        
        # 加载样本数据
        self.samples = self._load_samples()
        
        # 如果启用了超声图像特定优化，创建专门的变换
        if self.ultrasound_specific:
            self.ultrasound_transforms = self._create_ultrasound_transforms()
        
        # 验证文件存在性
        if validate_files:
            self._validate_files()
        
        # 统计信息
        self.stats = {
            'total_samples': len(self.samples),
            'target_size': target_image_size,
            'augmentation_enabled': augment,
            'ultrasound_optimized': ultrasound_specific
        }
        
        logger.info(f"超声图像数据集加载完成: {len(self.samples)} 个样本")
    
    def _load_samples(self) -> List[Tuple[str, str]]:
        """加载样本数据，支持多种JSON格式"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 兼容两种常见顶层结构
            if isinstance(data, dict) and 'DataInfo' in data:
                data = data['DataInfo']
            
            samples = []
            for item in (data.values() if isinstance(data, dict) else data):
                # 兼容多种键名
                img_path = item.get('data_path') or item.get('image_path') or item.get('image')
                caption = item.get('refined_caption') or item.get('CLIPcaption') or item.get('caption') or item.get('text')
                
                if img_path and caption:
                    samples.append((img_path, caption))
            
            # 限制样本数量
            if self.max_samples and len(samples) > self.max_samples:
                samples = samples[:self.max_samples]
                logger.info(f"限制样本数量为: {self.max_samples}")
            
            return samples
            
        except Exception as e:
            logger.error(f"加载样本数据失败: {e}")
            return []
    
    def _validate_files(self):
        """验证文件存在性"""
        valid_samples = []
        invalid_count = 0
        
        for img_path, caption in self.samples:
            if os.path.exists(img_path):
                valid_samples.append((img_path, caption))
            else:
                invalid_count += 1
                logger.warning(f"图像文件不存在: {img_path}")
        
        if invalid_count > 0:
            logger.warning(f"发现 {invalid_count} 个无效文件")
        
        self.samples = valid_samples
        self.stats['valid_samples'] = len(valid_samples)
        self.stats['invalid_files'] = invalid_count
    
    def _create_ultrasound_transforms(self):
        """创建超声图像特定的变换"""
        transforms_list = []
        
        if self.augment:
            # PIL阶段的几何/光照增强（对超声友好且幅度很小）
            transforms_list.extend([
                # 轻微旋转，模拟探头角度变化
                transforms.RandomRotation(
                    degrees=5,  # 最大 ±5°
                    fill=(128, 128, 128)  # 旋转后空洞填充为中性灰
                ),
                # 轻微水平翻转（超声图像通常不建议，但可以增加数据多样性）
                transforms.RandomHorizontalFlip(p=0.2),
                # 轻微模糊，模拟成像不稳
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3)
                ], p=0.2),
                # 亮度/对比度微调（幅度很小）
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.10, contrast=0.10)
                ], p=0.3),
                # 随机Gamma校正
                RandomGamma(gamma_range=(0.9, 1.1), p=0.3),
            ])
        
        # 等比例缩放+填充变换
        transforms_list.append(ResizeWithPadding(self.target_image_size))
        
        # 转换为张量
        transforms_list.append(transforms.ToTensor())
        
        if self.augment:
            # 张量阶段的噪声与标准化
            transforms_list.extend([
                # 斑点噪声（超声常见噪声）
                SpeckleNoise(sigma_range=(0.0, 0.05), p=0.5),
                # 标准化到模型预期分布
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            # 验证时只做标准化
            transforms_list.append(
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )
        
        return transforms.Compose(transforms_list)
    
    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """加载并预处理图像"""
        try:
            image = Image.open(image_path)
            
            # 确保是RGB格式
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            return image
            
        except Exception as e:
            logger.warning(f"图像加载失败 {image_path}: {e}")
            return None
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        image_path, text = self.samples[idx]
        
        # 加载图像
        image = self._cached_load_image(image_path)
        
        if image is None:
            image = self._get_placeholder_image()
            logger.warning(f"使用占位符图像替代: {image_path}")
        
        # 应用超声图像特定变换
        if self.ultrasound_specific and hasattr(self, 'ultrasound_transforms'):
            image = self.ultrasound_transforms(image)
        elif self.transform:
            # 使用通用变换
            image = self.transform(image)
        
        return {
            'images': image,  # 保持与现有接口一致
            'text': text,
            'image_path': image_path
        }


class ResizeWithPadding:
    """
    等比例缩放到不超过目标尺寸的最大边，然后用常数像素在四周"对称"填充至正方形
    这样不会拉伸图像内容，适配模型的固定输入尺寸
    """
    
    def __init__(self, target_size: int, fill_color: Tuple[int, int, int] = (128, 128, 128)):
        self.target_size = target_size
        self.fill_color = fill_color
    
    def __call__(self, im: Image.Image) -> Image.Image:
        orig_w, orig_h = im.size
        
        # 若任一边超过目标尺寸，按比例缩放
        if orig_w > self.target_size or orig_h > self.target_size:
            scale = self.target_size / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            im = im.resize((new_w, new_h), Image.BILINEAR)
        else:
            new_w, new_h = orig_w, orig_h
        
        # 计算需要填充的像素数
        pad_w, pad_h = self.target_size - new_w, self.target_size - new_h
        
        # 左、上、右、下四个方向的填充，尽可能左右/上下对称
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        im = ImageOps.expand(im, padding, fill=self.fill_color)
        
        return im


class RandomGamma:
    """
    随机gamma校正（轻度），模拟探头/增益变化对亮度的影响
    在PIL图像阶段执行，避免精度过早丢失
    """
    
    def __init__(self, gamma_range: Tuple[float, float] = (0.9, 1.1), p: float = 0.3):
        self.gamma_range = gamma_range
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() < self.p:
            gamma = torch.empty(1).uniform_(*self.gamma_range).item()
            # torchvision F.adjust_gamma支持PIL，gain=1.0表示不额外缩放
            return TF.adjust_gamma(img, gamma=gamma, gain=1.0)
        return img


class SpeckleNoise:
    """
    斑点噪声（speckle）：超声常见噪声的简化模拟（乘性噪声）
    在张量阶段执行：x <- clamp(x + x * N(0, sigma), 0, 1)
    """
    
    def __init__(self, sigma_range: Tuple[float, float] = (0.0, 0.05), p: float = 0.5):
        self.sigma_range = sigma_range
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            sigma = torch.empty(1).uniform_(*self.sigma_range).item()
            noise = torch.randn_like(x) * sigma
            x = x + x * noise  # 乘性扰动（依赖局部强度）
            x = x.clamp(0.0, 1.0)  # 保持在[0,1]，便于后续标准化
        return x


# 在工厂中添加超声数据集支持
def register_ultrasound_dataset():
    """注册超声数据集到工厂中"""
    # 这个方法可以在factory.py中调用，将超声数据集集成到现有系统中
    pass
