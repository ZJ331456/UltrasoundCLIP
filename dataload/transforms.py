"""
数据变换模块
包含各种图像预处理和数据增强方法
"""

import torchvision.transforms as T
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any
import random
import torch


def ensure_rgb_image(img_array: np.ndarray) -> Image.Image:
    """确保numpy数组转换为RGB格式的PIL图像"""
    if len(img_array.shape) == 2:
        # 灰度图转RGB
        img_array = np.stack([img_array, img_array, img_array], axis=2)
    elif len(img_array.shape) == 3:
        if img_array.shape[2] == 1:
            # 单通道转RGB
            img_array = np.repeat(img_array, 3, axis=2)
        elif img_array.shape[2] > 3:
            # 多通道截取前3个
            img_array = img_array[:, :, :3]
    
    # 确保数据类型为uint8
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def force_rgb_conversion(img: Image.Image) -> Image.Image:
    """强制转换图像为标准3通道RGB格式"""
    # 转换为numpy数组进行处理
    img_array = np.array(img)
    
    # 确保是3通道
    if len(img_array.shape) == 2:
        # 灰度图
        img_array = np.stack([img_array, img_array, img_array], axis=2)
    elif len(img_array.shape) == 3:
        channels = img_array.shape[2]
        if channels == 1:
            # 单通道转RGB
            img_array = np.repeat(img_array, 3, axis=2)
        elif channels == 4:
            # RGBA转RGB，丢弃alpha通道
            img_array = img_array[:, :, :3]
        elif channels > 4:
            # 多通道，取前3个
            img_array = img_array[:, :, :3]
        elif channels == 2:
            # 双通道，扩展为3通道
            img_array = np.concatenate([img_array, img_array[:, :, :1]], axis=2)
    
    # 确保数据类型和范围
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # 转回PIL图像并确保模式
    result_img = Image.fromarray(img_array)
    if result_img.mode != 'RGB':
        result_img = result_img.convert('RGB')
    
    return result_img


class UltrasoundAugmentation:
    """超声图片特定的数据增强"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化增强版超声图片数据增强
        Args:
            config: 增强配置字典
        """
        self.enabled = config['enabled']
        self.contrast_enhancement = config['contrast_enhancement']
        self.noise_reduction = config['noise_reduction']
        self.edge_enhancement = config['edge_enhancement']
        self.gamma_correction = config['gamma_correction']
        self.rotation_range = config['rotation_range']
        self.brightness_range = config['brightness_range']
        self.contrast_range = config['contrast_range']
        
        # 新增增强选项
        self.elastic_deformation = config.get('elastic_deformation', False)
        self.cutout_probability = config.get('cutout_probability', 0.0)
        self.mixup_alpha = config.get('mixup_alpha', 0.0)
        self.cutmix_alpha = config.get('cutmix_alpha', 0.0)
        self.random_erasing = config.get('random_erasing', 0.0)
        self.gaussian_blur = config.get('gaussian_blur', 0.0)
        self.speckle_noise = config.get('speckle_noise', 0.0)
        self.motion_blur = config.get('motion_blur', 0.0)
        
        # 超声图片不建议水平翻转，保持医学结构的完整性
        self.horizontal_flip = False
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """执行超声图片特定的数据增强"""
        if not self.enabled:
            return img
        
        # 对比度增强
        if self.contrast_enhancement:
            img = self._enhance_contrast(img)
        
        # 降噪处理
        if self.noise_reduction:
            img = self._reduce_noise(img)
        
        # 边缘增强
        if self.edge_enhancement:
            img = self._enhance_edges(img)
        
        # 伽马校正
        if self.gamma_correction:
            img = self._gamma_correction(img)
        
        # 亮度调整
        if self.brightness_range:
            img = self._adjust_brightness(img)
        
        # 对比度调整
        if self.contrast_range:
            img = self._adjust_contrast(img)
        
        # 小角度旋转（保持医学结构）
        if self.rotation_range:
            img = self._rotate_image(img)
        
        # 弹性变形（新增）
        if self.elastic_deformation and random.random() < 0.3:
            img = self._elastic_deformation(img)
        
        # Cutout增强（新增）
        if random.random() < self.cutout_probability:
            img = self._cutout_augmentation(img)
        
        # 高斯模糊
        if random.random() < self.gaussian_blur:
            img = self._gaussian_blur(img)
        
        # 斑点噪声（模拟超声伪影）
        if random.random() < self.speckle_noise:
            img = self._speckle_noise(img)
        
        # 运动模糊
        if random.random() < self.motion_blur:
            img = self._motion_blur(img)
        
        # 随机擦除
        if random.random() < self.random_erasing:
            img = self._random_erasing(img)
        
        # 最终强制转换为标准RGB格式
        return force_rgb_conversion(img)
    
    def _elastic_deformation(self, img: Image.Image) -> Image.Image:
        """弹性变形增强，模拟超声探头角度变化"""
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # 生成变形网格
        dx = np.random.uniform(-1, 1, (h//20, w//20)) * 5
        dy = np.random.uniform(-1, 1, (h//20, w//20)) * 5
        
        # 插值到原始尺寸
        dx = cv2.resize(dx, (w, h))
        dy = cv2.resize(dy, (w, h))
        
        # 生成变形映射
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # 应用变形
        if len(img_array.shape) == 3:
            deformed = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR)
        else:
            deformed = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR)
        
        # 确保返回正确的图像模式
        return force_rgb_conversion(Image.fromarray(deformed))
    
    def _cutout_augmentation(self, img: Image.Image) -> Image.Image:
        """Cutout增强，模拟超声图像中的阴影遮挡"""
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # 随机选择cutout区域
        cutout_size = min(h, w) // 8
        x = random.randint(0, w - cutout_size)
        y = random.randint(0, h - cutout_size)
        
        # 应用cutout（用平均值填充）
        if len(img_array.shape) == 3:
            cutout_value = img_array.mean()
            img_array[y:y+cutout_size, x:x+cutout_size] = cutout_value
        else:
            cutout_value = img_array.mean()
            img_array[y:y+cutout_size, x:x+cutout_size] = cutout_value
        
        # 确保返回RGB格式
        return force_rgb_conversion(Image.fromarray(img_array))
    
    def _enhance_contrast(self, img: Image.Image) -> Image.Image:
        """增强对比度"""
        # 使用CLAHE（对比度受限的自适应直方图均衡化）
        img_array = np.array(img)
        if len(img_array.shape) == 3:
            # 彩色图像，转换为LAB空间
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # 灰度图像
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_array = clahe.apply(img_array)
        
        # 确保返回RGB格式 - 统一为3通道
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            return Image.fromarray(img_array.astype(np.uint8))
        elif len(img_array.shape) == 2:
            # 灰度图转RGB
            img_array = np.stack([img_array, img_array, img_array], axis=2)
            return Image.fromarray(img_array.astype(np.uint8))
        else:
            # 其他情况，确保为3通道
            if img_array.shape[2] > 3:
                img_array = img_array[:, :, :3]
            elif img_array.shape[2] < 3:
                img_array = np.repeat(img_array, 3 // img_array.shape[2] + 1, axis=2)[:, :, :3]
            return Image.fromarray(img_array.astype(np.uint8))
    
    def _reduce_noise(self, img: Image.Image) -> Image.Image:
        """降噪处理"""
        img_array = np.array(img)
        if len(img_array.shape) == 3:
            # 彩色图像，分别处理每个通道
            denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        else:
            # 灰度图像
            denoised = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)
        
        return force_rgb_conversion(ensure_rgb_image(denoised))
    
    def _enhance_edges(self, img: Image.Image) -> Image.Image:
        """边缘增强"""
        # 使用UnsharpMask增强边缘
        img_array = np.array(img)
        if len(img_array.shape) == 3:
            # 彩色图像，转换为灰度进行边缘检测
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # 使用高斯模糊创建模糊版本
        blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
        # 计算锐化图像
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        
        if len(img_array.shape) == 3:
            # 将锐化结果应用到所有通道
            result = img_array.copy()
            result[:, :, 0] = np.clip(result[:, :, 0] * (sharpened / 255.0), 0, 255)
            result[:, :, 1] = np.clip(result[:, :, 1] * (sharpened / 255.0), 0, 255)
            result[:, :, 2] = np.clip(result[:, :, 2] * (sharpened / 255.0), 0, 255)
            result_img = Image.fromarray(result.astype(np.uint8))
            if result_img.mode != 'RGB':
                result_img = result_img.convert('RGB')
            return result_img
        else:
            result_img = Image.fromarray(sharpened)
            if result_img.mode != 'RGB':
                result_img = result_img.convert('RGB')
            return result_img
    
    def _gamma_correction(self, img: Image.Image) -> Image.Image:
        """伽马校正"""
        gamma = random.uniform(self.gamma_correction[0], self.gamma_correction[1])
        img_array = np.array(img, dtype=np.float32) / 255.0
        corrected = np.power(img_array, gamma)
        corrected = np.clip(corrected * 255, 0, 255).astype(np.uint8)
        result_img = Image.fromarray(corrected)
        if result_img.mode != 'RGB':
            result_img = result_img.convert('RGB')
        return result_img
    
    def _adjust_brightness(self, img: Image.Image) -> Image.Image:
        """调整亮度"""
        factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    def _adjust_contrast(self, img: Image.Image) -> Image.Image:
        """调整对比度"""
        factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    def _rotate_image(self, img: Image.Image) -> Image.Image:
        """小角度旋转"""
        angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
        return img.rotate(angle, fillcolor=(128, 128, 128))
    
    def _gaussian_blur(self, img: Image.Image) -> Image.Image:
        """高斯模糊增强"""
        # 随机选择模糊半径
        radius = random.uniform(0.5, 2.0)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _speckle_noise(self, img: Image.Image) -> Image.Image:
        """斑点噪声，模拟超声图像特有的伪影"""
        img_array = np.array(img).astype(np.float32)
        
        # 生成乘性噪声
        noise = np.random.gamma(shape=1.0, scale=1.0, size=img_array.shape)
        noise = (noise - 1.0) * 0.1  # 控制噪声强度
        
        # 应用噪声
        noisy = img_array * (1 + noise)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return force_rgb_conversion(ensure_rgb_image(noisy))
    
    def _motion_blur(self, img: Image.Image) -> Image.Image:
        """运动模糊增强"""
        img_array = np.array(img)
        
        # 随机选择模糊方向和强度
        size = random.randint(3, 7)
        angle = random.randint(0, 180)
        
        # 创建运动模糊核
        kernel = np.zeros((size, size))
        kernel[size//2, :] = 1.0
        kernel = kernel / size
        
        # 旋转核
        center = (size//2, size//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (size, size))
        
        # 应用模糊
        if len(img_array.shape) == 3:
            blurred = cv2.filter2D(img_array, -1, kernel)
        else:
            blurred = cv2.filter2D(img_array, -1, kernel)
        
        # 确保返回RGB图像
        return force_rgb_conversion(Image.fromarray(blurred))
    
    def _random_erasing(self, img: Image.Image) -> Image.Image:
        """随机擦除增强"""
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # 随机选择擦除区域
        area = h * w
        target_area = random.uniform(0.02, 0.4) * area
        aspect_ratio = random.uniform(0.3, 3.3)
        
        erase_h = int(round(np.sqrt(target_area * aspect_ratio)))
        erase_w = int(round(np.sqrt(target_area / aspect_ratio)))
        
        if erase_h < h and erase_w < w:
            x = random.randint(0, w - erase_w)
            y = random.randint(0, h - erase_h)
            
            # 用随机值填充
            if len(img_array.shape) == 3:
                random_values = np.random.randint(0, 256, (erase_h, erase_w, img_array.shape[2]))
            else:
                random_values = np.random.randint(0, 256, (erase_h, erase_w))
            
            img_array[y:y+erase_h, x:x+erase_w] = random_values
        
        # 确保返回RGB格式
        return force_rgb_conversion(Image.fromarray(img_array))


class KeepAspectRatioResize:
    """保持宽高比的resize变换"""
    
    def __init__(self, target_size: int, fill_color: Tuple[int, int, int] = (0, 0, 0)):
        self.target_size = target_size
        self.fill_color = fill_color
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """执行变换"""
        w, h = img.size
        scale = min(self.target_size / w, self.target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 缩放图像
        img = img.resize((new_w, new_h), Image.BICUBIC)
        
        # 创建目标尺寸的空白图像并居中粘贴
        result = Image.new('RGB', (self.target_size, self.target_size), self.fill_color)
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        result.paste(img, (paste_x, paste_y))
        
        return result


class EnsureRGBTransform:
    """确保图像为RGB格式的变换"""
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """转换图像为RGB格式"""
        if img.mode != 'RGB':
            if img.mode in ['RGBA', 'LA']:
                # 如果是RGBA或LA，转换为RGB
                img = img.convert('RGB')
            elif img.mode == 'L':
                # 如果是灰度图，转换为RGB
                img = img.convert('RGB')
            elif img.mode in ['P', 'CMYK']:
                # 如果是调色板或CMYK，先转换为RGB
                img = img.convert('RGB')
            else:
                # 其他情况，强制转换为RGB
                img = img.convert('RGB')
        return img


class TransformFactory:
    """数据变换工厂类"""
    
    # 标准化参数
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    @classmethod
    def get_ultrasound_transforms(cls, 
                                image_size: int = 588,
                                config: Dict[str, Any] = None,
                                is_training: bool = False) -> T.Compose:
        """获取超声图片特定的数据变换"""
        transforms = []
        
        # 基础变换
        transforms.append(T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC))
        
        # 训练时的超声图片特定增强
        if is_training and config:
            ultrasound_aug = config['ultrasound_augmentation']
            if ultrasound_aug['enabled']:
                transforms.append(UltrasoundAugmentation(ultrasound_aug))
        
        # 标准化
        normalization = config['image_normalization']
        if normalization == 'clip':
            mean, std = cls.CLIP_MEAN, cls.CLIP_STD
        else:
            mean, std = cls.IMAGENET_MEAN, cls.IMAGENET_STD
        
        transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        
        return T.Compose(transforms)
    
    @classmethod
    def get_clip_transforms(cls, image_size: int = 224, is_training: bool = False) -> T.Compose:
        """获取CLIP模型的数据变换"""
        transforms = []
        
        # 基础变换
        transforms.append(T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC))
        
        # 训练时的数据增强
        if is_training:
            transforms.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                T.RandomRotation(degrees=5),
            ])
        
        # 最终变换
        transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=cls.CLIP_MEAN, std=cls.CLIP_STD)
        ])
        
        return T.Compose(transforms)
    
    @classmethod
    def get_sam_transforms(cls, image_size: int = 1024) -> T.Compose:
        """获取SAM模型的数据变换"""
        return T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=cls.IMAGENET_MEAN, std=cls.IMAGENET_STD)
        ])
    
    @classmethod
    def get_custom_transforms(cls, 
                            image_size: int = 224,
                            mean: List[float] = None,
                            std: List[float] = None,
                            is_training: bool = False) -> T.Compose:
        """获取自定义数据变换"""
        if mean is None:
            mean = cls.CLIP_MEAN
        if std is None:
            std = cls.CLIP_STD
        
        transforms = []
        transforms.append(T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC))
        
        if is_training:
            transforms.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ])
        
        transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        
        return T.Compose(transforms)
    
    @classmethod
    def get_keep_aspect_transforms(cls, image_size: int = 224) -> T.Compose:
        """获取保持宽高比的数据变换"""
        return T.Compose([
            KeepAspectRatioResize(image_size),
            T.ToTensor(),
            T.Normalize(mean=cls.CLIP_MEAN, std=cls.CLIP_STD)
        ])


class MedicalSafeAugmentation:
    """
    医学安全数据增强策略 - 对照组实验专用
    专为医学图像设计的保守增强策略，保护医学结构完整性
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化医学安全增强
        Args:
            config: 增强配置字典
        """
        self.enabled = config.get('enabled', True)
        self.use_medical_safe = config.get('use_medical_optimized', False)
        
        # 基础参数
        self.gamma_correction = config.get('gamma_correction', [0.95, 1.05])
        self.rotation_range = config.get('rotation_range', [-2, 2])
        self.brightness_range = config.get('brightness_range', [0.95, 1.05])
        self.contrast_range = config.get('contrast_range', [0.95, 1.05])
        
        # 安全增强策略配置
        self.safe_policy = config.get('safe_augmentation_policy', {})
        self.no_horizontal_flip = self.safe_policy.get('no_horizontal_flip', True)
        self.preserve_anatomy = self.safe_policy.get('preserve_anatomy', True)
        self.max_rotation_degrees = self.safe_policy.get('max_rotation_degrees', 2)
        self.color_jitter_strength = self.safe_policy.get('color_jitter_strength', 0.05)
        
        # 禁用的高风险增强
        self.cutout_probability = 0.0  # 禁用cutout
        self.mixup_alpha = 0.0         # 禁用mixup
        self.cutmix_alpha = 0.0        # 禁用cutmix
        self.random_erasing = 0.0      # 禁用随机擦除
        self.gaussian_blur = 0.0       # 禁用高斯模糊
        self.elastic_deformation = False # 禁用弹性变形
        
        print(f"    - 医学安全增强: {'启用' if self.use_medical_safe else '禁用'}")
        if self.use_medical_safe:
            print(f"    - 禁用水平翻转: {self.no_horizontal_flip}")
            print(f"    - 保护解剖结构: {self.preserve_anatomy}")
            print(f"    - 最大旋转角度: {self.max_rotation_degrees}°")
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """执行医学安全数据增强"""
        if not self.enabled or not self.use_medical_safe:
            return img
        
        img = self._apply_safe_augmentations(img)
        return img
    
    def _apply_safe_augmentations(self, img: Image.Image) -> Image.Image:
        """应用安全的医学图像增强"""
        # 1. 非常保守的几何变换
        if random.random() < 0.3:  # 30%概率应用旋转
            angle = random.uniform(-self.max_rotation_degrees, self.max_rotation_degrees)
            img = img.rotate(angle, fillcolor=128)  # 使用灰色填充
        
        # 2. 轻微的颜色调整
        if random.random() < 0.5:  # 50%概率应用亮度调整
            brightness_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)
        
        if random.random() < 0.5:  # 50%概率应用对比度调整
            contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)
        
        # 3. 轻微的gamma校正
        if random.random() < 0.3:  # 30%概率应用gamma校正
            gamma = random.uniform(self.gamma_correction[0], self.gamma_correction[1])
            img = self._apply_gamma_correction(img, gamma)
        
        # 绝对不应用的变换：
        # - 水平翻转（会改变医学图像的方向）
        # - 垂直翻转（不符合医学影像习惯）
        # - 强烈的几何变形（会破坏解剖结构）
        # - 随机擦除/cutout（会遮挡重要的医学信息）
        # - mixup/cutmix（会混合不同的医学案例）
        
        return img
    
    def _apply_gamma_correction(self, img: Image.Image, gamma: float) -> Image.Image:
        """应用gamma校正"""
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.power(img_array, 1.0 / gamma)
        img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)


def create_medical_safe_transforms(image_size: int, is_training: bool = True, 
                                 normalization: str = "imagenet",
                                 augmentation_config: dict = None) -> T.Compose:
    """
    创建医学安全的数据变换流水线（对照组实验专用）
    Args:
        image_size: 目标图像尺寸
        is_training: 是否为训练模式
        normalization: 标准化类型 ("imagenet", "clip", "custom")
        augmentation_config: 增强配置
    Returns:
        变换流水线
    """
    transforms = []
    
    if is_training:
        # 训练时的医学安全增强
        if augmentation_config and augmentation_config.get('use_medical_optimized', False):
            transforms.append(MedicalSafeAugmentation(augmentation_config))
    
    # 基础变换
    transforms.extend([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    
    # 标准化
    if normalization == "imagenet":
        # 使用ImageNet预训练的标准化参数
        transforms.append(T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
    elif normalization == "clip":
        # CLIP模型的标准化参数
        transforms.append(T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ))
    else:
        # 自定义标准化（如果提供）
        pass
    
    return T.Compose(transforms)


class AdvancedAugmentation:
    """
    高级数据增强类 - 第三轮实验专用
    包含RandAugment、透视变换等高级增强技术
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化高级数据增强
        Args:
            config: 高级增强配置字典
        """
        self.enabled = config.get('enabled', False)
        self.randaugment_config = config.get('randaugment', {})
        self.perspective_distortion = config.get('perspective_distortion', 0.0)
        
        # RandAugment配置
        self.use_randaugment = self.randaugment_config.get('enabled', False)
        self.randaugment_magnitude = self.randaugment_config.get('magnitude', 8)
        self.randaugment_num_ops = self.randaugment_config.get('num_ops', 2)
        
        if self.enabled:
            print(f"    - 高级数据增强: 启用")
            if self.use_randaugment:
                print(f"    - RandAugment: 启用 (magnitude={self.randaugment_magnitude}, ops={self.randaugment_num_ops})")
            if self.perspective_distortion > 0:
                print(f"    - 透视变换: {self.perspective_distortion}")
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """执行高级数据增强"""
        if not self.enabled:
            return img
        
        # 应用RandAugment
        if self.use_randaugment and random.random() < 0.5:
            img = self._apply_randaugment(img)
        
        # 应用透视变换
        if self.perspective_distortion > 0 and random.random() < self.perspective_distortion:
            img = self._apply_perspective_distortion(img)
        
        return img
    
    def _apply_randaugment(self, img: Image.Image) -> Image.Image:
        """应用RandAugment数据增强"""
        # RandAugment操作列表（医学图像安全版）
        operations = [
            self._auto_contrast,
            self._equalize,
            self._rotate,
            self._solarize,
            self._color,
            self._posterize,
            self._brightness,
            self._contrast,
            self._sharpness,
        ]
        
        # 随机选择指定数量的操作
        selected_ops = random.sample(operations, min(self.randaugment_num_ops, len(operations)))
        
        for op in selected_ops:
            magnitude = random.randint(1, self.randaugment_magnitude)
            img = op(img, magnitude)
        
        return img
    
    def _auto_contrast(self, img: Image.Image, magnitude: int) -> Image.Image:
        """自动对比度调整"""
        return ImageOps.autocontrast(img)
    
    def _equalize(self, img: Image.Image, magnitude: int) -> Image.Image:
        """直方图均衡化"""
        return ImageOps.equalize(img)
    
    def _rotate(self, img: Image.Image, magnitude: int) -> Image.Image:
        """旋转（医学安全版：小角度）"""
        # 限制旋转角度，保护医学结构
        max_angle = min(magnitude * 2, 10)  # 最大10度
        angle = random.uniform(-max_angle, max_angle)
        return img.rotate(angle, fillcolor=(128, 128, 128))
    
    def _solarize(self, img: Image.Image, magnitude: int) -> Image.Image:
        """曝光处理"""
        threshold = 256 - magnitude * 20
        return ImageOps.solarize(img, threshold)
    
    def _color(self, img: Image.Image, magnitude: int) -> Image.Image:
        """颜色调整"""
        factor = 1 + magnitude * 0.05
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)
    
    def _posterize(self, img: Image.Image, magnitude: int) -> Image.Image:
        """色调分离"""
        bits = max(1, 8 - magnitude)
        return ImageOps.posterize(img, bits)
    
    def _brightness(self, img: Image.Image, magnitude: int) -> Image.Image:
        """亮度调整"""
        factor = 1 + magnitude * 0.05
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    def _contrast(self, img: Image.Image, magnitude: int) -> Image.Image:
        """对比度调整"""
        factor = 1 + magnitude * 0.05
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    def _sharpness(self, img: Image.Image, magnitude: int) -> Image.Image:
        """锐度调整"""
        factor = 1 + magnitude * 0.05
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)
    
    def _apply_perspective_distortion(self, img: Image.Image) -> Image.Image:
        """应用透视变换"""
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # 随机生成透视变换的四个角点偏移
        offset = min(h, w) * 0.1  # 最大偏移为图像尺寸的10%
        
        # 原始四个角点
        src_points = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])
        
        # 扰动后的四个角点
        dst_points = np.float32([
            [random.uniform(-offset, offset), random.uniform(-offset, offset)],
            [w + random.uniform(-offset, offset), random.uniform(-offset, offset)],
            [w + random.uniform(-offset, offset), h + random.uniform(-offset, offset)],
            [random.uniform(-offset, offset), h + random.uniform(-offset, offset)]
        ])
        
        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 应用透视变换
        transformed = cv2.warpPerspective(img_array, matrix, (w, h), borderValue=(128, 128, 128))
        
        return Image.fromarray(transformed)


def create_enhanced_transforms(image_size: int, 
                             is_training: bool = True,
                             augmentation_config: dict = None,
                             normalization: str = "imagenet") -> T.Compose:
    """
    创建增强版数据变换流水线 - 第三轮实验专用
    Args:
        image_size: 目标图像尺寸
        is_training: 是否为训练模式
        augmentation_config: 增强配置
        normalization: 标准化类型
    Returns:
        增强版变换流水线
    """
    transforms = []
    
    if is_training and augmentation_config:
        # 使用增强版超声数据增强
        if augmentation_config.get('enabled', False):
            # 检查是否需要高级增强
            advanced_config = augmentation_config.get('advanced_augmentations', {})
            if advanced_config.get('enabled', False):
                # 创建带有高级增强的超声增强器
                enhanced_augmentor = UltrasoundAugmentation(augmentation_config)
                advanced_augmentor = AdvancedAugmentation(advanced_config)
                
                class CombinedAugmentation:
                    def __init__(self, base_aug, advanced_aug):
                        self.base_aug = base_aug
                        self.advanced_aug = advanced_aug
                    
                    def __call__(self, img):
                        img = self.base_aug(img)
                        img = self.advanced_aug(img)
                        return img
                
                transforms.append(CombinedAugmentation(enhanced_augmentor, advanced_augmentor))
            else:
                # 只使用基础超声增强
                transforms.append(UltrasoundAugmentation(augmentation_config))
    
    # 基础变换
    transforms.extend([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    
    # 标准化
    if normalization == "imagenet":
        transforms.append(T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
    elif normalization == "clip":
        transforms.append(T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ))
    
    return T.Compose(transforms)
