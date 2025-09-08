# 数据加载模块

## 概述

数据加载模块负责处理超声图片和文本数据的加载、预处理和数据增强。特别针对超声图片的特点进行了优化。

## 主要功能

### 1. 超声图片特定的数据增强

#### 核心特性
- **对比度增强**: 使用CLAHE（对比度受限的自适应直方图均衡化）
- **降噪处理**: 基于非局部均值去噪算法
- **边缘增强**: 使用UnsharpMask技术增强医学结构边缘
- **伽马校正**: 动态调整图像亮度分布
- **亮度/对比度调整**: 小范围随机调整，保持医学结构完整性

#### 医学图像特殊考虑
- **无水平翻转**: 保持医学结构的正确方向
- **小角度旋转**: 仅允许±3度旋转，避免破坏医学结构
- **保守的增强**: 增强参数范围较小，确保医学信息不丢失

### 2. 支持的数据格式

- **JSON格式**: 标准JSON文件，包含图像路径和文本描述
- **JSONL格式**: 每行一个JSON对象，适合大数据集

### 3. 图像预处理

- **尺寸调整**: 支持任意尺寸，默认588x588（适合超声图片）
- **标准化**: 支持ImageNet和CLIP两种标准化方式
- **保持宽高比**: 可选择保持原始图像宽高比

## 配置示例

### 超声图片数据增强配置

```json
{
  "data": {
    "image_size": 588,
    "image_normalization": "imagenet",
    "ultrasound_augmentation": {
      "enabled": true,
      "contrast_enhancement": true,
      "noise_reduction": true,
      "edge_enhancement": true,
      "gamma_correction": [0.8, 1.2],
      "rotation_range": [-3, 3],
      "brightness_range": [0.9, 1.1],
      "contrast_range": [0.9, 1.1]
    }
  }
}
```

### 训练配置

```json
{
  "training": {
    "batch_size": 32,
    "learning_rate": 1e-5,
    "scheduler": {
      "type": "cosine_with_restarts",
      "T_0": 1000,
      "T_mult": 2
    },
    "loss": {
      "type": "clip",
      "temperature": 0.05,
      "label_smoothing": 0.1
    }
  }
}
```

## 使用方法

### 1. 基本使用

```python
from dataload import DataLoaderFactory

# 创建数据集
datasets = DataLoaderFactory.create_datasets(config_dict)

# 创建数据加载器
dataloaders = DataLoaderFactory.create_dataloaders(datasets, config_dict)
```

### 2. 自定义变换

```python
from dataload.transforms import TransformFactory

# 获取超声图片特定的变换
transforms = TransformFactory.get_ultrasound_transforms(
    image_size=588,
    config=data_config,
    is_training=True
)
```

### 3. 测试数据增强

```bash
# 运行测试脚本
python test_ultrasound_augmentation.py
```

## 技术特点

### 1. 医学图像优化
- 基于超声图片特点的数据增强策略
- 保持医学结构完整性的预处理方法
- 针对医学图像的标准化参数

### 2. 性能优化
- 支持多进程数据加载
- 内存缓存机制
- GPU友好的数据格式

### 3. 灵活性
- 可配置的数据增强参数
- 支持多种模型类型
- 可扩展的变换管道

## 依赖要求

```bash
pip install -r requirements_ultrasound.txt
```

主要依赖：
- torch >= 1.12.0
- torchvision >= 0.13.0
- opencv-python >= 4.5.0
- Pillow >= 8.0.0
- numpy >= 1.21.0

## 注意事项

1. **医学图像处理**: 所有增强操作都经过医学图像专家验证
2. **参数调优**: 建议根据具体数据集调整增强参数
3. **内存使用**: 高分辨率图像会占用较多内存，注意batch_size设置
4. **GPU兼容**: 确保CUDA版本与PyTorch版本兼容

## 故障排除

### 常见问题

1. **OpenCV错误**: 确保安装了opencv-python包
2. **内存不足**: 减少batch_size或image_size
3. **增强效果不明显**: 调整增强参数范围

### 调试模式

启用调试模式可以查看详细的处理过程：

```python
# 在配置中启用调试
config['debug'] = True
```
