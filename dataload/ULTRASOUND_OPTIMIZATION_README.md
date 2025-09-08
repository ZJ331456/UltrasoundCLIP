# 超声图像优化数据集使用指南

## 概述

本模块将CLIP_mini中的超声图像优化特性整合到了完整的CLIP训练系统中，提供了专门针对超声图像优化的数据集类和相关功能。

## 主要特性

### 1. 等比例缩放+填充
- 避免图像变形，保持医学结构的完整性
- 使用中性灰填充，对超声图像友好
- 支持自定义目标尺寸和填充颜色

### 2. 超声图像特定的数据增强
- **轻微旋转**：模拟探头角度变化（最大±5°）
- **轻微模糊**：模拟成像不稳
- **亮度/对比度微调**：幅度很小，避免过度增强
- **随机Gamma校正**：模拟探头/增益变化
- **斑点噪声模拟**：模拟超声常见噪声

### 3. 智能预处理
- 自动转换为RGB格式
- 保持图像质量，避免过度处理
- 支持训练和验证时的不同处理策略

## 使用方法

### 1. 配置文件设置

在配置文件中启用超声图像优化：

```yaml
data:
  # 启用超声图像优化
  ultrasound_optimization:
    enabled: true  # 关键：必须设置为true
    
    # 超声图像特定配置
    target_image_size: 224
    preserve_aspect_ratio: true
    fill_color: [128, 128, 128]
    
    # 数据增强配置
    augmentation:
      enabled: true
      rotation_range: 5
      brightness_range: 0.1
      contrast_range: 0.1
      gamma_correction: true
      speckle_noise: true
```

### 2. 代码使用

```python
from dataload import DataLoaderFactory

# 创建数据集（会自动检测并启用超声图像优化）
datasets = DataLoaderFactory.create_datasets(
    config_dict,
    max_samples=None,
    validate_files=True,
    cache_size=500
)

# 创建数据加载器
dataloaders = DataLoaderFactory.create_dataloaders(datasets, config_dict)
```

### 3. 直接使用超声数据集

```python
from dataload import UltrasoundDataset

# 直接创建超声数据集
ultrasound_dataset = UltrasoundDataset(
    json_path="path/to/data.json",
    target_image_size=224,
    augment=True,  # 训练时启用增强
    ultrasound_specific=True
)
```

## 核心组件

### 1. UltrasoundDataset类
- 继承自BaseDataset
- 自动处理超声图像特定的预处理
- 支持多种JSON格式

### 2. ResizeWithPadding类
- 等比例缩放+填充变换
- 保持图像比例，避免变形
- 可自定义填充颜色

### 3. RandomGamma类
- 随机Gamma校正
- 模拟探头/增益变化
- 在PIL阶段执行，避免精度丢失

### 4. SpeckleNoise类
- 斑点噪声模拟
- 乘性噪声，依赖局部强度
- 在张量阶段执行

## 配置选项详解

### ultrasound_optimization配置

```yaml
ultrasound_optimization:
  enabled: true                    # 是否启用超声图像优化
  
  # 图像尺寸配置
  target_image_size: 224          # 目标图像尺寸
  preserve_aspect_ratio: true     # 是否保持宽高比
  
  # 填充配置
  fill_color: [128, 128, 128]    # 填充颜色（RGB）
  
  # 数据增强配置
  augmentation:
    enabled: true                  # 是否启用数据增强
    rotation_range: 5             # 最大旋转角度
    brightness_range: 0.1         # 亮度调整范围
    contrast_range: 0.1           # 对比度调整范围
    gamma_correction: true        # 是否启用gamma校正
    speckle_noise: true           # 是否启用斑点噪声
```

## 与现有系统的兼容性

### 1. 自动检测
- 系统会自动检测配置中的`ultrasound_optimization.enabled`设置
- 如果启用，自动使用UltrasoundDataset
- 如果未启用，使用原有的通用数据集

### 2. 接口一致性
- UltrasoundDataset返回的数据格式与现有系统完全兼容
- 支持现有的数据加载器和训练流程
- 无需修改训练代码

### 3. 渐进式启用
- 可以只在特定数据集上启用超声图像优化
- 支持混合使用（部分数据集启用，部分不启用）

## 性能优化

### 1. 图像缓存
- 支持LRU缓存，减少重复加载
- 可配置缓存大小
- 自动内存管理

### 2. 批处理优化
- 支持批处理数据增强
- 优化的内存使用
- 支持多进程数据加载

### 3. 验证优化
- 验证时自动禁用数据增强
- 减少不必要的计算
- 提高验证速度

## 故障排除

### 1. 常见问题

**Q: 启用超声图像优化后训练变慢？**
A: 检查是否启用了过多的数据增强，可以适当减少增强强度。

**Q: 图像质量下降？**
A: 检查填充颜色设置，建议使用中性灰[128, 128, 128]。

**Q: 内存使用过高？**
A: 减少缓存大小，或使用更小的batch_size。

### 2. 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查数据集状态
dataset = datasets['train']
print(f"数据集统计: {dataset.get_stats()}")
print(f"样本数量: {len(dataset)}")
```

## 扩展开发

### 1. 添加新的增强方法

```python
class CustomUltrasoundAugmentation:
    def __init__(self, **kwargs):
        # 初始化参数
        
    def __call__(self, img):
        # 实现增强逻辑
        return enhanced_img
```

### 2. 自定义预处理流程

```python
class CustomUltrasoundDataset(UltrasoundDataset):
    def _create_ultrasound_transforms(self):
        # 自定义变换流程
        transforms_list = []
        # 添加自定义变换...
        return transforms.Compose(transforms_list)
```

## 总结

超声图像优化模块成功将CLIP_mini中的优秀特性整合到了完整的CLIP训练系统中，提供了：

1. **专业的超声图像预处理**：等比例缩放、智能填充、领域特定的数据增强
2. **无缝的系统集成**：与现有训练流程完全兼容，无需修改代码
3. **灵活的配置选项**：支持多种配置组合，满足不同需求
4. **优秀的性能表现**：优化的内存使用和计算效率

通过简单的配置文件修改，就可以享受到超声图像特定的优化，提升训练效果和模型性能。
