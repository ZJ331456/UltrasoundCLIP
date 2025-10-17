# 平衡批次采样器使用指南

## 概述

平衡批次采样器（BalancedBatchSampler）是为了解决超声图像数据集中类别不平衡问题而设计的。它确保每个训练batch包含多个类别的样本，避免某些大类（如brachial plexus、carotid artery）主导训练过程。

## 主要特性

### 1. BalancedBatchSampler
- **目标**: 确保每个batch包含指定数量的不同类别样本
- **策略**: 每个batch包含8个类别，每个类别8个样本（batch_size=64时）
- **过采样**: 支持对小类进行过采样，避免小类样本不足
- **类别提取**: 自动从文本描述中提取类别信息

### 2. StratifiedBatchSampler
- **目标**: 保持每个batch中各类别比例与整体数据集比例相近
- **策略**: 分层采样，限制每个batch的类别数量
- **适用场景**: 当希望保持原始数据分布时使用

## 配置参数

在配置文件的`data`部分添加`balanced_sampling`配置：

```json
{
  "data": {
    "batch_size": 64,
    "balanced_sampling": {
      "enabled": true,
      "type": "balanced",
      "num_classes_per_batch": 8,
      "samples_per_class": 8,
      "oversample_small_classes": true,
      "max_oversample_ratio": 3.0,
      "shuffle": true,
      "drop_last": true,
      "class_key": "class",
      "text_key": "text"
    }
  }
}
```

### 参数说明

- `enabled`: 是否启用平衡采样（true/false）
- `type`: 采样器类型（"balanced" 或 "stratified"）
- `num_classes_per_batch`: 每个batch包含的类别数量
- `samples_per_class`: 每个类别的样本数量（如果为null，自动计算为batch_size/num_classes_per_batch）
- `oversample_small_classes`: 是否对小类进行过采样
- `max_oversample_ratio`: 最大过采样比例
- `shuffle`: 是否打乱数据
- `drop_last`: 是否丢弃最后一个不完整的batch
- `class_key`: 类别信息的键名（如果数据中有明确的类别字段）
- `text_key`: 文本信息的键名（用于从文本中提取类别）

## 类别提取逻辑

采样器会自动从文本描述中提取类别信息，支持的类别包括：

- brachial_plexus (臂丛神经)
- carotid_artery (颈动脉)
- thyroid (甲状腺)
- liver (肝脏)
- kidney (肾脏)
- heart (心脏)
- lung (肺部)
- breast (乳腺)
- ovary (卵巢)
- uterus (子宫)
- prostate (前列腺)
- bladder (膀胱)
- pancreas (胰腺)
- spleen (脾脏)
- gallbladder (胆囊)
- aorta (主动脉)
- vena_cava (下腔静脉)
- portal_vein (门静脉)
- bile_duct (胆管)
- lymph_node (淋巴结)

## 使用示例

### 1. 基本使用

```python
from dataload.factory import DataLoaderFactory

# 创建数据集
datasets = DataLoaderFactory.create_datasets(config)

# 创建数据加载器（会自动使用平衡采样器）
dataloaders = DataLoaderFactory.create_dataloaders(datasets, config)
```

### 2. 自定义配置

```python
# 在配置中自定义平衡采样参数
config = {
    "data": {
        "batch_size": 64,
        "balanced_sampling": {
            "enabled": true,
            "type": "balanced",
            "num_classes_per_batch": 10,  # 每个batch包含10个类别
            "samples_per_class": 6,       # 每个类别6个样本
            "oversample_small_classes": true,
            "max_oversample_ratio": 2.0   # 最大过采样2倍
        }
    }
}
```

### 3. 获取统计信息

```python
# 在训练器中会自动打印平衡采样的统计信息
trainer = Trainer(model, train_loader, val_loader, config, device)

# 输出示例：
# - 平衡采样器: 启用 (BalancedBatchSampler)
#   - 类别数量: 15
#   - 每batch类别数: 8
#   - 每类别样本数: 8
#   - 类别不平衡比例: 54.58
#   - 总batch数: 1250
```

## 性能优化建议

### 1. 批次大小设置
- 建议batch_size能被num_classes_per_batch整除
- 例如：batch_size=64, num_classes_per_batch=8, samples_per_class=8

### 2. 类别数量控制
- 如果类别太多，可以适当增加num_classes_per_batch
- 如果类别太少，可以减少num_classes_per_batch

### 3. 过采样策略
- 对于极度不平衡的数据，启用oversample_small_classes
- 设置合理的max_oversample_ratio避免过度过采样

### 4. 内存使用
- 过采样会增加内存使用，注意监控GPU内存
- 可以适当减少batch_size或max_oversample_ratio

## 故障排除

### 1. 类别提取失败
如果采样器无法正确提取类别，检查：
- 文本描述是否包含类别关键词
- class_key和text_key是否正确
- 数据格式是否符合预期

### 2. 内存不足
如果出现内存不足，尝试：
- 减少batch_size
- 降低max_oversample_ratio
- 减少num_classes_per_batch

### 3. 训练速度慢
如果训练速度变慢，检查：
- 是否启用了过采样（会增加数据量）
- 数据加载器的num_workers设置
- 是否使用了persistent_workers

## 实验建议

### 1. 对比实验
建议进行以下对比实验：
- 不使用平衡采样 vs 使用平衡采样
- BalancedBatchSampler vs StratifiedBatchSampler
- 不同的num_classes_per_batch设置

### 2. 监控指标
重点关注以下指标：
- 各类别的召回率（R@1, R@5, R@10）
- 整体检索性能
- 训练稳定性

### 3. 超参数调优
可以调整的参数：
- num_classes_per_batch: [4, 6, 8, 10, 12]
- samples_per_class: [4, 6, 8, 10]
- max_oversample_ratio: [1.5, 2.0, 3.0, 5.0]

## 注意事项

1. **数据一致性**: 平衡采样器会改变数据的采样顺序，确保验证集不使用平衡采样
2. **类别定义**: 确保类别提取逻辑与你的数据标注一致
3. **性能监控**: 定期检查各类别的性能表现，确保平衡采样确实改善了小类的性能
4. **内存管理**: 过采样会增加内存使用，注意监控系统资源

## 更新日志

- v1.0: 初始版本，支持基本的平衡批次采样
- v1.1: 添加分层采样器支持
- v1.2: 优化类别提取逻辑，支持更多医学类别
- v1.3: 添加统计信息输出和性能监控
