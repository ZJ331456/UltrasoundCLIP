# 图像编码器类别分析

## 概述

这个脚本用于比较两个图像编码器（QwenVLCLIP 和 SAMBERT）在不同类型图片上的表现和聚合度分析。

## 功能特性

### 1. 数据集加载
- 自动扫描指定路径下的所有JSON文件
- 每个文件最多读取500条数据
- 按文件名自动分类
- 支持递归目录扫描

### 2. 特征提取
- 批量提取图像特征
- 支持GPU加速
- 自动设备检测
- 内存优化处理

### 3. 聚类质量分析
- 轮廓系数计算
- 类别内聚合度分析
- 模型性能对比
- 详细统计报告

### 4. 可视化分析
- PCA降维可视化
- 聚合度热力图
- 特征相似度矩阵
- 类别边界标记

## 使用方法

### 1. 直接运行Python脚本
```bash
cd /media/ps/data-ssd/UltrasoundRAG/CLIP/show_embedding
python compare_encoders_by_category.py
```

### 2. 使用运行脚本（推荐）
```bash
# 给脚本执行权限
chmod +x run_category_analysis.sh

# 运行分析
bash run_category_analysis.sh
```

### 3. 后台运行
```bash
# 后台运行并保存日志
nohup bash run_category_analysis.sh > category_analysis.log 2>&1 &

# 查看日志
tail -f category_analysis.log
```

## 配置说明

### 数据路径配置
在 `compare_encoders_by_category.py` 中修改：

```python
# 基础数据路径
base_path = "/media/ps/data-ssd/UltrasoundRAG/clip_caption/data-anatomy/更新路径之后的json数据集/split_with_llm_enhance_refined"

# 输出目录
output_dir = "/media/ps/data-ssd/UltrasoundRAG/CLIP/show_embedding/output/category_analysis"
```

### 模型配置路径
```python
# Qwen模型配置
qwen_config_path = '/media/ps/data-ssd/UltrasoundRAG/CLIP/config/qwen/qwen_vl_clip_train_optimized.json'

# SAM模型配置
sam_config_path = '/media/ps/data-ssd/UltrasoundRAG/CLIP/config/ultrasam/sam_bert_train.json'
```

### 处理参数
```python
# 每个文件最大样本数
max_samples_per_file = 500

# 批处理大小
batch_size = 8

# 工作进程数
num_workers = 4

# 图像尺寸
qwen_image_size = 588  # Qwen模型
sam_image_size = 1024  # SAM模型
```

## 输出结果

### 1. 可视化图表
- `pca_comparison.png`: PCA降维对比图
- `QwenVLCLIP_clustering_heatmap.png`: Qwen模型聚合度热力图
- `SAMBERT_clustering_heatmap.png`: SAM模型聚合度热力图
- `QwenVLCLIP_similarity_matrix.png`: Qwen模型相似度矩阵
- `SAMBERT_similarity_matrix.png`: SAM模型相似度矩阵

### 2. 分析结果
- `analysis_results.json`: 详细的分析结果和统计数据

### 3. 控制台输出
- 数据集加载信息
- 模型加载状态
- 特征提取进度
- 聚类质量分析
- 模型性能对比

## 分析指标

### 1. 轮廓系数 (Silhouette Score)
- **范围**: -1 到 1
- **含义**: 
  - 接近1: 类别内聚合度高，类别间分离度好
  - 接近0: 类别边界模糊
  - 接近-1: 类别划分错误

### 2. 类别内聚合度
- 计算每个类别内样本的平均距离
- 距离越小，聚合度越高
- 归一化处理便于比较

### 3. 特征相似度矩阵
- 余弦相似度计算
- 可视化类别边界
- 识别异常样本

## 性能优化

### 1. GPU加速
- 自动检测CUDA设备
- 批量特征提取
- 混合精度计算

### 2. 内存管理
- 分批处理大数据集
- 及时释放中间结果
- 优化数据加载

### 3. 并行处理
- 多进程数据加载
- 异步I/O操作
- 流水线处理

## 故障排除

### 1. 常见问题
- **路径错误**: 检查数据路径和模型配置路径
- **内存不足**: 减少batch_size或max_samples_per_file
- **GPU错误**: 检查CUDA版本和GPU内存

### 2. 调试模式
```python
# 在脚本中添加调试信息
print(f"DEBUG: 当前处理文件: {json_file}")
print(f"DEBUG: 特征形状: {features.shape}")
```

### 3. 日志查看
```bash
# 查看运行日志
tail -f category_analysis.log

# 查看错误信息
grep "ERROR\|Exception" category_analysis.log
```

## 扩展功能

### 1. 添加新的编码器
```python
# 在main函数中添加新模型
new_model = NewEncoderModel(config)
new_features, new_categories = extract_features_by_category(
    new_model, image_paths, categories, 
    batch_size=8, num_workers=4, image_size=512, is_sam=False
)
```

### 2. 自定义分析指标
```python
# 添加新的评估函数
def custom_metric(features, categories):
    # 实现自定义指标
    pass
```

### 3. 批量分析
```python
# 支持多个数据集路径
dataset_paths = [
    "path1",
    "path2",
    "path3"
]

for path in dataset_paths:
    analyze_dataset(path)
```

## 注意事项

1. **数据格式**: 确保JSON文件包含正确的字段（data_path, CLIPcaption等）
2. **图像路径**: 验证图像文件路径的有效性
3. **模型配置**: 确保模型配置文件路径正确
4. **GPU资源**: 监控GPU内存使用情况
5. **存储空间**: 确保有足够的磁盘空间保存结果

## 联系支持

如果遇到问题，请检查：
1. 日志文件中的错误信息
2. 数据路径和文件格式
3. 模型配置和依赖包
4. 系统资源和权限设置
