# CLIP模型评估使用指南

## 概述

整合后的CLIP模型评估系统，提供标准检索指标和细粒度分析功能。

## 文件结构

```
CLIP/
├── evaluator.py              # 整合的评估器（支持标准+细粒度评估）
├── eval.py                   # 整合的评估脚本
├── run_eval_pretrain.sh      # 预训练模型评估脚本
├── run_eval_finetuned.sh     # 微调模型评估脚本
└── README_evaluation_final.md # 本使用指南
```

## 核心功能

### 1. 标准检索评估 (`--eval_type standard`)
- **Recall@1/5/10**: 图像↔文本双向检索指标
- **排名统计**: 平均排名和中位数排名
- **相似度分析**: 正确匹配的相似度分布

### 2. 细粒度分析 (`--eval_type finegrained`)
- **解剖部位识别**: 针对超声图像的部位分类准确率
- **病理特征识别**: 病变检测准确率分析
- **Top-K准确率**: Top-3和Top-5准确率
- **可视化报告**: 相似度分布图和混淆矩阵

### 3. 综合评估 (`--eval_type both`)
- 同时执行标准检索和细粒度分析
- 生成完整的评估报告和摘要

## 快速开始

### 方法1: 使用预配置脚本（推荐）

```bash
# 评估预训练模型
./run_eval_pretrain.sh

# 评估微调后模型
./run_eval_finetuned.sh
```

### 方法2: 手动命令行

```bash
# 标准检索评估
python eval.py \
    --config config/qwen/qwen_vl_clip_opt_3.json \
    --dataset test \
    --eval_type standard \
    --output_dir ./eval_results/standard \
    --verbose

# 微调模型评估
python eval.py \
    --config config/qwen/qwen_vl_clip_opt_4.json \
    --model_path output/best_model.pth \
    --dataset test \
    --eval_type both \
    --output_dir ./eval_results/finetuned \
    --max_batches 100 \
    --verbose
```

## 参数说明

| 参数 | 说明 | 默认值 | 选项 |
|------|------|--------|------|
| `--config` | 模型配置文件路径 | 必需 | - |
| `--model_path` | 微调权重路径（可选） | 空 | - |
| `--dataset` | 评估数据集 | `test` | `valid`, `test` |
| `--device` | 计算设备 | `auto` | `auto`, `cuda:0`, `cpu` |
| `--eval_type` | 评估类型 | `both` | `standard`, `finegrained`, `both` |
| `--output_dir` | 结果保存目录 | `./eval_results` | - |
| `--max_batches` | 细粒度评估批次限制 | `100` | - |
| `--verbose` | 显示详细信息 | `False` | - |
| `--output` | 兼容性输出路径（可选） | 空 | - |

## 输出文件

评估完成后生成以下文件：

```
eval_results/
├── standard_evaluation.json          # 标准检索指标
├── finegrained_evaluation.json       # 细粒度分析结果
├── comprehensive_evaluation.json     # 综合评估结果
├── evaluation_summary.md             # 评估摘要报告
├── similarity_distribution.png       # 相似度分布图
└── confusion_matrix.png             # 混淆矩阵图
```

## 指标解释

### 标准检索指标

- **Recall@K**: 正确答案排在前K位的查询比例
  - Recall@1: 第1位命中率
  - Recall@5: 前5位命中率
  - Recall@10: 前10位命中率
- **Mean Rank**: 正确答案的平均排名
- **Median Rank**: 正确答案的中位数排名

### 细粒度指标

- **整体准确率**: 预测完全正确的样本比例
- **Top-K准确率**: 正确答案在前K个预测中的比例
- **部位识别**: 各解剖部位的识别准确率
- **病理识别**: 各病理特征的识别准确率
- **相似度分离度**: 正负样本对相似度的差值

## 性能基准

| 指标 | 优秀 | 良好 | 一般 | 需改进 |
|------|------|------|------|--------|
| Mean Recall@1 | ≥0.80 | 0.60-0.79 | 0.40-0.59 | <0.40 |
| Mean Recall@5 | ≥0.95 | 0.85-0.94 | 0.70-0.84 | <0.70 |
| 相似度分离度 | ≥0.30 | 0.20-0.29 | 0.10-0.19 | <0.10 |

## 脚本配置

### 预训练模型评估 (`run_eval_pretrain.sh`)
```bash
# 主要配置
CONFIG_FILE="config/qwen/qwen_vl_clip_opt_3.json"
DATASET="test"
DEVICE="cuda:0"
OUTPUT_DIR="eval_results/pretrain_model"
EVAL_TYPE="both"
```

### 微调模型评估 (`run_eval_finetuned.sh`)
```bash
# 主要配置
CONFIG_FILE="config/qwen/qwen_vl_clip_opt_4.json"
MODEL_PATH="output/qwen_vl_clip_7_enhanced_contrast/best_model.pth"
DATASET="test"
DEVICE="cuda:0"
OUTPUT_DIR="eval_results/finetuned_model"
EVAL_TYPE="both"
```

## 故障排除

### 常见问题

1. **配置文件不存在**
   ```bash
   # 检查配置文件路径
   ls config/qwen/
   ```

2. **模型权重不存在**
   ```bash
   # 查找可用的模型文件
   find output/ -name "*.pth" -type f
   ```

3. **CUDA内存不足**
   ```bash
   # 减少批次大小
   --max_batches 50
   ```

4. **数据集加载失败**
   ```bash
   # 检查数据集配置
   cat config/qwen/qwen_vl_clip_opt_3.json | grep -A5 "data"
   ```

### 性能优化

1. **GPU利用率低**
   - 增加`--max_batches`参数
   - 检查数据加载器的`num_workers`设置

2. **评估速度慢**
   - 使用`--eval_type standard`跳过细粒度分析
   - 减少`--max_batches`数量

3. **内存占用过高**
   - 设置较小的`--max_batches`值
   - 使用`--eval_type standard`模式

## 结果对比

比较不同模型的性能：

```bash
# 运行两个评估
./run_eval_pretrain.sh
./run_eval_finetuned.sh

# 比较摘要报告
diff eval_results/pretrain_model/evaluation_summary.md \
     eval_results/finetuned_model/evaluation_summary.md

# 比较数值结果
python -c "
import json
with open('eval_results/pretrain_model/standard_evaluation.json') as f:
    pre = json.load(f)
with open('eval_results/finetuned_model/standard_evaluation.json') as f:
    fine = json.load(f)
print(f'微调前 Mean R@1: {pre[\"mean_r1\"]:.4f}')
print(f'微调后 Mean R@1: {fine[\"mean_r1\"]:.4f}')
print(f'提升: {fine[\"mean_r1\"] - pre[\"mean_r1\"]:.4f}')
"
```

## 自定义评估

如需自定义评估设置，可修改脚本中的配置变量：

```bash
# 编辑预训练模型评估脚本
nano run_eval_pretrain.sh

# 编辑微调模型评估脚本  
nano run_eval_finetuned.sh
```

主要可修改的参数：
- `CONFIG_FILE`: 模型配置文件
- `MODEL_PATH`: 模型权重路径
- `DATASET`: 评估数据集
- `DEVICE`: 计算设备
- `EVAL_TYPE`: 评估类型
- `MAX_BATCHES`: 批次限制
- `CUDA_VISIBLE_DEVICES`: GPU设备
