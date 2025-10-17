# 模块化CLIP训练框架

这是一个专为超声影像设计的模块化CLIP（Contrastive Language-Image Pre-training）训练框架，将复杂的多模态学习任务分解为多个独立、可维护的模块。框架支持多种视觉编码器（ConvNeXt、Qwen-VL、FetalCLIP、SAM-BERT等）和丰富的训练策略（LoRA微调、EMA、多种损失函数等）。

---

## 目录结构

```
UltrasoundRAG/CLIP/
├── 📁 核心模块
│   ├── train.py                    # 训练入口脚本
│   ├── eval.py                     # 评估入口脚本
│   ├── trainer.py                  # 训练器核心逻辑
│   ├── evaluator.py                # 评估器核心逻辑
│   ├── loss.py                     # 损失函数集合
│   └── config_manager.py           # 配置管理器
│
├── 📁 models/                      # 模型模块
│   ├── __init__.py
│   ├── model_factory.py            # 模型工厂（统一创建接口）
│   ├── convnext_model.py           # ConvNeXt-CLIP模型
│   ├── qwen_vl_clip_model.py       # Qwen-VL-CLIP模型
│   ├── fetalclip_pre_model.py      # FetalCLIP预训练模型
│   ├── fetalclip_post_model.py     # FetalCLIP微调模型
│   ├── ultrasam_bert_pre_model.py  # UltraSAM-BERT预训练
│   ├── ultrasam_bert_post_model.py # UltraSAM-BERT微调
│   ├── medclip_model.py            # MedCLIP模型
│   ├── ConvNeXt/                   # ConvNeXt模型实现
│   ├── MedCLIP/                    # MedCLIP模型实现
│   ├── UniMed-CLIP/                # UniMed-CLIP模型
│   ├── open_clip/                  # OpenCLIP基础库
│   └── qwen-2.5-vl/                # Qwen-2.5-VL模型库
│
├── 📁 dataload/                    # 数据加载模块
│   ├── __init__.py
│   ├── factory.py                  # 数据加载器工厂
│   ├── base_dataset.py             # 数据集基类
│   ├── json_dataset.py             # JSON格式数据集
│   ├── jsonl_dataset.py            # JSONL格式数据集
│   ├── ultrasound_dataset.py       # 超声图像专用数据集
│   ├── collator.py                 # 数据批处理整理器
│   ├── transforms.py               # 图像变换和数据增强
│   ├── balanced_sampler.py         # 平衡采样器
│   ├── example.py                  # 使用示例
│   ├── README.md                   # 数据加载模块说明
│   ├── BALANCED_SAMPLING_README.md # 平衡采样说明
│   └── ULTRASOUND_OPTIMIZATION_README.md  # 超声数据优化说明
│
├── 📁 config/                      # 配置文件目录
│   ├── ultra_clip_config.json      # 通用CLIP配置模板
│   ├── convnext/                   # ConvNeXt模型配置
│   ├── qwen/                       # Qwen-VL模型配置
│   ├── fetal/                      # FetalCLIP模型配置
│   └── ultrasam/                   # UltraSAM-BERT模型配置
│
├── 📁 scripts/                     # 训练脚本目录
│   ├── run_train.sh                # 通用训练脚本
│   ├── run_convnext_train.sh       # ConvNeXt训练脚本
│   ├── run_convnext_train_lora.sh  # ConvNeXt LoRA训练
│   ├── run_convnext_coco_train.sh  # ConvNeXt COCO数据集训练
│   ├── run_qwen_train.sh           # Qwen-VL训练脚本
│   ├── run_qwen_train_optimized.sh # Qwen-VL优化版本
│   ├── run_eval_pretrain.sh        # 预训练模型评估
│   ├── run_eval_finetuned.sh       # 微调模型评估
│   ├── train_experiments.sh        # 批量实验脚本
│   ├── evaluate_experiments.py     # 实验评估脚本
│   └── 说明.md                     # 脚本使用说明
│
├── 📁 tools/                       # 工具脚本
│   ├── plot_training_curves.py     # 绘制训练曲线
│   ├── validate_configs.py         # 验证配置文件
│   ├── split_json_by_ratio.py      # 数据集分割工具
│   ├── convert_coco_captions.py    # COCO数据格式转换
│   ├── extract_medical_terms.py    # 提取医学术语
│   └── api_config.py               # API配置管理
│
├── 📁 analysize/                   # 分析模块
│   ├── __init__.py
│   ├── model_analyzer.py           # 模型分析器
│   ├── feature_analysis.py         # 特征分析
│   ├── error_analysis.py           # 错误分析
│   ├── ablation_study.py           # 消融实验
│   ├── evaluation_utils.py         # 评估工具函数
│   ├── visualization.py            # 可视化工具
│   ├── run_analysis.py             # 分析脚本入口
│   └── run_analysis.sh             # 分析执行脚本
│
├── 📁 utils/                       # 工具函数
│   ├── __init__.py
│   ├── model_cache.py              # 模型缓存管理
│   └── performance.py              # 性能监控工具
│
├── 📁 example_configs/             # 示例配置
│   ├── baseline_config.json        # 基线配置
│   ├── lora_optimized_config.json  # LoRA优化配置
│   ├── medical_enhanced_config.json # 医学增强配置
│   ├── simplified_loss.py          # 简化的损失函数示例
│   ├── 快速开始指南.md             # 快速入门
│   └── 优化指南.md                 # 优化建议
│
├── 📁 输出目录
│   ├── checkpoints/                # 模型检查点保存目录
│   ├── logs/                       # 训练日志目录
│   ├── output/                     # 输出结果目录
│   ├── eval_results/               # 评估结果目录
│   └── png/                        # 可视化图片目录
│
├── 📁 外部依赖
│   ├── cn_clip/                    # CN-CLIP参考实现
│   ├── convnext/                   # ConvNeXt原始实现
│   └── related_clip_code/          # 相关CLIP代码
│
├── 📁 实验文档
│   ├── 实验实施总结.md
│   ├── 第二轮实验运行指南.md
│   ├── 第三轮实验运行指南.md
│   ├── 第三轮实验结果总结.md
│   ├── 对照组实验使用指南.md
│   ├── 新的实验方案..md
│   └── 防止过拟合优化.md
│
├── requirements.txt                 # Python依赖包
├── README.md                        # 本文件
└── monitor_training.py              # 训练监控工具
```

---

## 核心模块详解

### 1. 训练入口（train.py）

**功能**：统一的训练入口脚本，负责整个训练流程的协调。

**主要特性**：
- ✅ 自动查找最新checkpoint进行断点续训
- ✅ 支持命令行参数和配置文件混合配置
- ✅ 设备自动选择（CPU/GPU/Multi-GPU）
- ✅ 内存和显存管理优化
- ✅ 异常处理和错误恢复

**使用示例**：
```bash
# 基本训练
python train.py --config config/convnext/my_config.json

# 断点续训
python train.py --config config/convnext/my_config.json --resume

# 指定checkpoint路径
python train.py --config config/convnext/my_config.json --checkpoint path/to/checkpoint.pth

# 指定GPU
CUDA_VISIBLE_DEVICES=0,1 python train.py --config config/convnext/my_config.json
```

**关键函数**：
- `find_latest_checkpoint()`: 自动查找最新的checkpoint
- `main()`: 主训练流程

---

### 2. 评估入口（eval.py）

**功能**：整合的CLIP模型评估脚本，支持多种评估模式。

**评估类型**：
1. **标准检索评估**：图像到文本、文本到图像的检索性能
2. **细粒度分析**：混淆矩阵、类别准确率、错误分析
3. **综合评估**：同时执行上述两种评估

**支持的指标**：
- Recall@K (R@1, R@5, R@10)
- Mean Reciprocal Rank (MRR)
- Median Rank
- Mean Rank
- 类别级准确率

**使用示例**：
```bash
# 标准检索评估
python eval.py --config config.json --model_path checkpoints/best_model.pth --eval_type standard

# 细粒度分析
python eval.py --config config.json --model_path checkpoints/best_model.pth --eval_type finegrained

# 综合评估
python eval.py --config config.json --model_path checkpoints/best_model.pth --eval_type both

# 指定输出目录
python eval.py --config config.json --model_path checkpoints/best_model.pth --output_dir ./my_results
```

---

### 3. 训练器（trainer.py）

**功能**：核心训练逻辑实现，包含完整的训练和验证流程。

**核心功能**：
1. **优化器管理**：
   - AdamW优化器
   - 学习率预热（Warmup）
   - 余弦退火调度
   - 梯度裁剪

2. **训练策略**：
   - 混合精度训练（AMP）
   - 梯度累积
   - 梯度检查点（Gradient Checkpointing）
   - EMA（指数移动平均）
   - LoRA微调支持

3. **模型管理**：
   - 定期保存checkpoint
   - 最佳模型保存
   - 断点续训
   - 紧急保存机制

4. **监控和日志**：
   - 训练指标记录
   - 验证指标监控
   - TensorBoard集成（可选）
   - 实时进度显示

**关键方法**：
- `train_epoch()`: 单个epoch的训练
- `validate()`: 验证评估
- `_save_checkpoint()`: 保存检查点
- `_load_checkpoint()`: 加载检查点
- `_setup_lora()`: 配置LoRA微调

---

### 4. 评估器（evaluator.py）

**功能**：全面的模型评估器，支持标准检索和细粒度分析。

**核心能力**：
1. **特征提取**：
   - 批量图像特征提取
   - 批量文本特征提取
   - 高效内存管理
   - GPU加速计算

2. **检索评估**：
   - 图像到文本检索（I2T）
   - 文本到图像检索（T2I）
   - Top-K检索准确率
   - 排名指标计算

3. **细粒度分析**：
   - 混淆矩阵生成
   - 类别级准确率
   - 错误案例分析
   - 可视化报告

4. **性能优化**：
   - 参考CN-CLIP的批处理优化
   - 内存高效的特征存储
   - 向量化计算加速

**主要方法**：
- `evaluate()`: 标准检索评估
- `fine_grained_evaluation()`: 细粒度评估
- `_extract_all_features()`: 提取所有特征
- `_compute_retrieval_metrics()`: 计算检索指标

---

### 5. 损失函数（loss.py）

**功能**：提供多种CLIP训练损失函数的实现。

**支持的损失函数**：

1. **CLIPLoss**：标准CLIP对比损失
   - 双向对比学习（图像→文本，文本→图像）
   - 可配置温度参数
   - 标签平滑支持

2. **HardNegativeLoss**：硬负样本损失
   - 自动挖掘困难样本
   - Top-K困难样本选择
   - 提升模型判别能力

3. **FocalCLIPLoss**：焦点损失
   - 关注难分类样本
   - 缓解类别不平衡
   - 可调整焦点参数

4. **InfoNCELoss**：信息噪声对比估计损失
   - 理论基础扎实
   - 适用于大批量训练
   - 高效计算

5. **SigmoidLoss**：Sigmoid损失
   - 二元交叉熵变体
   - 适用于特定场景
   - 数值稳定

6. **MultiSimilarityLoss**：多相似度损失
   - 多粒度相似度建模
   - 复杂关系学习
   - 高级对比学习

7. **HybridLoss**：混合损失
   - 组合多个损失函数
   - 可配置权重
   - 灵活的训练策略

**使用方式**：
```python
from loss import LossFactory

# 创建标准CLIP损失
loss_fn = LossFactory.create_loss({
    'type': 'clip',
    'temperature': 0.07,
    'label_smoothing': 0.1
})

# 创建硬负样本损失
loss_fn = LossFactory.create_loss({
    'type': 'hard_negative',
    'temperature': 0.07,
    'num_hard_negatives': 5
})

# 创建混合损失
loss_fn = LossFactory.create_loss({
    'type': 'hybrid',
    'losses': [
        {'type': 'clip', 'weight': 0.7},
        {'type': 'hard_negative', 'weight': 0.3}
    ]
})
```

---

### 6. 配置管理器（config_manager.py）

**功能**：统一管理训练配置，支持配置文件加载和默认值设置。

**配置结构**：
```json
{
  "model": {
    "type": "convnext_clip",
    "config_path": "...",
    "pretrained_path": "...",
    "embed_dim": 768,
    "temperature": 0.07
  },
  "data": {
    "train_json": "...",
    "valid_json": "...",
    "image_size": 224,
    "batch_size": 32,
    "num_workers": 4,
    "max_text_length": 256
  },
  "training": {
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "max_epochs": 100,
    "gradient_clip_val": 1.0,
    "save_interval": 5
  },
  "loss": {
    "type": "clip",
    "temperature": 0.07,
    "label_smoothing": 0.1
  }
}
```

**主要方法**：
- `load_config()`: 加载配置文件
- `validate_config()`: 验证配置有效性
- `get()`: 获取配置项
- `update()`: 更新配置

---

## 模型模块（models/）

### 模型工厂（model_factory.py）

**功能**：提供统一的模型创建接口，根据配置自动实例化不同类型的模型。

**支持的模型类型**：

1. **ConvNeXt-CLIP**（`convnext_clip`）
   - 基于ConvNeXt的视觉编码器
   - BERT文本编码器
   - 高效的特征提取
   - 适用于超声图像

2. **Qwen-VL-CLIP**（`qwen_vl_clip`）
   - 基于Qwen-VL的多模态模型
   - 强大的视觉理解能力
   - 支持LoRA微调
   - 中文文本支持

3. **FetalCLIP**（`fetalclip`）
   - 专为胎儿超声设计
   - 预训练模型和微调模型
   - 领域特定优化

4. **UltraSAM-BERT**（`sam_bert`）
   - 结合SAM的超声图像编码器
   - BERT文本编码器
   - 适用于超声影像分析

5. **MedCLIP**（`medclip`）
   - 医学图像CLIP模型
   - 医学术语理解
   - 临床场景优化

**使用示例**：
```python
from models.model_factory import ModelFactory

# 创建ConvNeXt-CLIP模型
config = {
    'type': 'convnext_clip',
    'vision_model_name': 'convnext_base',
    'text_model_name': 'bert-base-chinese',
    'embed_dim': 768
}
model = ModelFactory.create_model(config)

# 创建Qwen-VL-CLIP模型
config = {
    'type': 'qwen_vl_clip',
    'model_name': 'Qwen2-VL-2B',
    'use_lora': True,
    'lora_config': {...}
}
model = ModelFactory.create_model(config)
```

### 各模型实现文件

- **convnext_model.py**：ConvNeXt视觉编码器实现
- **qwen_vl_clip_model.py**：Qwen-VL多模态模型适配
- **fetalclip_pre_model.py**：FetalCLIP预训练模型
- **fetalclip_post_model.py**：FetalCLIP微调模型
- **ultrasam_bert_pre_model.py**：UltraSAM-BERT预训练
- **ultrasam_bert_post_model.py**：UltraSAM-BERT微调
- **medclip_model.py**：MedCLIP模型实现

---

## 数据加载模块（dataload/）

### 数据加载器工厂（factory.py）

**功能**：根据配置创建数据加载器，支持多种数据格式和采样策略。

**支持的数据格式**：
- JSON格式（`json_dataset.py`）
- JSONL格式（`jsonl_dataset.py`）
- 超声图像专用格式（`ultrasound_dataset.py`）

**采样策略**：
- 随机采样
- 平衡采样（`balanced_sampler.py`）
- 类别权重采样

### 数据集基类（base_dataset.py）

**功能**：定义数据集的通用接口和基础功能。

**核心方法**：
- `__getitem__()`: 获取单个数据样本
- `__len__()`: 返回数据集大小
- `load_data()`: 加载数据
- `preprocess()`: 数据预处理

### 数据增强（transforms.py）

**功能**：提供多种图像变换和数据增强方法。

**支持的变换**：
1. **基础变换**：
   - 随机裁剪
   - 随机翻转
   - 随机旋转
   - 颜色抖动

2. **高级增强**：
   - MixUp
   - CutMix
   - RandAugment
   - AutoAugment

3. **超声图像专用**：
   - 超声噪声模拟
   - 伪影增强
   - 对比度调整

### 批处理整理器（collator.py）

**功能**：将数据样本整理成批次，处理变长数据。

**主要功能**：
- 图像批处理和堆叠
- 文本序列填充
- Mask生成
- 数据类型转换

---

## 配置文件（config/）

### 配置文件结构

每个模型有独立的配置目录，包含：
- 基础配置
- 优化配置
- 实验配置

### 主要配置目录

1. **convnext/**：ConvNeXt模型配置
   - 不同规模的模型（tiny, small, base, large）
   - LoRA微调配置
   - 数据增强配置

2. **qwen/**：Qwen-VL模型配置
   - Qwen2-VL-2B/7B配置
   - 量化配置
   - LoRA参数设置

3. **fetal/**：FetalCLIP配置
   - 预训练配置
   - 微调配置
   - 领域适应设置

4. **ultrasam/**：UltraSAM-BERT配置
   - SAM参数配置
   - BERT配置
   - 融合策略

---

## 脚本目录（scripts/）

### 训练脚本

1. **run_train.sh**：通用训练脚本模板
2. **run_convnext_train.sh**：ConvNeXt全量训练
3. **run_convnext_train_lora.sh**：ConvNeXt LoRA微调
4. **run_qwen_train.sh**：Qwen-VL训练
5. **run_qwen_train_optimized.sh**：Qwen-VL优化版本

### 评估脚本

1. **run_eval_pretrain.sh**：评估预训练模型
2. **run_eval_finetuned.sh**：评估微调模型

### 实验管理

1. **train_experiments.sh**：批量运行多个实验
2. **evaluate_experiments.py**：批量评估实验结果

**使用示例**：
```bash
# 运行单个实验
bash scripts/run_convnext_train.sh

# 运行批量实验
bash scripts/train_experiments.sh

# 评估所有实验
python scripts/evaluate_experiments.py --exp_dir experiments/
```

---

## 工具脚本（tools/）

### 1. plot_training_curves.py

**功能**：从训练日志绘制训练曲线。

**支持的曲线**：
- 损失曲线
- 学习率曲线
- 验证指标曲线
- R@K曲线

**使用示例**：
```bash
python tools/plot_training_curves.py --log_file logs/training.log --output_dir png/
```

### 2. validate_configs.py

**功能**：验证配置文件的正确性和完整性。

**检查项**：
- 必需字段检查
- 路径有效性检查
- 参数范围检查
- 兼容性检查

**使用示例**：
```bash
python tools/validate_configs.py --config config/my_config.json
```

### 3. split_json_by_ratio.py

**功能**：按比例分割数据集为训练集、验证集和测试集。

**使用示例**：
```bash
python tools/split_json_by_ratio.py \
    --input data.json \
    --train_ratio 0.7 \
    --valid_ratio 0.15 \
    --test_ratio 0.15 \
    --output_dir data/splits/
```

### 4. convert_coco_captions.py

**功能**：将COCO格式的数据转换为本框架支持的格式。

**使用示例**：
```bash
python tools/convert_coco_captions.py \
    --coco_json coco_captions.json \
    --output ultrasound_data.json
```

### 5. extract_medical_terms.py

**功能**：从文本中提取医学术语和关键词。

**使用示例**：
```bash
python tools/extract_medical_terms.py \
    --input data.json \
    --output medical_terms.txt
```

---

## 分析模块（analysize/）

### 1. model_analyzer.py

**功能**：全面的模型分析工具。

**分析内容**：
- 模型参数统计
- 计算复杂度（FLOPs）
- 推理速度测试
- 内存占用分析

### 2. feature_analysis.py

**功能**：特征空间分析和可视化。

**分析方法**：
- t-SNE降维可视化
- PCA主成分分析
- 特征分布统计
- 类别可分性分析

### 3. error_analysis.py

**功能**：错误案例分析。

**分析内容**：
- 识别常见错误模式
- 错误案例可视化
- 混淆类别分析
- 改进建议生成

### 4. ablation_study.py

**功能**：消融实验工具。

**实验类型**：
- 模块消融
- 损失函数消融
- 数据增强消融
- 超参数敏感性分析

### 5. visualization.py

**功能**：各种可视化工具。

**可视化类型**：
- 特征可视化
- 注意力图
- 检索结果展示
- 混淆矩阵热图

### 使用示例

```bash
# 运行完整分析
python analysize/run_analysis.py \
    --config config.json \
    --model_path checkpoints/best_model.pth \
    --output_dir analysis_results/

# 或使用shell脚本
bash analysize/run_analysis.sh
```

---

## 工具函数（utils/）

### 1. model_cache.py

**功能**：模型缓存管理，避免重复下载。

**功能**：
- 自动缓存下载的模型
- 检查缓存有效性
- 清理过期缓存

### 2. performance.py

**功能**：性能监控和分析。

**监控内容**：
- GPU显存使用
- CPU和内存使用
- 训练速度统计
- 吞吐量计算

---

## 示例配置（example_configs/）

### 1. baseline_config.json

**说明**：基线配置，使用标准参数和简单策略。

**适用场景**：快速验证、初步实验

### 2. lora_optimized_config.json

**说明**：LoRA微调优化配置，参数高效。

**适用场景**：显存受限、快速迭代

### 3. medical_enhanced_config.json

**说明**：医学增强配置，针对医学图像优化。

**适用场景**：医学影像任务、专业领域

### 4. 快速开始指南.md

**内容**：新手入门指南，5分钟上手。

### 5. 优化指南.md

**内容**：性能优化建议和最佳实践。

---

## 依赖管理（requirements.txt）

**核心依赖**：
- PyTorch >= 1.12.0
- Transformers >= 4.20.0
- Pillow >= 8.0.0
- NumPy >= 1.21.0

**可选依赖**：
- peft：LoRA微调
- albumentations：数据增强
- matplotlib：可视化
- seaborn：统计图表

**安装方法**：
```bash
pip install -r requirements.txt
```

---

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
cd UltrasoundRAG/CLIP

# 安装依赖
pip install -r requirements.txt

# 准备数据
# 将数据整理成指定格式（见dataload/README.md）
```

### 2. 配置文件

复制示例配置并修改：
```bash
cp example_configs/baseline_config.json config/my_config.json
# 编辑my_config.json，设置数据路径和模型参数
```

### 3. 开始训练

```bash
# 使用Python脚本
python train.py --config config/my_config.json

# 或使用Shell脚本
bash scripts/run_train.sh
```

### 4. 模型评估

```bash
python eval.py \
    --config config/my_config.json \
    --model_path checkpoints/best_model.pth \
    --eval_type both
```

### 5. 结果分析

```bash
# 绘制训练曲线
python tools/plot_training_curves.py --log_file logs/training.log

# 运行完整分析
python analysize/run_analysis.py --config config/my_config.json --model_path checkpoints/best_model.pth
```

---

## 高级功能

### 1. LoRA微调

在配置文件中启用LoRA：
```json
{
  "model": {
    "type": "convnext_clip",
    "use_lora": true,
    "lora_config": {
      "r": 8,
      "lora_alpha": 16,
      "lora_dropout": 0.1,
      "target_modules": ["q_proj", "v_proj"]
    }
  }
}
```

### 2. 混合精度训练

```json
{
  "training": {
    "mixed_precision": true,
    "gradient_checkpointing": true
  }
}
```

### 3. 多GPU训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config config.json
```

### 4. 断点续训

```bash
# 自动查找最新checkpoint
python train.py --config config.json --resume

# 指定checkpoint路径
python train.py --config config.json --checkpoint path/to/checkpoint.pth
```

### 5. EMA（指数移动平均）

```json
{
  "training": {
    "use_ema": true,
    "ema_decay": 0.9999
  }
}
```

---

## 实验管理

### 实验配置

所有实验配置保存在`config/`目录下，建议命名规范：
```
config/
  ├── convnext/
  │   ├── exp_baseline.json
  │   ├── exp_lora_r8.json
  │   ├── exp_lora_r16.json
  │   └── exp_hard_negative.json
```

### 实验记录

- 训练日志：`logs/training_*.log`
- Checkpoint：`checkpoints/<exp_name>/`
- 评估结果：`eval_results/<exp_name>/`
- 可视化：`png/<exp_name>/`

### 批量实验

使用`scripts/train_experiments.sh`运行多个实验：
```bash
#!/bin/bash
configs=(
    "config/exp1.json"
    "config/exp2.json"
    "config/exp3.json"
)

for config in "${configs[@]}"; do
    echo "Running experiment: $config"
    python train.py --config "$config"
done
```

---

## 监控和调试

### 1. 训练监控

使用`monitor_training.py`实时监控训练：
```bash
python monitor_training.py --log_file logs/training.log
```

### 2. 日志分析

日志文件包含详细的训练信息：
- Epoch统计
- 批次损失
- 学习率变化
- 验证指标
- 内存使用

### 3. 可视化

绘制训练曲线：
```bash
python tools/plot_training_curves.py \
    --log_file logs/training.log \
    --output_dir png/training_curves/
```

---

## 最佳实践

### 1. 数据准备

✅ **建议**：
- 数据格式统一使用JSON/JSONL
- 图像预处理（去除黑边、归一化）
- 文本清洗（去除特殊字符、统一术语）
- 数据增强（针对超声图像特点）

❌ **避免**：
- 图像尺寸差异过大
- 文本长度极端不均
- 类别分布严重不平衡

### 2. 模型选择

- **显存充足**：使用大模型（ConvNeXt-Large, Qwen-VL-7B）
- **显存受限**：使用LoRA微调或小模型（ConvNeXt-Tiny, Qwen-VL-2B）
- **中文任务**：优先选择Qwen-VL或BERT-Chinese
- **医学领域**：考虑MedCLIP或FetalCLIP

### 3. 超参数调优

**优先级**：
1. 学习率（最关键）
2. Batch Size（影响稳定性）
3. 温度参数（影响对比学习）
4. 损失函数类型

**建议范围**：
- 学习率：1e-6 ~ 5e-5
- Batch Size：16 ~ 128
- 温度：0.01 ~ 0.2
- Warmup Steps：500 ~ 2000

### 4. 训练技巧

1. **从小到大**：先用小数据集验证，再全量训练
2. **逐步调优**：一次只改变一个参数
3. **定期评估**：每个epoch或固定步数评估一次
4. **早停机制**：验证损失不再下降时停止训练
5. **多次实验**：使用不同随机种子验证稳定性

### 5. 性能优化

1. **数据加载优化**：
   - 增加`num_workers`
   - 使用`pin_memory=True`
   - 预加载到内存（小数据集）

2. **训练加速**：
   - 混合精度训练
   - 梯度累积（模拟大batch size）
   - 梯度检查点（节省显存）

3. **模型优化**：
   - LoRA微调（参数高效）
   - 模型量化（降低精度）
   - 知识蒸馏（压缩模型）

---

## 常见问题

### Q1: 显存不足怎么办？

**A**: 尝试以下方法：
1. 减小batch size
2. 使用梯度累积
3. 启用梯度检查点
4. 使用LoRA微调
5. 降低图像分辨率
6. 使用混合精度训练

### Q2: 训练速度慢怎么办？

**A**: 优化建议：
1. 增加num_workers
2. 使用SSD存储数据
3. 预处理数据并缓存
4. 使用混合精度
5. 检查数据加载瓶颈

### Q3: 如何解决过拟合？

**A**: 防止过拟合的方法：
1. 增加数据增强
2. 使用Dropout
3. 增加weight_decay
4. 早停策略
5. 增加训练数据
6. 使用标签平滑
7. 参考`防止过拟合优化.md`

### Q4: 验证指标不稳定怎么办？

**A**: 提高稳定性：
1. 增大验证集大小
2. 使用EMA
3. 延长warmup阶段
4. 降低学习率
5. 增加batch size

### Q5: 如何选择损失函数？

**A**: 选择建议：
- **标准任务**：CLIPLoss
- **困难样本多**：HardNegativeLoss
- **类别不平衡**：FocalCLIPLoss
- **大规模训练**：InfoNCELoss
- **复杂场景**：HybridLoss

---

## 贡献指南

### 代码规范

1. **Python风格**：遵循PEP 8
2. **命名规范**：
   - 类名：大驼峰（CamelCase）
   - 函数名：下划线（snake_case）
   - 常量：全大写（UPPER_CASE）
3. **文档字符串**：所有公共函数需要docstring
4. **类型注解**：关键函数添加类型注解

### 添加新模型

1. 在`models/`下创建新的模型文件
2. 继承`nn.Module`并实现必要接口
3. 在`model_factory.py`中注册新模型
4. 添加配置模板到`config/`
5. 更新文档

### 添加新损失函数

1. 在`loss.py`中实现新的损失类
2. 在`LossFactory`中注册
3. 添加使用示例
4. 更新文档

### 测试

```bash
# 运行单元测试
python -m pytest test/

# 运行集成测试
bash scripts/test_run_train.sh
```

---

## 更新日志

### v2.0（当前版本）
- ✅ 重构为模块化架构
- ✅ 支持多种CLIP模型
- ✅ 完善的配置管理
- ✅ 丰富的损失函数
- ✅ 全面的评估体系
- ✅ LoRA微调支持
- ✅ EMA支持
- ✅ 断点续训
- ✅ 混合精度训练

### v1.0
- 初始版本
- 基础CLIP训练功能

---

## 参考文献

1. **CLIP**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
2. **ConvNeXt**: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
3. **LoRA**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
4. **CN-CLIP**: [Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese](https://arxiv.org/abs/2211.01335)

---

## 许可证

[根据实际情况填写]

---

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件
- 项目Wiki

---

## 致谢

感谢所有为本项目做出贡献的研究者和开发者。

---

**最后更新时间**：2025年10月
