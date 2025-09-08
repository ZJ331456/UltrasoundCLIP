#!/bin/bash
# 微调前模型评估脚本
# 用于评估基础预训练模型的性能

# 设置环境变量
export CUDA_VISIBLE_DEVICES=2

# 配置信息
CONFIG_FILE="config/qwen/qwen_vl_clip_opt_6_medclip.json"
DATASET="test"
DEVICE="cuda:0"
OUTPUT_DIR="eval_results/8-26/medclip/medclip_pretrain"
EVAL_TYPE="both"  # standard, finegrained, both
MAX_BATCHES=100
EVAL_BATCH_SIZE=32768

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 打印配置信息
echo "启动微调前模型评估..."
echo "配置信息:"
echo "  - 配置文件: $CONFIG_FILE"
echo "  - 评估数据集: $DATASET"
echo "  - 使用设备: $DEVICE"
echo "  - GPU设备: $CUDA_VISIBLE_DEVICES"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 评估类型: $EVAL_TYPE"
echo "  - 最大批次: $MAX_BATCHES"
echo ""

# 运行评估脚本（不加载微调权重）
echo "开始评估预训练模型..."
python eval.py \
    --config "$CONFIG_FILE" \
    --dataset "$DATASET" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --eval_type "$EVAL_TYPE" \
    --max_batches "$MAX_BATCHES" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --verbose

echo ""
echo "预训练模型评估完成！"
echo "结果已保存到: $OUTPUT_DIR"
echo "  - standard_evaluation.json: 标准检索指标"
echo "  - finegrained_evaluation.json: 细粒度分析"
echo "  - comprehensive_evaluation.json: 综合评估结果"
echo "  - evaluation_summary.md: 评估摘要报告"
