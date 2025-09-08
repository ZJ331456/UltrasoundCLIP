#!/bin/bash
# 微调后模型评估脚本
# 用于评估加载微调权重后的模型性能

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1

# 配置信息
CONFIG_FILE="config/convnext/convnext_config_2_combined_best.json"
MODEL_PATH="/media/ps/data-ssd/UltrasoundRAG/CLIP/output/convnext_config_2_combined_best/best_model.pth"
DATASET="test"
DEVICE="cuda:0"
OUTPUT_DIR="eval_results/8-26/medclip/medclip_pretrain_finetuned"
EVAL_TYPE="both"  # standard, finegrained, both
MAX_BATCHES=100
EVAL_BATCH_SIZE=32768  # CN-CLIP optimized batch size

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "模型文件不存在: $MODEL_PATH"
    echo "请检查模型路径是否正确或选择其他模型文件"
    echo "可选模型文件:"
    find output/ -name "*.pth" -type f 2>/dev/null | head -5
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 打印配置信息
echo "启动微调后模型评估..."
echo "配置信息:"
echo "  - 配置文件: $CONFIG_FILE"
echo "  - 模型权重: $MODEL_PATH"
echo "  - 评估数据集: $DATASET"
echo "  - 使用设备: $DEVICE"
echo "  - GPU设备: $CUDA_VISIBLE_DEVICES"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 评估类型: $EVAL_TYPE"
echo "  - 最大批次: $MAX_BATCHES"
echo ""

# 运行评估脚本（加载微调权重）
echo "开始评估微调模型..."
python eval.py \
    --config "$CONFIG_FILE" \
    --model_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --eval_type "$EVAL_TYPE" \
    --max_batches "$MAX_BATCHES" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --verbose

echo ""
echo "微调模型评估完成！"
echo "结果已保存到: $OUTPUT_DIR"
echo "  - standard_evaluation.json: 标准检索指标"
echo "  - finegrained_evaluation.json: 细粒度分析"
echo "  - comprehensive_evaluation.json: 综合评估结果"
echo "  - evaluation_summary.md: 评估摘要报告"
echo ""
echo "可以使用以下命令比较两个模型的性能:"
echo "diff eval_results/pretrain_model/evaluation_summary.md eval_results/finetuned_model/evaluation_summary.md"
