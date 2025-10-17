#!/bin/bash
# 模型性能分析脚本
# 基于run_eval_finetuned.sh的评估方式，运行全面的模型性能分析

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1

# 配置信息
CONFIG_FILE="config/convnext/convnext_config_2_combined_best_all_data_mask_3_v1.json"
MODEL_PATH="/media/ps/data-ssd/UltrasoundRAG/CLIP/output/convnext_config_2_combined_best_all_data_mask_3_v1/best_model.pth"
DEVICE="cuda:0"
OUTPUT_DIR="analysis_results/convnext_mask_3_v1_$(date +%Y%m%d_%H%M%S)"
MAX_BATCHES=50
ANALYSIS_TYPE="comprehensive"  # comprehensive, basic, visualization, feature, error, ablation

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
echo "启动模型性能分析..."
echo "配置信息:"
echo "  - 配置文件: $CONFIG_FILE"
echo "  - 模型权重: $MODEL_PATH"
echo "  - 使用设备: $DEVICE"
echo "  - GPU设备: $CUDA_VISIBLE_DEVICES"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 分析类型: $ANALYSIS_TYPE"
echo "  - 最大批次: $MAX_BATCHES"
echo ""

# 运行分析脚本
echo "开始分析模型性能..."
python analysize/run_analysis.py \
    --config "$CONFIG_FILE" \
    --model_path "$MODEL_PATH" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --max_batches "$MAX_BATCHES" \
    --analysis_type "$ANALYSIS_TYPE"

echo ""
echo "模型性能分析完成！"
echo "结果已保存到: $OUTPUT_DIR"
echo ""
echo "生成的文件:"
find "$OUTPUT_DIR" -type f -name "*.png" -o -name "*.md" -o -name "*.json" | head -10
echo ""
echo "可以使用以下命令查看结果:"
echo "  - 查看综合报告: cat $OUTPUT_DIR/comprehensive_analysis_report.md"
echo "  - 查看错误分析: cat $OUTPUT_DIR/error_analysis/error_analysis_report.md"
echo "  - 查看消融实验: cat $OUTPUT_DIR/ablation_study/ablation_study_report.md"
echo ""
echo "可视化结果:"
echo "  - 注意力图: $OUTPUT_DIR/visualizations/attention_maps/"
echo "  - 特征空间: $OUTPUT_DIR/visualizations/feature_space/"
echo "  - 错误案例: $OUTPUT_DIR/error_analysis/"
echo ""
echo "可以使用以下命令比较不同模型的分析结果:"
echo "diff analysis_results/model1/comprehensive_analysis_report.md analysis_results/model2/comprehensive_analysis_report.md"
