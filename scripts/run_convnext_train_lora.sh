#!/bin/bash

# 训练脚本 - ConvNeXt + RoBERTa CLIP

export PYTHONPATH="/media/ps/data-ssd/UltrasoundRAG/CLIP:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0

# 高性能设置
export OMP_NUM_THREADS=8
export CUDA_CACHE_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

CONFIG_FILE="/media/ps/data-ssd/UltrasoundRAG/CLIP/config/convnext/convnext_config_lora_3_new_dataload_updata_data.json"
SAVE_DIR="/media/ps/data-ssd/UltrasoundRAG/CLIP/output/convnext_clip_experiment_lora_5__new_dataload_and_image_encode_updata_data"
LOG_FILE="/media/ps/data-ssd/UltrasoundRAG/CLIP/logs/training_convnext_clip_lora_5__new_dataload_and_image_encode_updata_data.log"

mkdir -p "$(dirname "$SAVE_DIR")"
mkdir -p "$(dirname "$LOG_FILE")"

echo "=== ConvNeXt+RoBERTa CLIP 训练 ==="
echo "配置文件: $CONFIG_FILE"
echo "保存目录: $SAVE_DIR"
echo "日志文件: $LOG_FILE"
echo "GPU设备: $CUDA_VISIBLE_DEVICES"
echo ""

# 启动训练
python train.py \
    --config "$CONFIG_FILE" \
    --save_dir "$SAVE_DIR" \
    --device auto \
    2>&1 | tee "$LOG_FILE"

status=$?
if [ $status -eq 0 ]; then
    echo "\n训练完成，输出目录: $SAVE_DIR"
else
    echo "\n训练失败，请检查日志: $LOG_FILE"
fi


