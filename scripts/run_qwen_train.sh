#!/bin/bash
# Qwen2.5-VL CLIP 模型训练脚本

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clip

# 设置环境变量
export CUDA_VISIBLE_DEVICES=3
export HF_ENDPOINT=https://hf-mirror.com
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# 配置信息
CONFIG_FILE="config/qwen/qwen_vl_clip_opt_2.json"
SAVE_DIR="output/qwen_vl_clip_5_clip_loss"
DEVICE="cuda:0"
DEBUG_MODE=false
CHECKPOINT=""

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 创建保存目录
mkdir -p "$SAVE_DIR"

# 打印配置信息
echo "启动Qwen2.5-VL CLIP训练..."
echo "配置信息:"
echo "  - 配置文件: $CONFIG_FILE"
echo "  - 保存目录: $SAVE_DIR"
echo "  - 使用设备: $DEVICE"
echo "  - GPU设备: $CUDA_VISIBLE_DEVICES"

# 检查数据集路径
echo "检查数据集..."
DATA_ROOT="/media/ps/data/Datasets/ultrasound"
TRAIN_JSON="/media/ps/data-ssd/UltrasoundRAG/clip_caption/caption_breast/caption_breast_train.json"

if [ ! -d "$DATA_ROOT" ]; then
    echo "警告: 数据根目录不存在: $DATA_ROOT"
fi

if [ ! -f "$TRAIN_JSON" ]; then
    echo "警告: 训练数据JSON不存在: $TRAIN_JSON"
fi

# 检查模型路径
MODEL_PATH="/media/ps/data-ssd/DolphinV1/dolphinV1_3B"
if [ ! -d "$MODEL_PATH" ]; then
    echo "警告: Qwen2.5-VL模型路径不存在: $MODEL_PATH"
fi

# 清理GPU内存
echo "清理GPU内存..."
nvidia-smi --gpu-reset 2>/dev/null || echo "无法重置GPU，继续..."

# 运行训练脚本
if [ "$DEBUG_MODE" = true ]; then
    echo "调试模式..."
    python train.py --config "$CONFIG_FILE" --save_dir "$SAVE_DIR" --device "$DEVICE" --debug
else
    echo "开始训练..."
    python train.py --config "$CONFIG_FILE" --save_dir "$SAVE_DIR" --device "$DEVICE"
fi

echo "训练完成！"

# 使用方式：
# CUDA_VISIBLE_DEVICES=3 nohup bash run_qwen_train.sh > training_qwen_2.log 2>&1 &