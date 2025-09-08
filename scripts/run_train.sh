#!/bin/bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=1,3
# 配置信息
CONFIG_FILE="config/ultrasam/sam_bert_train.json"
SAVE_DIR="output/test"
DEVICE="cuda:1"
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
echo "启动CLIP训练..."
echo "配置信息:"
echo "  - 配置文件: $CONFIG_FILE"
echo "  - 保存目录: $SAVE_DIR"
echo "  - 使用设备: $DEVICE"
echo "  - GPU设备: $CUDA_VISIBLE_DEVICES"

# 运行训练脚本
if [ "$DEBUG_MODE" = true ]; then
    echo "调试模式..."
    python train.py --config "$CONFIG_FILE" --save_dir "$SAVE_DIR" --device "$DEVICE" --debug
else
    echo "开始训练..."
    python train.py --config "$CONFIG_FILE" --save_dir "$SAVE_DIR" --device "$DEVICE"
fi

echo "训练完成！"
