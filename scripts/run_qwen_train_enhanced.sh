#!/bin/bash
# 增强版Qwen CLIP训练脚本
# 优化对比学习效果，提高正负样本区分度

# 设置环境变量
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,expandable_segments:True

# 配置信息
CONFIG_FILE="config/qwen/qwen_vl_clip_opt_4.json"
SAVE_DIR="output/qwen_vl_clip_8_enhanced_contrast"
DEVICE="cuda:0"
DEBUG_MODE=false
CHECKPOINT=""

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 清理GPU内存
echo "清理GPU内存..."
nvidia-smi --gpu-reset -i 1 2>/dev/null || echo "无法重置GPU，继续..."

# 创建保存目录
mkdir -p "$SAVE_DIR"

# 打印配置信息
echo "启动增强版Qwen CLIP训练..."
echo "主要优化："
echo "  - 降低温度参数 (0.05) 增加对比尖锐度"
echo "  - 增强硬负样本挖掘 (50%)"
echo "  - 新增三元组边界损失"
echo "  - 自适应焦点损失"
echo "  - 对比正则化防止特征坍塌"
echo "  - 分层学习率优化"
echo "  - 增强数据增强策略"
echo ""
echo "配置信息:"
echo "  - 配置文件: $CONFIG_FILE"
echo "  - 保存目录: $SAVE_DIR"
echo "  - 使用设备: $DEVICE"
echo "  - GPU设备: $CUDA_VISIBLE_DEVICES"
echo "  - Batch Size: 24 (优化后)"
echo "  - 学习率: 3e-5 (分层优化)"
echo "  - 训练轮数: 25 (增加)"
echo "  - 温度参数: 0.05 (降低增加尖锐度)"

# 运行训练脚本
if [ "$DEBUG_MODE" = true ]; then
    echo "调试模式..."
    python train.py --config "$CONFIG_FILE" --save_dir "$SAVE_DIR" --device "$DEVICE" --debug
else
    echo "开始增强训练..."
    nohup python train.py --config "$CONFIG_FILE" --save_dir "$SAVE_DIR" --device "$DEVICE" > logs/training_qwen_8_enhancelosss.log 2>&1 &
    echo "训练已在后台启动，日志文件: logs/training_qwen_enhanced.log"
    echo "使用 tail -f logs/training_qwen_enhanced.log 查看训练进度"
fi

echo "增强版训练启动完成！"
