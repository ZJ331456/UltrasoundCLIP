#!/bin/bash

# 设置环境变量 - 高性能训练配置
export PYTHONPATH="/media/ps/data-ssd/UltrasoundRAG/CLIP:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=3
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0

# 高性能设置
export OMP_NUM_THREADS=8
export CUDA_CACHE_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 训练参数配置
CONFIG_FILE="/media/ps/data-ssd/UltrasoundRAG/CLIP/config/qwen/qwen_vl_clip_opt_6_defaultpolling.json"
SAVE_DIR="/media/ps/data-ssd/UltrasoundRAG/CLIP/output/qwen_vl_clip_14_select_image_defaultpolling"
LOG_FILE="/media/ps/data-ssd/UltrasoundRAG/CLIP/logs/training_qwen_14_select_image_defaultpolling.log"

# 创建必要的目录
mkdir -p "$(dirname "$SAVE_DIR")"
mkdir -p "$(dirname "$LOG_FILE")"

echo "=== 超声图像CLIP训练 - 高资源利用率版本 ==="
echo "配置文件: $CONFIG_FILE"
echo "保存目录: $SAVE_DIR"
echo "日志文件: $LOG_FILE"
echo "GPU设备: $CUDA_VISIBLE_DEVICES"
echo ""

# 检查GPU可用性
if ! nvidia-smi > /dev/null 2>&1; then
    echo "错误: 未检测到NVIDIA GPU或驱动程序"
    exit 1
fi

echo "GPU状态:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
echo ""

echo "训练配置优化:"
echo "  - Batch Size: 32 (提升至原来的2倍)"
echo "  - 梯度累计: 4步 (有效batch: 32×4=128)"
echo "  - 数据加载: 12 workers + prefetch 6"
echo "  - 混合精度: 启用"
echo "  - 预期显存使用: ~35-40G"
echo ""

# 启动训练
echo "开始高性能训练..."
python train.py \
    --config "$CONFIG_FILE" \
    --save_dir "$SAVE_DIR" \
    2>&1 | tee "$LOG_FILE"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=== 训练完成 ==="
    echo "模型保存在: $SAVE_DIR"
    echo "训练日志: $LOG_FILE"
    
    # 显示最新checkpoint
    if [ -d "$SAVE_DIR" ]; then
        echo ""
        echo "保存的checkpoints:"
        find "$SAVE_DIR" -name "*.pth" -type f -exec ls -lh {} \; | sort -k9
    fi
    
    # 显示最终GPU状态
    echo ""
    echo "训练结束时GPU状态:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
    
    echo ""
    echo "可以运行评估脚本验证模型性能:"
    echo "bash run_qwen_eval_debug.sh"
    
else
    echo ""
    echo "=== 训练失败 ==="
    echo "请检查日志文件: $LOG_FILE"
    echo "错误代码: $?"
    
    # 显示GPU状态以便调试
    echo ""
    echo "失败时GPU状态:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
    
    exit 1
fi
