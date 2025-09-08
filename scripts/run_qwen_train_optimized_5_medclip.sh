#!/bin/bash

# 设置环境变量 - MedCLIP验证实验配置
export PYTHONPATH="/media/ps/data-ssd/UltrasoundRAG/CLIP:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0

# 高性能设置
export OMP_NUM_THREADS=8
export CUDA_CACHE_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# MedCLIP验证实验参数配置
CONFIG_FILE="/media/ps/data-ssd/UltrasoundRAG/CLIP/config/qwen/qwen_vl_clip_opt_6_medclip.json"
SAVE_DIR="/media/ps/data-ssd/UltrasoundRAG/CLIP/output/medclip_validation_experiment_2"
LOG_FILE="/media/ps/data-ssd/UltrasoundRAG/CLIP/logs/training_medclip_validation_2.log"

# 创建必要的目录
mkdir -p "$(dirname "$SAVE_DIR")"
mkdir -p "$(dirname "$LOG_FILE")"

echo "=== MedCLIP验证实验 - 超声CLIP训练框架问题排查 ==="
echo "配置文件: $CONFIG_FILE"
echo "保存目录: $SAVE_DIR"
echo "日志文件: $LOG_FILE"
echo "GPU设备: $CUDA_VISIBLE_DEVICES"
echo ""

# 检查预训练权重
MEDCLIP_WEIGHTS="/media/ps/data-ssd/UltrasoundRAG/CLIP/checkpoints/MedCLIP/medclip-vit-pretrained/pytorch_model.bin"
if [ ! -f "$MEDCLIP_WEIGHTS" ]; then
    echo "错误: MedCLIP预训练权重不存在: $MEDCLIP_WEIGHTS"
    echo "请确保已下载MedCLIP预训练模型"
    exit 1
fi

echo "✓ MedCLIP预训练权重检查通过: $MEDCLIP_WEIGHTS"

# 检查GPU可用性
if ! nvidia-smi > /dev/null 2>&1; then
    echo "错误: 未检测到NVIDIA GPU或驱动程序"
    exit 1
fi

echo "GPU状态:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
echo ""

echo "MedCLIP验证实验配置:"
echo "  - 视觉编码器: MedCLIP-ViT (预训练)"
echo "  - 文本编码器: BiomedBERT (与原实验相同)"
echo "  - 图像尺寸: 224x224 (适配MedCLIP)"
echo "  - Batch Size: 16"
echo "  - 学习率: 5e-5"
echo "  - 实验目的: 验证训练框架稳定性"
echo "  - 控制变量: 仅更换视觉编码器"
echo ""

echo "实验假设:"
echo "  ✓ 如果MedCLIP收敛稳定 → 原框架正常，问题在模型选择"
echo "  ✗ 如果MedCLIP仍过拟合 → 框架存在问题，需要调优训练策略"
echo ""

# 启动训练
echo "开始MedCLIP验证实验..."
python train.py \
    --config "$CONFIG_FILE" \
    --save_dir "$SAVE_DIR" \
    2>&1 | tee "$LOG_FILE"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=== MedCLIP验证实验完成 ==="
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
    echo "实验结果分析指南:"
    echo "1. 检查训练loss曲线是否平滑下降"
    echo "2. 观察验证loss是否出现过拟合"
    echo "3. 对比原实验，判断问题根源："
    echo "   - 如果MedCLIP表现良好 → 考虑Qwen视觉编码器适配问题"
    echo "   - 如果MedCLIP仍有问题 → 检查数据处理和训练策略"
    echo ""
    echo "下一步建议:"
    echo "- 如果验证成功，可以尝试ConvNeXt: bash run_convnext_train.sh"
    echo "- 如果问题依然存在，需要调整数据增强和正则化策略"
    
else
    echo ""
    echo "=== MedCLIP验证实验失败 ==="
    echo "请检查日志文件: $LOG_FILE"
    echo "错误代码: $?"
    
    # 显示GPU状态以便调试
    echo ""
    echo "失败时GPU状态:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
    
    echo ""
    echo "常见问题排查:"
    echo "1. 检查MedCLIP依赖是否正确安装: pip list | grep medclip"
    echo "2. 确认预训练权重路径是否正确"
    echo "3. 验证CUDA内存是否足够"
    echo "4. 检查数据路径是否可访问"
    
    exit 1
fi
