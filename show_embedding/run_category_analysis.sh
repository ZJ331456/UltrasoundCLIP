#!/bin/bash
# 运行图像编码器类别分析脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/media/ps/data-ssd/UltrasoundRAG/CLIP:$PYTHONPATH"

# 进入脚本目录
cd /media/ps/data-ssd/UltrasoundRAG/CLIP/show_embedding

# 创建输出目录
mkdir -p output/category_analysis

# 运行分析脚本
echo "开始运行图像编码器类别分析..."
echo "时间: $(date)"
echo "GPU设备: $CUDA_VISIBLE_DEVICES"

# 运行Python脚本
python compare_encoders_by_category.py

echo "分析完成！时间: $(date)"
echo "结果保存在: output/category_analysis/"
