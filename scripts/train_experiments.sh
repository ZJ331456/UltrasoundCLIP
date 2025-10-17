#!/bin/bash
# 实验训练脚本 - 用于运行不同版本的超声图文对齐实验

# 设置基础路径
BASE_DIR="/media/ps/data-ssd/UltrasoundRAG/CLIP"
CONFIG_DIR="${BASE_DIR}/config/convnext"
LOG_DIR="${BASE_DIR}/logs/experiments"
CHECKPOINT_DIR="${BASE_DIR}/checkpoints/experiments"

# 创建必要的目录
mkdir -p ${LOG_DIR}
mkdir -p ${CHECKPOINT_DIR}

# 基线模型（无mask监督）
echo "=== 训练基线模型（无mask监督） ==="
python ${BASE_DIR}/train.py \
    --config ${CONFIG_DIR}/convnext_config_2_combined_best_all_data_mask.json \
    --save_dir ${CHECKPOINT_DIR}/baseline \
    2>&1 | tee ${LOG_DIR}/baseline.log

# 版本1：Attention Side-Head + Mask Supervision
echo "=== 训练版本1：Attention Side-Head + Mask Supervision ==="
python ${BASE_DIR}/train.py \
    --config ${CONFIG_DIR}/convnext_config_2_combined_best_all_data_mask_v1.json \
    --save_dir ${CHECKPOINT_DIR}/v1_mask_supervision \
    2>&1 | tee ${LOG_DIR}/v1_mask_supervision.log

# 版本2：添加Gating机制
echo "=== 训练版本2：Gating融合机制 ==="
python ${BASE_DIR}/train.py \
    --config ${CONFIG_DIR}/convnext_config_2_combined_best_all_data_mask_v2.json \
    --save_dir ${CHECKPOINT_DIR}/v2_gating \
    2>&1 | tee ${LOG_DIR}/v2_gating.log

# 版本3：多尺度特征
echo "=== 训练版本3：多尺度特征提取 ==="
python ${BASE_DIR}/train.py \
    --config ${CONFIG_DIR}/convnext_config_2_combined_best_all_data_mask_v3.json \
    --save_dir ${CHECKPOINT_DIR}/v3_multiscale \
    2>&1 | tee ${LOG_DIR}/v3_multiscale.log

# 版本4：稀疏注意力正则化
echo "=== 训练版本4：稀疏注意力正则化 ==="
python ${BASE_DIR}/train.py \
    --config ${CONFIG_DIR}/convnext_config_2_combined_best_all_data_mask_v4.json \
    --save_dir ${CHECKPOINT_DIR}/v4_sparse_attention \
    2>&1 | tee ${LOG_DIR}/v4_sparse_attention.log

echo "=== 所有实验训练已启动 ==="
echo "日志保存在: ${LOG_DIR}"
echo "模型保存在: ${CHECKPOINT_DIR}"
