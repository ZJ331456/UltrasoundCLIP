#!/bin/bash

echo "开始SAM+BERT模型嵌入向量可视化对比..."

# 设置工作目录
cd /home/zhoujun/NormalUltraCLIP

# 创建输出目录
mkdir -p show_embedding/output

# 设置GPU
export CUDA_VISIBLE_DEVICES=1,2,3

echo "1. 运行预训练模型可视化..."
python show_embedding/sam_bert_pretrained_svd.py
if [ $? -eq 0 ]; then
    echo "✓ 预训练模型可视化完成"
else
    echo "✗ 预训练模型可视化失败"
fi

echo ""
echo "2. 运行微调模型可视化..."
python show_embedding/sam_bert_finetuned_svd.py
if [ $? -eq 0 ]; then
    echo "✓ 微调模型可视化完成"
else
    echo "✗ 微调模型可视化失败"
fi

echo ""
echo "可视化完成！结果保存在 show_embedding/output/ 目录下"
echo "生成的文件："
ls -la show_embedding/output/
echo ""
echo "  - sam_bert_pretrained_*_svd.png: 预训练模型结果"
echo "  - sam_bert_finetuned_*_svd.png: 微调模型结果"
echo "  - pretrained_dataset_info.txt: 预训练模型数据信息"
echo "  - finetuned_dataset_info.txt: 微调模型数据信息"
