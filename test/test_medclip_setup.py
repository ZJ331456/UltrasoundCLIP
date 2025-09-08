#!/usr/bin/env python3
"""
MedCLIP配置测试脚本
验证模型加载和基本功能
"""

import sys
import os
import json
import torch
from pathlib import Path

# 添加路径
sys.path.append('/media/ps/data-ssd/UltrasoundRAG/CLIP')
sys.path.append('/media/ps/data-ssd/UltrasoundRAG/CLIP/models/MedCLIP')

def test_medclip_setup():
    """测试MedCLIP设置"""
    print("=== MedCLIP配置测试 ===")
    
    # 1. 测试配置文件加载
    config_path = "/media/ps/data-ssd/UltrasoundRAG/CLIP/config/qwen/qwen_vl_clip_opt_6_medclip.json"
    print(f"测试配置文件: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"✓ 配置文件加载成功")
    print(f"  模型类型: {config['model']['type']}")
    print(f"  嵌入维度: {config['model']['embed_dim']}")
    
    # 2. 测试预训练权重
    checkpoint_path = config['model']['vision_encoder']['checkpoint_path']
    print(f"\n测试预训练权重: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 预训练权重目录不存在: {checkpoint_path}")
        return False
    
    weight_file = os.path.join(checkpoint_path, "pytorch_model.bin")
    if not os.path.exists(weight_file):
        print(f"❌ 权重文件不存在: {weight_file}")
        return False
    
    print(f"✓ 预训练权重检查通过")
    
    # 3. 测试MedCLIP依赖
    print(f"\n测试MedCLIP依赖...")
    try:
        from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
        print(f"✓ MedCLIP导入成功")
    except ImportError as e:
        print(f"❌ MedCLIP导入失败: {e}")
        print("请运行: pip install medclip")
        return False
    
    # 4. 测试模型创建
    print(f"\n测试模型创建...")
    try:
        from models.model_factory import ModelFactory
        
        model = ModelFactory.create_model(config['model'])
        print(f"✓ MedCLIP包装器创建成功")
        print(f"  模型类型: {type(model)}")
        
        # 测试设备移动
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"✓ 模型移动到CUDA成功")
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False
    
    # 5. 测试文本编码器
    print(f"\n测试文本编码器...")
    try:
        test_texts = ["超声图像显示正常的胎儿心脏结构"]
        text_features = model.encode_text(test_texts)
        print(f"✓ 文本编码成功")
        print(f"  文本特征形状: {text_features.shape}")
    except Exception as e:
        print(f"❌ 文本编码失败: {e}")
        return False
    
    print(f"\n=== 所有测试通过! ===")
    print(f"MedCLIP验证实验环境准备就绪")
    print(f"可以运行训练脚本: bash run_qwen_train_optimized_5_medclip.sh")
    
    return True

if __name__ == "__main__":
    success = test_medclip_setup()
    sys.exit(0 if success else 1)
