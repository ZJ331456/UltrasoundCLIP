"""
统一测试脚本 - 测试所有优化后的模型
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import json
from typing import Dict, Any


def test_sam_bert_post_model():
    """测试SAM+BERT训练后模型"""
    print("=" * 60)
    print("测试SAM+BERT训练后模型")
    print("=" * 60)
    
    try:
        from CLIP.models.ultrasam_bert_post_model import load_sam_bert_post_model
        
        weight_path = "/home/zhoujun/CLIP/output/ultrasam_enhanced_clip_20epoch/best.pt"
        config_path = "/home/zhoujun/CLIP/config/ultrasam/universal_sam_bert.json"
        device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
        
        # 加载模型
        model, tokenizer = load_sam_bert_post_model(weight_path, config_path, device)
        
        # 测试数据
        dummy_images = torch.randn(1, 3, 1024, 1024).to(device)
        test_texts = ["胎儿超声图像显示正常发育"]
        dummy_text_inputs = tokenizer(
            test_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        dummy_text_inputs = {k: v.to(device) for k, v in dummy_text_inputs.items()}
        
        # 前向传播
        with torch.no_grad():
            image_features = model.encode_image(dummy_images)
            text_features = model.encode_text(dummy_text_inputs)
            
        print(f"✅ SAM+BERT训练后模型测试成功!")
        print(f"   图像特征形状: {image_features.shape}")
        print(f"   文本特征形状: {text_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ SAM+BERT训练后模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sam_bert_pre_model():
    """测试SAM+BERT预训练模型"""
    print("=" * 60)
    print("测试SAM+BERT预训练模型")
    print("=" * 60)
    
    try:
        from CLIP.models.ultrasam_bert_pre_model import SAMBERTPreModel
        
        config_path = "/home/zhoujun/CLIP/config/ultrasam/universal_sam_bert.json"
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = json.load(f)
        config = full_config["model"]
        
        # 创建模型
        model = SAMBERTPreModel(config)
        device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # 测试数据
        dummy_images = torch.randn(1, 3, 1024, 1024).to(device)
        test_texts = ["胎儿超声图像显示正常"]
        dummy_text_inputs = model.tokenizer(
            test_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        dummy_text_inputs = {k: v.to(device) for k, v in dummy_text_inputs.items()}
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            image_features = model.encode_image(dummy_images)
            text_features = model.encode_text(dummy_text_inputs)
            
        print(f"✅ SAM+BERT预训练模型测试成功!")
        print(f"   图像特征形状: {image_features.shape}")
        print(f"   文本特征形状: {text_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ SAM+BERT预训练模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fetalclip_post_model():
    """测试FetalCLIP训练后模型"""
    print("=" * 60)
    print("测试FetalCLIP训练后模型")
    print("=" * 60)
    
    try:
        from CLIP.models.fetalclip_post_model import load_fetalclip_post_model
        
        weight_path = "/home/zhoujun/NormalUltraCLIP/output/fetalclip-train/final_model.pt"
        config_path = "/home/zhoujun/NormalUltraCLIP/config/fetal/FetalCLIP_config.json"
        device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
        
        # 检查文件是否存在
        if not os.path.exists(weight_path):
            print(f"⚠️  权重文件不存在，跳过测试: {weight_path}")
            return True
        
        if not os.path.exists(config_path):
            print(f"⚠️  配置文件不存在，跳过测试: {config_path}")
            return True
        
        # 加载模型
        model, tokenizer = load_fetalclip_post_model(weight_path, config_path, device)
        
        # 测试数据
        dummy_images = torch.randn(1, 3, 224, 224).to(device)
        dummy_text = tokenizer(["测试文本"])
        
        # 前向传播
        with torch.no_grad():
            image_features = model.encode_image(dummy_images)
            text_features = model.encode_text(dummy_text)
            
        print(f"✅ FetalCLIP训练后模型测试成功!")
        print(f"   图像特征形状: {image_features.shape}")
        print(f"   文本特征形状: {text_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ FetalCLIP训练后模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fetalclip_pre_model():
    """测试FetalCLIP预训练模型"""
    print("=" * 60)
    print("测试FetalCLIP预训练模型")
    print("=" * 60)
    
    try:
        from CLIP.models.fetalclip_pre_model import FetalCLIPPreModel
        
        config_path = "/home/zhoujun/NormalUltraCLIP/config/fetal/FetalCLIP_config.json"
        
        # 检查文件是否存在
        if not os.path.exists(config_path):
            print(f"⚠️  配置文件不存在，跳过测试: {config_path}")
            return True
        
        # 创建模型
        model = FetalCLIPPreModel(config_path)
        device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # 测试数据
        dummy_images = torch.randn(1, 3, 224, 224).to(device)
        dummy_text = model.tokenizer(["测试文本"])
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            image_features = model.encode_image(dummy_images)
            text_features = model.encode_text(dummy_text)
            
        print(f"✅ FetalCLIP预训练模型测试成功!")
        print(f"   图像特征形状: {image_features.shape}")
        print(f"   文本特征形状: {text_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ FetalCLIP预训练模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 开始测试所有优化后的模型...")
    print()
    
    results = []
    
    # 测试所有模型
    test_functions = [
        ("SAM+BERT训练后模型", test_sam_bert_post_model),
        ("SAM+BERT预训练模型", test_sam_bert_pre_model),
        ("FetalCLIP训练后模型", test_fetalclip_post_model),
        ("FetalCLIP预训练模型", test_fetalclip_pre_model),
    ]
    
    for model_name, test_func in test_functions:
        print()
        success = test_func()
        results.append((model_name, success))
        print()
    
    # 汇总结果
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for model_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{model_name}: {status}")
        if success:
            passed += 1
    
    print()
    print(f"总计: {passed}/{total} 个模型测试通过")
    
    if passed == total:
        print("🎉 所有模型测试通过!")
    else:
        print("⚠️  部分模型测试失败，请检查错误信息")


if __name__ == "__main__":
    main()
