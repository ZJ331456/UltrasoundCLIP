"""
统一测试脚本 - 测试所有模型的加载和基本功能
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import json


def test_sam_bert_post_model():
    """测试SAM+BERT训练后模型"""
    print("=" * 50)
    print("测试SAM+BERT训练后模型")
    print("=" * 50)
    
    try:
        from models.ultrasam_bert_post_model import load_sam_bert_post_model
        
        weight_path = "/home/zhoujun/CLIP/output/ultrasam_enhanced_clip_20epoch/best.pt"
        config_path = "/home/zhoujun/CLIP/config/ultrasam/universal_sam_bert.json"
        device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
        
        # 加载模型
        model, tokenizer = load_sam_bert_post_model(weight_path, config_path, device)
        
        # 测试数据
        test_images = torch.randn(1, 3, 1024, 1024).to(device)
        test_texts = ["胎儿超声图像显示正常发育"]
        text_inputs = tokenizer(
            test_texts, padding=True, truncation=True, 
            max_length=256, return_tensors='pt'
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        # 前向传播测试
        with torch.no_grad():
            image_features = model.encode_image(test_images)
            text_features = model.encode_text(text_inputs)
            
        print(f"✅ 测试成功!")
        print(f"   图像特征: {image_features.shape}")
        print(f"   文本特征: {text_features.shape}")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_sam_bert_pre_model():
    """测试SAM+BERT预训练模型"""
    print("=" * 50)
    print("测试SAM+BERT预训练模型")
    print("=" * 50)
    
    try:
        from models.ultrasam_bert_pre_model import SAMBERTPreModel
        
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
        test_images = torch.randn(1, 3, 1024, 1024).to(device)
        test_texts = ["胎儿超声图像显示正常"]
        text_inputs = model.tokenizer(
            test_texts, padding=True, truncation=True,
            max_length=256, return_tensors='pt'
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        # 前向传播测试
        model.eval()
        with torch.no_grad():
            image_features = model.encode_image(test_images)
            text_features = model.encode_text(text_inputs)
            
        print(f"✅ 测试成功!")
        print(f"   图像特征: {image_features.shape}")
        print(f"   文本特征: {text_features.shape}")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_fetalclip_post_model():
    """测试FetalCLIP训练后模型"""
    print("=" * 50)
    print("测试FetalCLIP训练后模型")
    print("=" * 50)
    
    try:
        from models.fetalclip_post_model import load_fetalclip_post_model
        
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
        test_images = torch.randn(1, 3, 224, 224).to(device)
        test_texts = tokenizer(["测试文本"])
        
        # 前向传播测试
        with torch.no_grad():
            image_features = model.encode_image(test_images)
            text_features = model.encode_text(test_texts)
            
        print(f"✅ 测试成功!")
        print(f"   图像特征: {image_features.shape}")
        print(f"   文本特征: {text_features.shape}")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_fetalclip_pre_model():
    """测试FetalCLIP预训练模型"""
    print("=" * 50)
    print("测试FetalCLIP预训练模型")
    print("=" * 50)
    
    try:
        from models.fetalclip_pre_model import FetalCLIPPreModel
        
        config_path = "/home/zhoujun/NormalUltraCLIP/config/fetal/FetalCLIP_config.json"
        
        # 检查文件是否存在
        if not os.path.exists(config_path):
            print(f"⚠️  配置文件不存在，跳过测试: {config_path}")
            return True
        
        # 创建模型（不加载预训练权重）
        model = FetalCLIPPreModel(config_path)
        device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # 测试数据
        test_images = torch.randn(1, 3, 224, 224).to(device)
        test_texts = model.tokenizer(["测试文本"]).to(device)
        
        # 前向传播测试
        model.eval()
        with torch.no_grad():
            image_features = model.encode_image(test_images)
            text_features = model.encode_text(test_texts)
            
        print(f"✅ 测试成功!")
        print(f"   图像特征: {image_features.shape}")
        print(f"   文本特征: {text_features.shape}")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 开始测试所有模型...")
    
    # 测试所有模型
    tests = [
        ("SAM+BERT训练后模型", test_sam_bert_post_model),
        ("SAM+BERT预训练模型", test_sam_bert_pre_model),
        ("FetalCLIP训练后模型", test_fetalclip_post_model),
        ("FetalCLIP预训练模型", test_fetalclip_pre_model),
    ]
    
    results = []
    for model_name, test_func in tests:
        print()
        success = test_func()
        results.append((model_name, success))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for model_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{model_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 个模型测试通过")
    
    if passed == total:
        print("🎉 所有模型测试通过!")
    else:
        print("⚠️  部分模型测试失败")


if __name__ == "__main__":
    main()
