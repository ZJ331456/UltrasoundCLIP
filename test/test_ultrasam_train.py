import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from CLIP.models.ultrasam_bert_post_model import load_sam_bert_post_model
import json

def test_ultrasam_train_model():
    """测试UltraSAM+BERT训练后模型"""
    print("=" * 60)
    print("测试UltraSAM+BERT训练后模型")
    print("=" * 60)
    
    # 使用真实的配置文件路径
    weight_path = "/home/zhoujun/CLIP/output/ultrasam_enhanced_clip_20epoch/best.pt"
    config_path = "/home/zhoujun/CLIP/config/ultrasam/universal_sam_bert.json"  # 使用真实配置
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    
    print(f"权重文件: {weight_path}")
    print(f"配置文件: {config_path}")
    print(f"使用设备: {device}")
    
    try:
        # 加载训练后的模型
        print("\n正在加载UltraSAM+BERT训练后模型...")
        model, tokenizer = load_sam_bert_post_model(weight_path, config_path, device)
        print("模型加载成功！")
        
        # 创建测试数据
        print("\n正在准备测试数据...")
        
        # SAM需要1024x1024的输入图像
        dummy_images = torch.randn(2, 3, 1024, 1024).to(device)
        print(f"测试图像形状: {dummy_images.shape}")
        
        # 创建文本输入（BERT格式）
        test_texts = ["胎儿超声图像显示正常发育", "心脏四腔心切面结构清晰"]
        dummy_text_inputs = tokenizer(
            test_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        # 将文本输入移动到设备
        dummy_text_inputs = {k: v.to(device) for k, v in dummy_text_inputs.items()}
        
        print(f"测试文本: {test_texts}")
        print(f"文本token形状: {dummy_text_inputs['input_ids'].shape}")
        
        # 前向传播测试
        print("\n正在进行前向传播...")
        model.eval()
        with torch.no_grad():
            # 分别编码图像和文本
            image_features = model.encode_image(dummy_images)
            text_features = model.encode_text(dummy_text_inputs)
            
            # 联合前向传播
            img_feat, txt_feat = model(dummy_images, dummy_text_inputs)
            
        # 输出结果
        print("\n✅ 模型测试成功!")
        print(f"图像特征形状: {image_features.shape}")
        print(f"文本特征形状: {text_features.shape}")
        print(f"图像特征范围: [{image_features.min():.4f}, {image_features.max():.4f}]")
        print(f"文本特征范围: [{text_features.min():.4f}, {text_features.max():.4f}]")
        
        # 计算相似度
        similarity = torch.mm(image_features, text_features.t())
        print(f"图像-文本相似度矩阵形状: {similarity.shape}")
        # print(f"相似度矩阵:\n{similarity}")
        
        # 验证特征是否归一化
        img_norm = torch.norm(image_features, dim=1)
        txt_norm = torch.norm(text_features, dim=1)
        print(f"图像特征L2范数: {img_norm}")
        print(f"文本特征L2范数: {txt_norm}")
        
        # 额外测试：单个样本推理
        print("\n正在测试单个样本推理...")
        single_image = dummy_images[:1]
        single_text = {k: v[:1] for k, v in dummy_text_inputs.items()}
        
        with torch.no_grad():
            single_img_feat = model.encode_image(single_image)
            single_txt_feat = model.encode_text(single_text)
            single_similarity = torch.cosine_similarity(single_img_feat, single_txt_feat, dim=1)
            
        print(f"单样本相似度: {single_similarity.item():.4f}")
        
        print("\n🎉 UltraSAM+BERT训练后模型测试完成!")
        
        # 性能提示
        print("\n📊 性能信息:")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("请检查权重文件和配置文件路径是否正确")
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()

def create_sample_config():
    """创建示例配置文件"""
    sample_config = {
        "model": {
            "embed_dim": 768,
            "temperature": 0.07,
            "image_encoder": {
                "type": "sam",
                "model_name": "vit_l",
                "checkpoint_path": None
            },
            "text_encoder": {
                "type": "bert",
                "model_name": "hfl/chinese-macbert-base"
            }
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "num_epochs": 100,
            "warmup_steps": 1000
        }
    }
    
    config_path = "/tmp/ultrasam_bert_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=2, ensure_ascii=False)
    
    print(f"示例配置文件已创建: {config_path}")
    return config_path

if __name__ == "__main__":
    # 如果配置文件不存在，可以先创建示例配置
    # config_path = create_sample_config()
    
    test_ultrasam_train_model()
