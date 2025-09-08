import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from CLIP.models.ultrasam_bert_pre_model import SAMBERTPreModel
import json

def test_ultrasam_pretrain_model():
    """测试UltraSAM+BERT预训练模型"""
    print("=" * 60)
    print("测试UltraSAM+BERT预训练模型")
    print("=" * 60)
    
    # 加载真实配置文件
    config_path = "/home/zhoujun/CLIP/config/ultrasam/universal_sam_bert.json"
    print(f"加载配置文件: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = json.load(f)
    
    # 提取模型配置
    config = full_config["model"]
    print(f"模型配置加载成功:")
    print(f"  - 嵌入维度: {config['embed_dim']}")
    print(f"  - 图像编码器: SAM {config['image_encoder']['model_name']}")
    print(f"  - 文本编码器: {config['text_encoder']['model_name']}")
    print(f"  - SAM权重路径: {config['image_encoder']['checkpoint_path']}")
    
    try:
        # 创建模型实例
        print("正在创建SAM+BERT模型...")
        model = SAMBERTPreModel(config)
        
        print("模型创建成功！")
        print(f"嵌入维度: {model.embed_dim}")
        print(f"图像编码器类型: SAM {config['image_encoder']['model_name']}")
        print(f"文本编码器类型: {config['text_encoder']['model_name']}")
        
        # 创建测试数据
        print("\n正在准备测试数据...")
        
        # SAM需要1024x1024的输入图像
        dummy_images = torch.randn(2, 3, 1024, 1024)
        print(f"测试图像形状: {dummy_images.shape}")
        
        # 创建文本输入（BERT格式）
        tokenizer = model.tokenizer
        test_texts = ["胎儿超声图像显示正常", "心脏结构清晰可见"]
        dummy_text_inputs = tokenizer(
            test_texts,
            padding=True,
            truncation=True,
            max_length=256,  # 使用配置文件中的max_text_length
            return_tensors='pt'
        )
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
        print(f"相似度矩阵:\n{similarity}")
        
        # 验证特征是否归一化
        img_norm = torch.norm(image_features, dim=1)
        txt_norm = torch.norm(text_features, dim=1)
        print(f"图像特征L2范数: {img_norm}")
        print(f"文本特征L2范数: {txt_norm}")
        
        print("\n🎉 UltraSAM+BERT预训练模型测试完成!")
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ultrasam_pretrain_model()