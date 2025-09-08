import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from CLIP.models.fetalclip_pre_model import FetalCLIPModel

def test_fetalclip_model():
    # 配置文件路径
    config_path = "/home/zhoujun/NormalUltraCLIP/config/fetal/FetalCLIP_config.json"  # 替换为你的配置文件路径
    pretrained_path = "/home/zhoujun/NormalUltraCLIP/checkpoints/FetalCLIP_Pretrain/FetalCLIP_weights.pt"  # 替换为你的预训练权重路径

    # 创建模型实例
    model = FetalCLIPModel(config_path=config_path, pretrained_path=pretrained_path)

    # 测试图像和文本输入
    dummy_images = torch.randn(1, 3, 224, 224)  # 假设输入图像大小为 (1, 3, 224, 224)
    dummy_text_tokens = torch.randint(0, 1000, (1, 117))  # 文本 token 长度为 117

    # 前向传播
    image_features, text_features = model(dummy_images, dummy_text_tokens)

    # 输出特征
    print("图像特征:", image_features)
    print("文本特征:", text_features)

if __name__ == "__main__":
    test_fetalclip_model()