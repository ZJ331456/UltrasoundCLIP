import sys
sys.path.append('/media/ps/data-ssd/UltrasoundRAG/CLIP')

import torch
from models.qwen_image_encodel import QwenBERTModel

def test_qwen_bert():
    # 配置文件
    config = {
        'image_encoder': {
            'model_path': '/media/ps/data-ssd/Video_XL/Qwen2.5-VL-3B-Instruct'
        },
        'text_encoder': {
            'model_name': '/media/ps/data-ssd/UltrasoundRAG/CLIP/checkpoints/BiomedBERT/BiomedNLP-BiomedBERT-base-uncased-abstract'
        },
        'embed_dim': 768
    }

    # 初始化模型
    model = QwenBERTModel(config)

    # 创建测试数据
    images = torch.randn(2, 3, 224, 224)  # 假设输入图像大小为224x224
    text_inputs = {
        'input_ids': torch.randint(0, 30522, (2, 128)),  # 假设文本长度为128
        'attention_mask': torch.ones((2, 128))
    }

    # 前向传播
    image_features, text_features = model(images, text_inputs)

    print("图像特征:", image_features)
    print("文本特征:", text_features)

if __name__ == "__main__":
    test_qwen_bert()
