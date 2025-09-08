
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from pathlib import Path
from PIL import Image
from models.qwen_vl_clip_model import QwenVLCLIPModel
from config_manager import ConfigManager
import torch.nn.functional as F
def test_qwen_model():
    # 配置文件
    config_path = "/media/ps/data-ssd/UltrasoundRAG/CLIP/test/test-qwen.json"
    
    # 创建配置管理器实例
    config_manager = ConfigManager(config_path)
    config = config_manager.config  # 加载配置
    # 初始化模型
    model = QwenVLCLIPModel(config['model'],)

    # 测试图像路径
    image_path = "/media/ps/data/Datasets/ultrasound/bm_pre_json/21/21.Breast_Ultrasound_Segmentation_Dataset/img/case001749.png"
    text_input = "This is a malignant ultrasound image of breast with irregular shape, spiculated margins, hypoechoic echogenicity, and internal microcalcifications."

    # 编码图像
    image_features = model.encode_image(image_path, debug=True)
    print(f"Image Features: {image_features.shape}")

    # 编码文本
    text_features = model.encode_text(text_input)
    print(f"Text Features: {text_features.shape}")

    # 前向传播测试
    image_features, text_features = model.forward(image_path, text_input)
    print(f"Forward Image Features: {image_features.shape}")
    print(f"Forward Text Features: {text_features.shape}")
    # img_feat: [1, 512], txt_feat: [1, 512]
    sim = F.cosine_similarity(image_features, text_features, dim=1)   # 输出形状 [1]
    print(sim.item())   
if __name__ == "__main__":
    test_qwen_model()