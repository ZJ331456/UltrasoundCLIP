import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.qwen_vl_clip_model import QwenVLCLIPModel
import json
# 加载配置文件
config_path = "/media/ps/data-ssd/UltrasoundRAG/CLIP/config/qwen/qwen_vl_clip_train_optimized.json"
with open(config_path, "r") as f:
    config = json.load(f)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QwenVLCLIPModel(config['model']).to(device)

# 加载权重
checkpoint_path = "/media/ps/data-ssd/UltrasoundRAG/CLIP/output/qwen_vl_clip_3_cliploss/checkpoint_epoch_10.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# 切换到评估模式
model.eval()

# 测试输入
test_images = ["/media/ps/data/Datasets/ultrasound/bm_pre_json/21/21.Breast_Ultrasound_Segmentation_Dataset/img/case001749.png"]  # 替换为实际测试图像路径
test_texts = ["This is a malignant ultrasound image of breast with irregular shape, spiculated margins, hypoechoic echogenicity, and internal microcalcifications."]  # 替换为实际测试文本

# 编码图像和文本
image_features = model.encode_image(test_images)
text_features = model.encode_text(test_texts)

# 计算相似度
similarity = torch.matmul(image_features, text_features.T)
print(f"图像与文本的相似度: {similarity}")