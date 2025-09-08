
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CLIP.models.fetalclip_post_model import load_fetalclip_model
import torch

def test_fetalclip_model():
    # 示例配置
    weight_path = "/home/zhoujun/NormalUltraCLIP/output/fetalclip-train/final_model.pt"
    config_path = "/home/zhoujun/NormalUltraCLIP/config/fetal/FetalCLIP_config.json"
    model, tokenizer = load_fetalclip_model(weight_path, config_path, 'cuda:0')
    print("FetalCLIP模型加载成功！")
    
    # 测试模型
    import torchvision.transforms as T
    from PIL import Image
    
    # 测试图像编码
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                    std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # 创建测试数据
    dummy_image = torch.randn(1, 3, 224, 224).to('cuda:0')  # 移动到 GPU
    dummy_text = tokenizer(["测试文本"])
    dummy_text = torch.tensor(dummy_text).to('cuda:0')  # 转换为 Tensor 并移动到 GPU
    
    with torch.no_grad():
        image_features = model.encode_image(dummy_image)
        text_features = model.encode_text(dummy_text)
    
    print(f"图像特征: {image_features}")
    print(f"图像特征形状: {image_features.shape}")
    print(f"文本特征: {text_features}")
    print(f"文本特征形状: {text_features.shape}")

if __name__ == "__main__":
    test_fetalclip_model()