import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import json
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from PIL import Image
from UltrasoundRAG.CLIP.models.qwen_vl_clip_model import QwenVLCLIPModel
from UltrasoundRAG.CLIP.models.ultrasam_bert_pre_model import SAMBERTPreModel
from UltrasoundRAG.CLIP.dataload.transforms import TransformFactory

import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 加载数据集
def load_dataset(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            # 如果是list，直接返回
            if isinstance(data, list):
                return data
            # 如果是dict，包一层list返回
            elif isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            # 逐行读取，每行一个json对象
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]
            return data

# 提取图像特征

# 自定义Dataset
class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=588, is_sam=False):
        self.image_paths = image_paths
        # 根据是否使用SAM模型选择预处理方式
        if is_sam:
            self.transform = TransformFactory.get_sam_transforms(image_size=image_size)
        else:
            self.transform = TransformFactory.get_clip_transforms(image_size=image_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(image)

def extract_features(model, images, batch_size=8, num_workers=32, image_size=588, is_sam=False):
    dataset = ImagePathDataset(images, image_size=image_size, is_sam=is_sam)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    features = []
    # 自动检测模型设备
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for batch_tensor in tqdm(loader, desc='提取特征'):
        batch_tensor = batch_tensor.to(device)
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            with torch.amp.autocast(device_type='cuda'):
                batch_features = model.encode_image(batch_tensor)
        if hasattr(batch_features, 'detach'):
            batch_features = batch_features.detach().cpu().numpy()
        features.extend(batch_features)
    return features

# 保存特征到文件
def save_features(features, file_path):
    np.save(file_path, features)

# 加载特征从文件
def load_features(file_path):
    if os.path.exists(file_path):
        return np.load(file_path)
    return None

# 处理特征中的NaN值
def preprocess_features(features):
    features = np.nan_to_num(features, nan=0.0)  # 将NaN替换为0
    return features

# 可视化特征分布
def visualize_features(features1, features2, output_path):
    features1 = preprocess_features(features1)  # 预处理特征1
    features2 = preprocess_features(features2)  # 预处理特征2

    svd = TruncatedSVD(n_components=2)
    reduced_features1 = svd.fit_transform(features1)
    reduced_features2 = svd.fit_transform(features2)

    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_features1[:, 0], reduced_features1[:, 1], label='QwenVLCLIP', alpha=0.6, s=30, c='blue')
    plt.scatter(reduced_features2[:, 0], reduced_features2[:, 1], label='SAMBERT', alpha=0.6, s=30, c='orange')
    plt.legend(fontsize=12)
    plt.title('Feature Distribution Comparison (Optimized)', fontsize=16)
    plt.xlabel('SVD Component 1', fontsize=14)
    plt.ylabel('SVD Component 2', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    # 配置路径
    train_json = '/media/ps/data-ssd/UltrasoundRAG/clip_caption/caption_breast/caption_breast_train.json'
    output_path = '/media/ps/data-ssd/UltrasoundRAG/CLIP/show_embedding/output/visual_comparison.png'

    # 加载数据集
    dataset = load_dataset(train_json)
    # 适配DataInfo结构，提取图片路径和caption
    if isinstance(dataset, list) and len(dataset) == 1 and 'DataInfo' in dataset[0]:
        data_info = dataset[0]['DataInfo']
        image_paths = [item['data_path'] for item in data_info.values() if 'data_path' in item]
        captions = [item['CLIPcaption'] for item in data_info.values() if 'CLIPcaption' in item]
    elif isinstance(dataset, dict) and 'DataInfo' in dataset:
        data_info = dataset['DataInfo']
        image_paths = [item['data_path'] for item in data_info.values() if 'data_path' in item]
        captions = [item['CLIPcaption'] for item in data_info.values() if 'CLIPcaption' in item]
    else:
        image_paths = [item['image_path'] for item in dataset if 'image_path' in item]
        captions = [item.get('caption', '') for item in dataset]

    # 从外部json配置文件加载模型参数
    with open('/media/ps/data-ssd/UltrasoundRAG/CLIP/config/qwen/qwen_vl_clip_train_optimized.json', 'r') as f:
        # qwen_config = json.load(f)['model']
        qwen_config = json.load(f)
    qwen_model = QwenVLCLIPModel(qwen_config)

    with open('/media/ps/data-ssd/UltrasoundRAG/CLIP/config/ultrasam/sam_bert_train.json', 'r') as f:
        sam_config = json.load(f)['model']
    sam_model = SAMBERTPreModel(sam_config)

    # 提取特征（使用DataLoader加速）
    # Qwen模型用588，SAM模型用1024
    qwen_features_path = '/media/ps/data-ssd/UltrasoundRAG/CLIP/show_embedding/features/qwen_features.npy'
    sam_features_path = '/media/ps/data-ssd/UltrasoundRAG/CLIP/show_embedding/features/sam_features.npy'

    qwen_features = load_features(qwen_features_path)
    if qwen_features is None:
        qwen_features = extract_features(qwen_model, image_paths, image_size=588, is_sam=False)
        save_features(qwen_features, qwen_features_path)

    sam_features = load_features(sam_features_path)
    if sam_features is None:
        sam_features = extract_features(sam_model, image_paths, image_size=1024, is_sam=True)
        save_features(sam_features, sam_features_path)

    # 可视化
    visualize_features(qwen_features, sam_features, output_path)

    print(f'可视化结果已保存到 {output_path}')
