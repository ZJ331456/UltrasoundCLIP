import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import DataLoader
from models import model_factory
from PIL import Image

sys.path.append('/home/zhoujun/CLIP')

# 通用数据集类
class UltrasoundDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, texts, labels, transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text = self.texts[idx]
        label = self.labels[idx]
        return image, text, label

# SVD可视化函数
def visualize_embeddings_svd(embeddings, labels, title, save_path):
    svd = TruncatedSVD(n_components=2, random_state=42)
    embeddings_2d = svd.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='Spectral', s=10)
    plt.title(f"SVD Visualization of {title}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")

# 主函数
def main(model_type, model_path, config_path, data_path, output_dir):
    # 加载模型
    model, tokenizer, transform = model_factory.load_model(model_type, model_path, config_path)

    # 加载数据
    image_paths, texts, labels = model_factory.load_data(data_path)
    dataset = UltrasoundDataset(image_paths, texts, labels, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 提取嵌入
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    image_embeddings = []
    text_embeddings = []

    with torch.no_grad():
        for images, texts, _ in dataloader:
            images = images.to(device)
            text_tokens = tokenizer(texts).to(device)

            image_embeds = model.encode_image(images)
            text_embeds = model.encode_text(text_tokens)

            image_embeddings.append(image_embeds.cpu().numpy())
            text_embeddings.append(text_embeds.cpu().numpy())

    image_embeddings = np.concatenate(image_embeddings, axis=0)
    text_embeddings = np.concatenate(text_embeddings, axis=0)

    # 可视化
    os.makedirs(output_dir, exist_ok=True)
    visualize_embeddings_svd(image_embeddings, labels, f"Image Embeddings ({model_type})", os.path.join(output_dir, "image_embeddings_svd.png"))
    visualize_embeddings_svd(text_embeddings, labels, f"Text Embeddings ({model_type})", os.path.join(output_dir, "text_embeddings_svd.png"))

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python generate_svd_visualization.py <model_type> <model_path> <config_path> <data_path> <output_dir>")
        sys.exit(1)

    model_type = sys.argv[1]
    model_path = sys.argv[2]
    config_path = sys.argv[3]
    data_path = sys.argv[4]
    output_dir = sys.argv[5]

    main(model_type, model_path, config_path, data_path, output_dir)
