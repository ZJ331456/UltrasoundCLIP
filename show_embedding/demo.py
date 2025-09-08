import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import torch
from torch.utils.data import DataLoader
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models import model_factory

# 配置参数（请根据实际情况修改）
model_type = "fetalclip"  # 或其它模型类型
model_path = "/media/ps/data-ssd/UltrasoundRAG/clip_caption/data-anatomy/redefined_caption_model_output/final_model.pt"
config_path = "/home/zhoujun/NormalUltraCLIP/config/fetal/FetalCLIP_config.json"
data_path = "/media/ps/data-ssd/UltrasoundRAG/clip_caption/data-anatomy/更新路径之后的json数据集/split_with_llm_enhance_refined"
output_path = "output.png"

# 加载模型和数据
model, tokenizer, transform = model_factory.load_model(model_type, model_path, config_path)
image_paths, texts, labels = model_factory.load_data(data_path)

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

# SVD可视化（以图像嵌入为例）
svd = TruncatedSVD(n_components=2, random_state=42)
embedding_2d = svd.fit_transform(image_embeddings)
plt.figure(figsize=(10, 8))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='Spectral', s=10)
plt.title("SVD visualization of image embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.savefig(output_path)
print(f"可视化已保存到 {output_path}")
