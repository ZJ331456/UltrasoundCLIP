#!/usr/bin/env python3
"""
SAM+BERT微调模型嵌入可视化脚本
使用微调后的模型权重进行可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn.functional as F
import json
from PIL import Image
import torchvision.transforms as T
import os
from tqdm import tqdm
import glob
import re
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append('/home/zhoujun/NormalUltraCLIP')

# 导入现有的通用训练设置
from utils import setup_universal_training, ConfigManager
import matplotlib

# 使用标准英文字体
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class FinetunedSAMBERTModel:
    """微调后SAM+BERT模型包装器"""
    
    def __init__(self, config_path, checkpoint_path):
        print("正在加载微调后的SAM+BERT模型...")
        
        # 使用通用训练设置函数加载基础模型
        setup_result = setup_universal_training(config_path, model_path=None)
        
        self.config_manager = setup_result['config_manager']
        self.config = self.config_manager.config
        self.model = setup_result['model']
        self.tokenizer = setup_result['tokenizer']
        
        # 加载微调后的权重
        self.load_checkpoint(checkpoint_path)
        
        print(f"微调SAM+BERT模型加载完成:")
        print(f"  - 配置文件: {config_path}")
        print(f"  - 权重文件: {checkpoint_path}")
        print(f"  - 嵌入维度: {self.config['model']['embed_dim']}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                
                # 检查模型是否有Lightning包装
                if hasattr(self.model, 'model'):
                    # Lightning模型，需要加载到内部model
                    self.model.model.load_state_dict(state_dict, strict=False)
                else:
                    # 直接模型
                    self.model.load_state_dict(state_dict, strict=False)
                
                print(f"成功加载模型权重，epoch: {checkpoint.get('epoch', 'N/A')}")
                print(f"验证损失: {checkpoint.get('val_loss', 'N/A')}")
                print(f"验证准确率: {checkpoint.get('val_acc', 'N/A')}")
            else:
                print("Warning: No model_state_dict found in checkpoint, trying to load directly")
                if hasattr(self.model, 'model'):
                    self.model.model.load_state_dict(checkpoint, strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Continuing with pretrained weights...")
    
    def encode_image(self, images):
        """编码图像"""
        # 检查模型是否有Lightning包装
        if hasattr(self.model, 'model'):
            # Lightning模型
            return self.model.model.encode_image(images)
        else:
            # 直接模型
            return self.model.encode_image(images)
    
    def encode_text(self, text_tokens):
        """编码文本"""
        # 检查模型是否有Lightning包装
        if hasattr(self.model, 'model'):
            # Lightning模型
            return self.model.model.encode_text(text_tokens)
        else:
            # 直接模型
            return self.model.encode_text(text_tokens)
    
    def to(self, device):
        """移动模型到指定设备"""
        self.model = self.model.to(device)
        return self
    
    def eval(self):
        """设置为评估模式"""
        self.model.eval()
        return self

class UltrasoundDataset(Dataset):
    """超声数据集类"""
    def __init__(self, image_paths, texts, labels, transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 判断路径是否有效
        if not img_path or not os.path.isfile(img_path):
            # 用黑图像占位
            if self.transform:
                image = self.transform(Image.new('RGB', (1024, 1024), color='black'))
            else:
                image = torch.zeros(3, 1024, 1024)
            return {
                'image': image,
                'text': self.texts[idx],
                'label': self.labels[idx],
                'idx': idx
            }
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return {
                'image': image,
                'text': self.texts[idx],
                'label': self.labels[idx],
                'idx': idx
            }
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            if self.transform:
                dummy_image = self.transform(Image.new('RGB', (1024, 1024), color='black'))
            else:
                dummy_image = torch.zeros(3, 1024, 1024)
            return {
                'image': dummy_image,
                'text': self.texts[idx],
                'label': self.labels[idx],
                'idx': idx
            }

def load_json_files_parallel(data_dir, max_samples_per_file, max_workers):
    """
    遍历 data_dir 下所有 json 文件，包括相同编号的 train/val/test 文件
    """
    # 只遍历 data_dir 下的所有 json 文件（不递归子目录）
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    json_files.sort()
    
    print(f"Found {len(json_files)} JSON files")
    
    all_image_paths = []
    all_texts = []
    all_labels = []
    file_info = {}
    
    def process_json_file(json_file):
        filename = os.path.basename(json_file)
        match = re.match(r'^(\d+)_', filename)
        file_number = int(match.group(1)) if match else len(file_info)
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            image_paths = []
            texts = []
            labels = []
            
            data_info = data.get('DataInfo', {})
            items = list(data_info.items())
            
            # 统计有效样本 - 增加调试信息
            valid_items = []
            invalid_count = 0
            for sample_id, sample_data in items:
                img_path = sample_data.get('data_path')
                caption = sample_data.get('caption', sample_data.get('refined_caption', ''))
                
                # 详细检查每个条件
                if not img_path:
                    invalid_count += 1
                    continue
                if not caption:
                    invalid_count += 1
                    continue
                if not os.path.exists(img_path):
                    invalid_count += 1
                    print(f"  {filename}: 图片不存在 {img_path}")
                    continue
                    
                valid_items.append((sample_id, sample_data))
            
            print(f"{filename}: 总样本数 {len(items)}, 有效样本数 {len(valid_items)}, 无效样本数 {invalid_count}")
            
            # 统一采样
            if len(valid_items) > max_samples_per_file:
                indices = np.random.choice(len(valid_items), max_samples_per_file, replace=False)
                valid_items = [valid_items[i] for i in indices]
            
            for sample_id, sample_data in valid_items:
                img_path = sample_data.get('data_path')
                caption = sample_data.get('caption', sample_data.get('refined_caption', ''))
                image_paths.append(img_path)
                texts.append(caption)
                labels.append(file_number)
            
            # 只有有效样本才补齐，无效文件不补齐
            if len(valid_items) > 0 and len(valid_items) < max_samples_per_file:
                for _ in range(max_samples_per_file - len(valid_items)):
                    # 用空白图片和空caption补齐
                    image_paths.append('')
                    texts.append('')
                    labels.append(file_number)
            
            return {
                'file_number': file_number,
                'filename': filename,
                'image_paths': image_paths,
                'texts': texts,
                'labels': labels,
                'sample_count': len(image_paths),
                'valid_samples': len(valid_items)
            }
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_json_file, json_file): json_file 
                         for json_file in json_files}
        
        for future in tqdm(as_completed(future_to_file), total=len(json_files), desc="Loading JSON files"):
            result = future.result()
            if result:
                all_image_paths.extend(result['image_paths'])
                all_texts.extend(result['texts'])
                all_labels.extend(result['labels'])
                
                file_info[result['file_number']] = {
                    'filename': result['filename'],
                    'sample_count': result['sample_count'],
                    'total_samples': result['sample_count']
                }
                
                print(f"Loaded {result['sample_count']} samples from {result['filename']}")
    
    print(f"Total loaded {len(all_image_paths)} samples from {len(file_info)} files")
    
    return all_image_paths, all_texts, all_labels, file_info

def extract_embeddings_multi_gpu(model, tokenizer, dataset, device_ids=[0], batch_size=16):
    """多GPU并行提取嵌入特征"""
    if len(device_ids) > 1:
        device = f'cuda:{device_ids[0]}'
        print("注意：SAM+BERT模型暂不支持DataParallel，使用单GPU")
    else:
        device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    model.eval()
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    image_embeddings = []
    text_embeddings = []
    
    print(f"使用设备: {device}")
    print(f"开始提取嵌入特征, 批次大小: {batch_size}")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # 图像嵌入
            batch_images = batch['image'].to(device, non_blocking=True)
            img_feats = model.encode_image(batch_images)
            img_feats = F.normalize(img_feats, dim=-1)
            image_embeddings.append(img_feats.cpu())
            
            # 文本嵌入 - 使用模型内置的tokenizer
            batch_texts = batch['text']
            text_inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=256,
                return_tensors='pt'
            ).to(device, non_blocking=True)
            
            text_feats = model.encode_text(text_inputs)
            text_feats = F.normalize(text_feats, dim=-1)
            text_embeddings.append(text_feats.cpu())
    
    image_embeddings = torch.cat(image_embeddings, dim=0).numpy()
    text_embeddings = torch.cat(text_embeddings, dim=0).numpy()
    
    return image_embeddings, text_embeddings

def visualize_embeddings_svd(embeddings, labels, file_info, title, save_path):
    """使用SVD可视化嵌入"""
    svd = TruncatedSVD(n_components=2, random_state=42)
    embeddings_2d = svd.fit_transform(embeddings)
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    plt.figure(figsize=(15, 10))

    is_joint = 'Joint' in title

    if is_joint:
        n = embeddings_2d.shape[0] // 2
        for i, label in enumerate(unique_labels):
            mask_img = (labels[:n] == label)
            mask_txt = (labels[n:] == label)
            plt.scatter(embeddings_2d[:n][mask_img, 0], embeddings_2d[:n][mask_img, 1],
                        c=[colors[i]], marker='o', 
                        label=f'{file_info.get(label, {}).get("filename", f"File_{label}")} [Images]', 
                        s=30, alpha=0.7)
            plt.scatter(embeddings_2d[n:][mask_txt, 0], embeddings_2d[n:][mask_txt, 1],
                        c=[colors[i]], marker='^', 
                        label=f'{file_info.get(label, {}).get("filename", f"File_{label}")} [Text]', 
                        s=30, alpha=0.7)
    else:
        marker = 'o' if 'Image' in title or 'Images' in title else '^'
        for i, label in enumerate(unique_labels):
            mask = labels == label
            filename = file_info.get(label, {}).get('filename', f'File_{label}')
            sample_count = file_info.get(label, {}).get('sample_count', 0)
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        c=[colors[i]], marker=marker, 
                        label=f'{filename} (n={sample_count})', s=30, alpha=0.7)

    plt.title(f"SVD Visualization of {title}")
    plt.xlabel(f"Component 1 (Explained Variance: {svd.explained_variance_ratio_[0]:.3f})")
    plt.ylabel(f"Component 2 (Explained Variance: {svd.explained_variance_ratio_[1]:.3f})")
    
    if len(unique_labels) * (2 if is_joint else 1) <= 20:
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Image saved to: {save_path}")
    plt.close()
    return embeddings_2d

def main():
    """主函数"""
    # 设置路径
    config_path = "/home/zhoujun/NormalUltraCLIP/config/ultrasam/universal_sam_bert.json"
    checkpoint_path = "/home/zhoujun/NormalUltraCLIP/output/ultrasam_enhanced_clip_20epoch/best.pt"
    data_dir = "/media/ps/data-ssd/UltrasoundRAG/clip_caption/data-anatomy/更新路径之后的json数据集/split_with_llm_enhance_refined"
    
    # 检查文件是否存在
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # 检查可用GPU
    device_count = torch.cuda.device_count()
    print(f"Available GPUs: {device_count}")
    
    if device_count >= 1:
        device_ids = [0]
        batch_size = 2  # SAM模型很大，使用很小的批次
    else:
        print("Warning: No GPU detected, will use CPU (slow)")
        device_ids = []
        batch_size = 1
    
    print("Loading finetuned SAM+BERT model...")
    try:
        model = FinetunedSAMBERTModel(config_path, checkpoint_path)
        print("Finetuned SAM+BERT model loaded successfully!")
        
    except Exception as e:
        print(f"Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 设置图像变换 (SAM使用1024x1024)
    transform = T.Compose([
        T.Resize((1024, 1024), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
    ])
    
    print("Loading data in parallel...")
    try:
        # 只需设置max_samples_per_file即可
        image_paths, texts, labels, file_info = load_json_files_parallel(
            data_dir, max_samples_per_file=50, max_workers=10
        )
        print(f"Successfully loaded {len(image_paths)} sample paths")
        print(f"From {len(file_info)} files")
        
        # 移除随机选择逻辑 - 使用所有数据进行可视化
        print(f"使用全部 {len(image_paths)} 个样本进行可视化")
            
    except Exception as e:
        print(f"Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建数据集
    dataset = UltrasoundDataset(image_paths, texts, labels, transform)
    
    print("Extracting embeddings...")
    try:
        image_embeddings, text_embeddings = extract_embeddings_multi_gpu(
            model, model.tokenizer, dataset, device_ids, batch_size
        )
        
        print(f"Image embeddings shape: {image_embeddings.shape}")
        print(f"Text embeddings shape: {text_embeddings.shape}")
    except Exception as e:
        print(f"Embedding extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建保存目录
    save_dir = "/home/zhoujun/NormalUltraCLIP/show_embedding/output"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating SVD visualizations...")
    
    # 可视化图像嵌入
    img_2d = visualize_embeddings_svd(
        image_embeddings, np.array(labels), file_info,
        "Finetuned SAM Image Embeddings by Dataset", 
        os.path.join(save_dir, "sam_bert_finetuned_image_embeddings_svd.png")
    )
    
    # 可视化文本嵌入
    text_2d = visualize_embeddings_svd(
        text_embeddings, np.array(labels), file_info,
        "Finetuned BERT Text Embeddings by Dataset",
        os.path.join(save_dir, "sam_bert_finetuned_text_embeddings_svd.png")
    )
    
    # 可视化联合嵌入空间
    combined_embeddings = np.concatenate([image_embeddings, text_embeddings], axis=0)
    combined_labels = np.concatenate([labels, labels], axis=0)
    combined_file_info = {}
    for key, info in file_info.items():
        combined_file_info[key] = {
            'filename': info['filename'],
            'sample_count': info['sample_count'] * 2
        }
    
    combined_2d = visualize_embeddings_svd(
        combined_embeddings, combined_labels, combined_file_info,
        "Finetuned SAM+BERT Joint Embedding Space by Dataset",
        os.path.join(save_dir, "sam_bert_finetuned_joint_embeddings_svd.png")
    )
    
    # 保存数据集信息
    info_file = os.path.join(save_dir, "finetuned_dataset_info.txt")
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("Finetuned SAM+BERT Dataset Information Summary:\n")
        f.write("=" * 50 + "\n")
        total_samples = 0
        for file_num, info in sorted(file_info.items()):
            f.write(f"File No. {file_num}: {info['filename']}\n")
            f.write(f"  Sample count: {info['sample_count']}\n")
            f.write(f"  Total samples: {info['total_samples']}\n")
            f.write("-" * 30 + "\n")
            total_samples += info['sample_count']
        f.write(f"\nTotal: {len(file_info)} files, {total_samples} samples\n")
    
    print("Finetuned model visualization complete!")
    print(f"All images saved in: {save_dir}")
    print(f"Dataset info saved in: {info_file}")

if __name__ == "__main__":
    main()
    main()
