#!/usr/bin/env python3
"""
简化的FetalCLIP嵌入可视化脚本
直接使用OpenCLIP加载FetalCLIP权重并进行SVD可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn.functional as F
import open_clip
import json
from PIL import Image
import torchvision.transforms as T
import os
from tqdm import tqdm
import glob
import re
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 设置字体为英文字体
import matplotlib
import matplotlib.pyplot as plt

# 使用标准英文字体
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
def load_fetalclip_model(model_path, config_path):
    """
    加载FetalCLIP模型
    """
    # 读取配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 创建OpenCLIP模型
    arch_name = 'ViT-L-14'  # FetalCLIP使用的架构
    
    # 注册FetalCLIP配置
    open_clip.factory._MODEL_CONFIGS[arch_name] = config
    
    # 创建模型
    model, _, preprocess = open_clip.create_model_and_transforms(arch_name)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 移除module.前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    # 自动插值 visual.positional_embedding
    if 'visual.positional_embedding' in new_state_dict:
        pretrained_pe = new_state_dict['visual.positional_embedding']
        model_pe = model.visual.positional_embedding
        if pretrained_pe.shape != model_pe.shape:
            print(f"Auto interpolating visual.positional_embedding: {pretrained_pe.shape} -> {model_pe.shape}")
            # 只插值空间部分（去掉class token）
            num_cls_token = 1
            pretrained_grid = pretrained_pe[num_cls_token:]
            model_grid = model_pe[num_cls_token:]
            # 计算原始和目标网格尺寸
            N_pre = pretrained_grid.shape[0]
            N_model = model_grid.shape[0]
            D = pretrained_grid.shape[1]
            # 计算原始和目标网格边长
            S_pre = int(N_pre ** 0.5)
            S_model = int(N_model ** 0.5)
            if S_pre * S_pre != N_pre or S_model * S_model != N_model:
                print("Warning: Positional embedding grid is not square, skipping interpolation!")
            else:
                # 插值到目标尺寸
                pretrained_grid_reshape = pretrained_grid.reshape(S_pre, S_pre, D).permute(2, 0, 1).unsqueeze(0)  # [1, D, S_pre, S_pre]
                model_grid_reshape = torch.nn.functional.interpolate(
                    pretrained_grid_reshape,
                    size=(S_model, S_model),
                    mode='bicubic', align_corners=False
                ).squeeze(0).permute(1, 2, 0).reshape(N_model, D)
                # 拼回class token
                new_pe = torch.cat([pretrained_pe[:num_cls_token], model_grid_reshape], dim=0)
                new_state_dict['visual.positional_embedding'] = new_pe

    # 加载权重到模型
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 获取tokenizer
    tokenizer = open_clip.get_tokenizer(arch_name)

    return model, tokenizer, preprocess

class UltrasoundDataset(Dataset):
    """
    超声数据集类，用于并行数据加载
    """
    def __init__(self, image_paths, texts, labels, transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # 加载图像
            img_path = self.image_paths[idx]
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
            print(f"Failed to load image {self.image_paths[idx]}: {e}")
            # 返回一个零图像作为占位符
            if self.transform:
                dummy_image = self.transform(Image.new('RGB', (224, 224), color='black'))
            else:
                dummy_image = torch.zeros(3, 224, 224)
            return {
                'image': dummy_image,
                'text': self.texts[idx],
                'label': self.labels[idx],
                'idx': idx
            }

def load_json_files_parallel(data_dir, max_samples_per_file, max_workers):
    """
    并行加载JSON文件数据，遍历 data_dir 下所有以编号开头的 json 文件（如 03_fetal_dataset_train/val/test.json）
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
        """处理单个JSON文件"""
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
            
            # 随机采样以限制数量
            if len(items) > max_samples_per_file:
                indices = np.random.choice(len(items), max_samples_per_file, replace=False)
                items = [items[i] for i in indices]
            
            for sample_id, sample_data in items:
                img_path = sample_data.get('data_path')
                caption = sample_data.get('refined_caption')
                
                if img_path and caption and os.path.exists(img_path):
                    image_paths.append(img_path)
                    texts.append(caption)
                    labels.append(file_number)
            
            return {
                'file_number': file_number,
                'filename': filename,
                'image_paths': image_paths,
                'texts': texts,
                'labels': labels,
                'sample_count': len(image_paths)
            }
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            return None
    
    # 并行处理JSON文件
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

def extract_embeddings_multi_gpu(model, tokenizer, dataset, device_ids=[0, 1], batch_size=32):
    """
    多GPU并行提取嵌入特征
    """
    if len(device_ids) > 1:
        # 使用DataParallel进行多GPU并行
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        device = f'cuda:{device_ids[0]}'
        # 对于DataParallel，需要使用.module访问原始方法
        encode_image_func = model.module.encode_image
        encode_text_func = model.module.encode_text
    else:
        device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'
        encode_image_func = model.encode_image
        encode_text_func = model.encode_text
    
    model = model.to(device)
    model.eval()
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8,  # 增加worker数量
        pin_memory=True,
        prefetch_factor=2
    )
    
    image_embeddings = []
    text_embeddings = []
    
    print(f"Using devices: {device_ids}")
    print(f"Starting embedding extraction, batch size: {batch_size}")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # 图像嵌入
            batch_images = batch['image'].to(device, non_blocking=True)
            img_feats = encode_image_func(batch_images)
            img_feats = F.normalize(img_feats, dim=-1)
            image_embeddings.append(img_feats.cpu())
            
            # 文本嵌入
            batch_texts = batch['text']
            text_tokens = tokenizer(batch_texts).to(device, non_blocking=True)
            text_feats = encode_text_func(text_tokens)
            text_feats = F.normalize(text_feats, dim=-1)
            text_embeddings.append(text_feats.cpu())
    
    # 拼接所有嵌入
    image_embeddings = torch.cat(image_embeddings, dim=0).numpy()
    text_embeddings = torch.cat(text_embeddings, dim=0).numpy()
    
    return image_embeddings, text_embeddings

def visualize_embeddings_svd(embeddings, labels, file_info, title, save_path):
    """
    使用SVD可视化嵌入，根据文件序号使用不同颜色
    """
    # SVD降维到2D
    svd = TruncatedSVD(n_components=2, random_state=42)
    embeddings_2d = svd.fit_transform(embeddings)
    
    # 生成颜色映射
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    plt.figure(figsize=(15, 10))

    # 判断是否是联合空间（title里有Joint）
    is_joint = 'Joint' in title

    if is_joint:
        # 假设前半部分是图片，后半部分是文本
        n = embeddings_2d.shape[0] // 2
        for i, label in enumerate(unique_labels):
            mask_img = (labels[:n] == label)
            mask_txt = (labels[n:] == label)
            # 图片用圆点
            plt.scatter(embeddings_2d[:n][mask_img, 0], embeddings_2d[:n][mask_img, 1],
                        c=[colors[i]], marker='o', label=f'{file_info.get(label, {}).get("filename", f"File_{label}")} [Images]', s=30, alpha=0.7)
            # 文本用三角
            plt.scatter(embeddings_2d[n:][mask_txt, 0], embeddings_2d[n:][mask_txt, 1],
                        c=[colors[i]], marker='^', label=f'{file_info.get(label, {}).get("filename", f"File_{label}")} [Text]', s=30, alpha=0.7)
    else:
        # 非联合空间，marker根据title区分
        marker = 'o' if 'Image' in title or 'Images' in title else '^'
        for i, label in enumerate(unique_labels):
            mask = labels == label
            filename = file_info.get(label, {}).get('filename', f'File_{label}')
            sample_count = file_info.get(label, {}).get('sample_count', 0)
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        c=[colors[i]], marker=marker, label=f'{filename} (n={sample_count})', s=30, alpha=0.7)

    plt.title(f"SVD Visualization of {title}")
    
    # 设置坐标轴标签为英文
    plt.xlabel(f"Component 1 (Explained Variance: {svd.explained_variance_ratio_[0]:.3f})")
    plt.ylabel(f"Component 2 (Explained Variance: {svd.explained_variance_ratio_[1]:.3f})")
    
    # 图例使用默认字体
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
    """
    主函数 - 多GPU优化版本
    """
    # 设置路径
    model_path = "/media/ps/data-ssd/UltrasoundRAG/clip_caption/data-anatomy/redefined_caption_model_output/final_model.pt"
    # model_path = "/home/zhoujun/NormalUltraCLIP/checkpoints/FetalCLIP_Pretrain/FetalCLIP_weights.pt"
    config_path = "/home/zhoujun/NormalUltraCLIP/config/fetal/FetalCLIP_config.json"
    data_dir = "/media/ps/data-ssd/UltrasoundRAG/clip_caption/data-anatomy/更新路径之后的json数据集/split_with_llm_enhance_refined"
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # 检查可用GPU
    device_count = torch.cuda.device_count()
    print(f"Available GPUs: {device_count}")
    
    if device_count >= 2:
        device_ids = [0, 1]  # 使用前两张GPU
        batch_size = 64  # 增大批次大小
    elif device_count >= 1:
        device_ids = [0]
        batch_size = 32
    else:
        print("Warning: No GPU detected, will use CPU (slow)")
        device_ids = []
        batch_size = 8
    
    print("Loading FetalCLIP model...")
    try:
        model, tokenizer, preprocess = load_fetalclip_model(model_path, config_path)
        print("FetalCLIP model loaded successfully!")
    except Exception as e:
        print(f"Model loading failed: {e}")
        return
    
    # 设置图像变换
    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                   std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    print("Loading data in parallel...")
    try:
        image_paths, texts, labels, file_info = load_json_files_parallel(
            data_dir, max_samples_per_file=10, max_workers=8
        )
        print(f"Successfully loaded {len(image_paths)} sample paths")
        print(f"From {len(file_info)} files")
        
        # 移除随机选择逻辑 - 使用所有数据进行可视化
        print(f"使用全部 {len(image_paths)} 个样本进行可视化")
        
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
    
    # 创建数据集
    dataset = UltrasoundDataset(image_paths, texts, labels, transform)
    
    print("Extracting embeddings...")
    try:
        if device_ids:
            image_embeddings, text_embeddings = extract_embeddings_multi_gpu(
                model, tokenizer, dataset, device_ids, batch_size
            )
        else:
            # CPU fallback
            image_embeddings, text_embeddings = extract_embeddings_cpu(
                model, tokenizer, dataset, batch_size
            )
        
        print(f"Image embeddings shape: {image_embeddings.shape}")
        print(f"Text embeddings shape: {text_embeddings.shape}")
    except Exception as e:
        print(f"Embedding extraction failed: {e}")
        return
    
    # 创建保存目录
    save_dir = "/home/zhoujun/NormalUltraCLIP/show_embedding/output"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating SVD visualizations...")
    
    # 可视化图像嵌入
    img_2d = visualize_embeddings_svd(
        image_embeddings, np.array(labels), file_info,
        "FetalCLIP Image Embeddings by Dataset", 
        os.path.join(save_dir, "fetalclip_multi_dataset_image_embeddings_svd.png")
    )
    
    # 可视化文本嵌入
    text_2d = visualize_embeddings_svd(
        text_embeddings, np.array(labels), file_info,
        "FetalCLIP Text Embeddings by Dataset",
        os.path.join(save_dir, "fetalclip_multi_dataset_text_embeddings_svd.png")
    )
    
    # 可视化联合嵌入空间
    combined_embeddings = np.concatenate([image_embeddings, text_embeddings], axis=0)
    combined_labels = np.concatenate([labels, labels], axis=0)  # 保持相同的标签
    combined_file_info = {}
    for key, info in file_info.items():
        combined_file_info[key] = {
            'filename': info['filename'],
            'sample_count': info['sample_count'] * 2  # 图像+文本
        }
    
    combined_2d = visualize_embeddings_svd(
        combined_embeddings, combined_labels, combined_file_info,
        "FetalCLIP Joint Embedding Space by Dataset",
        os.path.join(save_dir, "fetalclip_multi_dataset_joint_embeddings_svd.png")
    )
    
    # 保存数据集信息
    info_file = os.path.join(save_dir, "dataset_info.txt")
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("Dataset Information Summary:\n")
        f.write("=" * 50 + "\n")
        total_samples = 0
        for file_num, info in sorted(file_info.items()):
            f.write(f"File No. {file_num}: {info['filename']}\n")
            f.write(f"  Sample count: {info['sample_count']}\n")
            f.write(f"  Total samples: {info['total_samples']}\n")
            f.write("-" * 30 + "\n")
            total_samples += info['sample_count']
        f.write(f"\nTotal: {len(file_info)} files, {total_samples} samples\n")
    
    print("Visualization complete!")
    print(f"All images saved in: {save_dir}")
    print(f"Dataset info saved in: {info_file}")

def extract_embeddings_cpu(model, tokenizer, dataset, batch_size=8):
    """
    CPU版本的嵌入提取（备用）
    """
    model = model.to('cpu')
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    image_embeddings = []
    text_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings (CPU)"):
            # 图像嵌入
            batch_images = batch['image']
            img_feats = model.encode_image(batch_images)
            img_feats = F.normalize(img_feats, dim=-1)
            image_embeddings.append(img_feats)
            
            # 文本嵌入
            batch_texts = batch['text']
            text_tokens = tokenizer(batch_texts)
            text_feats = model.encode_text(text_tokens)
            text_feats = F.normalize(text_feats, dim=-1)
            text_embeddings.append(text_feats)
    
    image_embeddings = torch.cat(image_embeddings, dim=0).numpy()
    text_embeddings = torch.cat(text_embeddings, dim=0).numpy()
    
    return image_embeddings, text_embeddings

if __name__ == "__main__":
    main()
