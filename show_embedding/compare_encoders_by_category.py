#!/usr/bin/env python3
"""
比较两个图像编码器在不同类型图片上的表现和聚合度分析
"""

import sys
import os
import json
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import glob
import re
import datetime
from matplotlib import font_manager, rcParams

# 添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from UltrasoundRAG.CLIP.models.qwen_vl_clip_model import QwenVLCLIPModel
from UltrasoundRAG.CLIP.dataload.transforms import TransformFactory


# 锁定到单卡（若外部未显式指定），避免占用全部GPU
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_chinese_font():
    """确保Matplotlib能正常显示中文，避免方框。"""
    try:
        rcParams['axes.unicode_minus'] = False
        # 优先使用常见中文字体名称
        preferred = ['Noto Sans CJK SC', 'SimHei', 'WenQuanYi Micro Hei', 'Microsoft YaHei', 'DejaVu Sans']
        rcParams['font.sans-serif'] = preferred

        # 常见中文字体文件路径（Linux常见发行版）
        candidate_paths = [
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/arphic/ukai.ttc',
            '/usr/share/fonts/truetype/arphic/uming.ttc',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/msyh.ttf',
            '/usr/share/fonts/truetype/simhei.ttf',
        ]
        added_any = False
        for p in candidate_paths:
            if os.path.exists(p):
                try:
                    font_manager.fontManager.addfont(p)
                    name = font_manager.FontProperties(fname=p).get_name()
                    # 将新字体放在首位以优先使用
                    current = list(rcParams.get('font.sans-serif', []))
                    if name not in current:
                        rcParams['font.sans-serif'] = [name] + current
                    added_any = True
                except Exception:
                    continue
        # 如果没有任何中文字体可用，至少保底DejaVu Sans（能显示部分符号）
        if not added_any:
            rcParams['font.family'] = 'sans-serif'
    except Exception:
        pass

class CategoryImageDataset(Dataset):
    """按类别组织的图像数据集"""
    
    def __init__(self, image_paths, categories, image_size=588, is_sam=False):
        self.image_paths = image_paths
        self.categories = categories
        
        # 根据是否使用SAM模型选择预处理方式
        if is_sam:
            self.transform = TransformFactory.get_sam_transforms(image_size=image_size)
        else:
            self.transform = TransformFactory.get_clip_transforms(image_size=image_size)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(image), self.categories[idx]


def load_category_datasets(base_path, max_samples_per_file=500):
    """
    加载按类别组织的数据集
    Args:
        base_path: 基础路径
        max_samples_per_file: 每个文件最大样本数
    Returns:
        image_paths: 图像路径列表
        categories: 类别列表
        category_info: 类别信息字典
    """
    print(f"正在加载数据集从: {base_path}")
    
    # 查找所有json文件
    json_files = glob.glob(os.path.join(base_path, "**/*.json"), recursive=True)
    print(f"找到 {len(json_files)} 个JSON文件")
    
    image_paths = []
    categories = []
    category_info = defaultdict(list)
    
    # 按文件名排序，确保相同序号的在一起
    json_files.sort()
    
    for json_file in tqdm(json_files, desc="加载JSON文件"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取文件名作为类别标识
            file_name = os.path.basename(json_file)
            category = os.path.splitext(file_name)[0]
            
            # 解析数据
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and 'DataInfo' in data:
                items = list(data['DataInfo'].values())
            else:
                items = [data]
            
            # 限制每个文件的样本数量
            items = items[:max_samples_per_file]
            
            for item in items:
                if 'data_path' in item:
                    image_path = item['data_path']
                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                        categories.append(category)
                        category_info[category].append({
                            'path': image_path,
                            'caption': item['refined_caption'],
                            'file': json_file
                        })
            
            print(f"  {category}: {len(category_info[category])} 个样本")
            
        except Exception as e:
            print(f"警告: 无法加载文件 {json_file}: {e}")
            continue
    
    print(f"\n总共加载了 {len(image_paths)} 个图像样本")
    print(f"类别数量: {len(category_info)}")
    
    return image_paths, categories, category_info


def extract_features_by_category(model, image_paths, categories, batch_size=8, 
                               num_workers=4, image_size=588, is_sam=False):
    """
    按类别提取图像特征
    """
    dataset = CategoryImageDataset(image_paths, categories, image_size, is_sam)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                       pin_memory=True, shuffle=False)
    
    features = []
    extracted_categories = []
    
    # 自动检测模型设备
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for batch_tensor, batch_categories in tqdm(loader, desc='提取特征'):
        batch_tensor = batch_tensor.to(device)
        
        with torch.no_grad():
            # 强制使用float32，禁用autocast，避免半精度导致的下溢和全零
            feats = model.encode_image(batch_tensor.float())

        # 清理非有限值并避免全零向量
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        row_norm = feats.norm(dim=1, keepdim=True)
        zero_mask = row_norm.squeeze(1) == 0
        if zero_mask.any():
            # 注入极小扰动并归一，避免后续除零
            feats[zero_mask] = torch.randn_like(feats[zero_mask]) * 1e-6
            row_norm = feats.norm(dim=1, keepdim=True).clamp_min(1e-12)
        feats = feats / row_norm.clamp_min(1e-12)

        batch_features = feats.detach().cpu().numpy()
        
        features.extend(batch_features)
        extracted_categories.extend(batch_categories)
    
    features_array = np.array(features)
    
    # 最终检查
    if np.any(np.isnan(features_array)) or np.any(~np.isfinite(features_array)):
        print(f"警告: 特征数组存在无效值，已清理并归一...")
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(features_array, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        features_array = features_array / norms
    
    print(f"特征提取完成，形状: {features_array.shape}")
    print(f"特征值范围: [{features_array.min():.6f}, {features_array.max():.6f}]")
    print(f"特征均值: {features_array.mean():.6f}, 标准差: {features_array.std():.6f}")
    
    return features_array, extracted_categories


def analyze_clustering_quality(features, categories):
    """
    分析聚类质量
    """
    print("\n=== 聚类质量分析 ===")
    
    # 数据验证
    if np.any(np.isnan(features)):
        print("警告: 特征数据包含NaN值，正在清理...")
        features = np.nan_to_num(features, nan=0.0)
    
    if np.any(np.isinf(features)):
        print("警告: 特征数据包含无穷值，正在清理...")
        features = np.nan_to_num(features, posinf=1e6, neginf=-1e6)
    
    # 检查特征是否全为0
    if np.all(features == 0):
        print("错误: 所有特征值都为0，无法进行分析")
        return 0.0, {}
    
    # 标准化特征
    try:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 再次检查标准化后的数据
        if np.any(np.isnan(features_scaled)):
            print("警告: 标准化后仍有NaN值，使用原始特征...")
            features_scaled = features
    except Exception as e:
        print(f"标准化失败: {e}，使用原始特征...")
        features_scaled = features
    
    # 计算轮廓系数
    unique_categories = list(set(categories))
    if len(unique_categories) < 2:
        print("错误: 类别数量少于2，无法计算轮廓系数")
        return 0.0, {}
    
    category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    category_labels = [category_to_idx[cat] for cat in categories]
    
    try:
        silhouette_avg = silhouette_score(features_scaled, category_labels)
        print(f"整体轮廓系数: {silhouette_avg:.4f}")
    except Exception as e:
        print(f"计算整体轮廓系数失败: {e}")
        silhouette_avg = 0.0
    
    # 按类别计算轮廓系数
    category_silhouettes = {}
    for category in unique_categories:
        category_mask = [i for i, cat in enumerate(categories) if cat == category]
        if len(category_mask) > 1:
            category_features = features_scaled[category_mask]
            category_labels_subset = [category_to_idx[category]] * len(category_mask)
            try:
                sil_score = silhouette_score(category_features, category_labels_subset)
                category_silhouettes[category] = sil_score
            except Exception as e:
                print(f"类别 {category} 轮廓系数计算失败: {e}")
                category_silhouettes[category] = 0.0
        else:
            category_silhouettes[category] = 0.0
    
    # 排序并显示
    sorted_categories = sorted(category_silhouettes.items(), key=lambda x: x[1], reverse=True)
    print("\n各类别轮廓系数:")
    for category, score in sorted_categories[:10]:  # 显示前10个
        print(f"  {category}: {score:.4f}")
    
    return silhouette_avg, category_silhouettes


def visualize_feature_comparison(features1, features2, categories1, categories2, 
                                model1_name, model2_name, output_dir):
    """
    可视化两个模型的特征比较
    """
    print(f"\n=== 生成可视化图表 ===")
    
    # 数据验证和清理
    if np.any(np.isnan(features1)):
        print(f"警告: {model1_name} 特征包含NaN值，正在清理...")
        features1 = np.nan_to_num(features1, nan=0.0)
    
    if np.any(np.isnan(features2)):
        print(f"警告: {model2_name} 特征包含NaN值，正在清理...")
        features2 = np.nan_to_num(features2, nan=0.0)
    
    if np.any(np.isinf(features1)):
        print(f"警告: {model1_name} 特征包含无穷值，正在清理...")
        features1 = np.nan_to_num(features1, posinf=1e6, neginf=-1e6)
    
    if np.any(np.isinf(features2)):
        print(f"警告: {model2_name} 特征包含无穷值，正在清理...")
        features2 = np.nan_to_num(features2, posinf=1e6, neginf=-1e6)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 分别展示每个模型的分布 (PCA)
    print("生成各模型PCA分布图...")
    
    # 模型1的PCA
    try:
        pca1 = PCA(n_components=2)
        features1_pca = pca1.fit_transform(features1)
        print(f"{model1_name} PCA完成，解释方差比: {pca1.explained_variance_ratio_}")
    except Exception as e:
        print(f"{model1_name} PCA失败: {e}，使用原始特征的前两维...")
        features1_pca = features1[:, :2] if features1.shape[1] >= 2 else np.zeros((features1.shape[0], 2))
    
    # 模型2的PCA
    try:
        pca2 = PCA(n_components=2)
        features2_pca = pca2.fit_transform(features2)
        print(f"{model2_name} PCA完成，解释方差比: {pca2.explained_variance_ratio_}")
    except Exception as e:
        print(f"{model2_name} PCA失败: {e}，使用原始特征的前两维...")
        features2_pca = features2[:, :2] if features2.shape[1] >= 2 else np.zeros((features2.shape[0], 2))
    
    # 绘制各模型分布图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 获取所有类别并排序（按数字编号）
    all_categories = list(set(categories1 + categories2))
    # 按数字编号排序
    all_categories.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else float('inf'))
    
    # 为每个类别分配颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_categories)))
    category_colors = {cat: colors[i] for i, cat in enumerate(all_categories)}
    
    # 模型1分布
    for category in all_categories:
        mask = [cat == category for cat in categories1]
        if any(mask):
            ax1.scatter(features1_pca[mask, 0], features1_pca[mask, 1], 
                       label=f"{category}", alpha=0.7, s=30, c=[category_colors[category]])
    
    ax1.set_title(f'{model1_name} - 特征分布', fontsize=16)
    ax1.set_xlabel('PC1', fontsize=14)
    ax1.set_ylabel('PC2', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 模型2分布
    for category in all_categories:
        mask = [cat == category for cat in categories2]
        if any(mask):
            ax2.scatter(features2_pca[mask, 0], features2_pca[mask, 1], 
                       label=f"{category}", alpha=0.7, s=30, c=[category_colors[category]])
    
    ax2.set_title(f'{model2_name} - 特征分布', fontsize=16)
    ax2.set_xlabel('PC1', fontsize=14)
    ax2.set_ylabel('PC2', fontsize=14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'individual_model_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 在一张图上对比两个模型
    print("生成模型对比图...")
    plt.figure(figsize=(16, 12))
    
    # 使用不同符号区分模型
    markers = ['o', 's']  # 圆形和方形
    model_names = [model1_name, model2_name]
    all_features = [features1_pca, features2_pca]
    all_categories_list = [categories1, categories2]
    
    for model_idx, (features_pca, categories, model_name) in enumerate(zip(all_features, all_categories_list, model_names)):
        marker = markers[model_idx]
        
        for category in all_categories:
            mask = [cat == category for cat in categories]
            if any(mask):
                plt.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                           label=f"{model_name}-{category}", 
                           alpha=0.7, s=40, 
                           c=[category_colors[category]], 
                           marker=marker,
                           edgecolors='black', linewidth=0.5)
    
    plt.title('两个模型特征分布对比', fontsize=18)
    plt.xlabel('PC1', fontsize=16)
    plt.ylabel('PC2', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 添加图例说明
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label=f'{model1_name} (圆形)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label=f'{model2_name} (方形)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 类别内聚合度热力图
    print("生成聚合度热力图...")
    create_clustering_heatmap(features1, categories1, model1_name, 
                             os.path.join(output_dir, f'{model1_name}_clustering_heatmap.png'))
    create_clustering_heatmap(features2, categories2, model2_name, 
                             os.path.join(output_dir, f'{model2_name}_clustering_heatmap.png'))
    
    # 4. 特征相似度矩阵
    print("生成特征相似度矩阵...")
    create_similarity_matrix(features1, categories1, model1_name, 
                           os.path.join(output_dir, f'{model1_name}_similarity_matrix.png'))
    create_similarity_matrix(features2, categories2, model2_name, 
                           os.path.join(output_dir, f'{model2_name}_similarity_matrix.png'))
    
    # 5. 新增：模型性能对比柱状图
    print("生成模型性能对比图...")
    create_model_performance_comparison(
        features1, categories1, features2, categories2,
        model1_name, model2_name, all_categories,
        os.path.join(output_dir, 'model_performance_comparison.png')
    )
    
    print(f"所有可视化图表已保存到: {output_dir}")


def create_clustering_heatmap(features, categories, model_name, output_path):
    """
    创建聚类质量热力图
    """
    # 数据验证
    if np.any(np.isnan(features)):
        print(f"警告: {model_name} 热力图特征包含NaN值，正在清理...")
        features = np.nan_to_num(features, nan=0.0)
    
    if np.any(np.isinf(features)):
        print(f"警告: {model_name} 热力图特征包含无穷值，正在清理...")
        features = np.nan_to_num(features, posinf=1e6, neginf=-1e6)
    
    unique_categories = list(set(categories))
    category_scores = {}
    
    # 计算每个类别的聚类质量
    for category in unique_categories:
        category_mask = [i for i, cat in enumerate(categories) if cat == category]
        if len(category_mask) > 1:
            category_features = features[category_mask]
            
            # 计算类内平均距离
            distances = []
            for i in range(len(category_features)):
                for j in range(i+1, len(category_features)):
                    try:
                        dist = np.linalg.norm(category_features[i] - category_features[j])
                        if np.isfinite(dist):  # 检查距离是否为有限值
                            distances.append(dist)
                    except Exception as e:
                        print(f"警告: 计算类别 {category} 距离时出错: {e}")
                        continue
            
            if distances:
                category_scores[category] = np.mean(distances)
            else:
                category_scores[category] = 0.0
        else:
            category_scores[category] = 0.0
    
    # 创建热力图
    plt.figure(figsize=(12, 8))
    categories_list = list(category_scores.keys())
    scores_list = [category_scores[cat] for cat in categories_list]
    
    # 归一化分数
    scores_array = np.array(scores_list)
    if scores_array.max() > 0:
        scores_normalized = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min())
    else:
        scores_normalized = scores_array
    
    # 创建热力图
    heatmap_data = scores_normalized.reshape(1, -1)
    plt.imshow(heatmap_data, 
                cmap='YlOrRd',
                aspect='auto',
                extent=[0, len(categories_list), 0, 1]) # 设置extent以控制颜色条位置
    
    plt.title(f'{model_name} - 各类别聚合度热力图', fontsize=16)
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('模型', fontsize=12)
    plt.xticks(np.arange(len(categories_list)), categories_list, rotation=45, ha='right')
    plt.yticks([0], [f'{model_name}']) # 只显示一个y轴标签
    
    # 添加颜色条
    cbar = plt.colorbar(label='归一化聚合度分数')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_similarity_matrix(features, categories, model_name, output_path):
    """
    创建特征相似度矩阵
    """
    # 数据验证
    if np.any(np.isnan(features)):
        print(f"警告: {model_name} 相似度矩阵特征包含NaN值，正在清理...")
        features = np.nan_to_num(features, nan=0.0)
    
    if np.any(np.isinf(features)):
        print(f"警告: {model_name} 相似度矩阵特征包含无穷值，正在清理...")
        features = np.nan_to_num(features, posinf=1e6, neginf=-1e6)
    
    try:
        # 计算余弦相似度
        features_normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
        similarity_matrix = np.dot(features_normalized, features_normalized.T)
        
        # 检查相似度矩阵是否包含NaN或无穷值
        if np.any(np.isnan(similarity_matrix)) or np.any(np.isinf(similarity_matrix)):
            print(f"警告: {model_name} 相似度矩阵包含无效值，正在清理...")
            similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
            # 确保相似度在[-1, 1]范围内
            similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
    except Exception as e:
        print(f"警告: {model_name} 计算相似度矩阵失败: {e}，使用单位矩阵...")
        similarity_matrix = np.eye(len(features))
    
    # 创建类别标签
    unique_categories = list(set(categories))
    category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    category_labels = [category_to_idx[cat] for cat in categories]
    
    # 按类别排序
    sorted_indices = np.argsort(category_labels)
    sorted_similarity = similarity_matrix[sorted_indices][:, sorted_indices]
    sorted_categories = [categories[i] for i in sorted_indices]
    
    # 绘制相似度矩阵
    plt.figure(figsize=(12, 10))
    
    # 创建类别边界
    category_boundaries = []
    current_category = sorted_categories[0]
    for i, cat in enumerate(sorted_categories):
        if cat != current_category:
            category_boundaries.append(i)
            current_category = cat
    
    # 绘制热力图
    im = plt.imshow(sorted_similarity, cmap='viridis', aspect='auto')
    
    # 添加类别边界线
    for boundary in category_boundaries:
        plt.axhline(y=boundary, color='red', linewidth=2, alpha=0.7)
        plt.axvline(x=boundary, color='red', linewidth=2, alpha=0.7)
    
    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('余弦相似度', fontsize=12)
    
    plt.title(f'{model_name} - 特征相似度矩阵', fontsize=16)
    plt.xlabel('样本索引', fontsize=12)
    plt.ylabel('样本索引', fontsize=12)
    
    # 添加类别标签
    category_centers = []
    current_cat = sorted_categories[0]
    start_idx = 0
    for i, cat in enumerate(sorted_categories):
        if cat != current_cat:
            center = (start_idx + i - 1) / 2
            category_centers.append((center, current_cat))
            start_idx = i
            current_cat = cat
    
    # 最后一个类别
    center = (start_idx + len(sorted_categories) - 1) / 2
    category_centers.append((center, current_cat))
    
    # 在y轴上添加类别标签
    for center, cat in category_centers:
        plt.text(-len(sorted_categories)*0.05, center, cat, 
                rotation=0, ha='right', va='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_model_performance_comparison(features1, categories1, features2, categories2, 
                                      model1_name, model2_name, all_categories, output_path):
    """
    创建模型性能对比柱状图
    """
    # 数据验证
    if np.any(np.isnan(features1)):
        print(f"警告: {model1_name} 性能对比特征包含NaN值，正在清理...")
        features1 = np.nan_to_num(features1, nan=0.0)
    
    if np.any(np.isnan(features2)):
        print(f"警告: {model2_name} 性能对比特征包含NaN值，正在清理...")
        features2 = np.nan_to_num(features2, nan=0.0)
    
    if np.any(np.isinf(features1)):
        print(f"警告: {model1_name} 性能对比特征包含无穷值，正在清理...")
        features1 = np.nan_to_num(features1, posinf=1e6, neginf=-1e6)
    
    if np.any(np.isinf(features2)):
        print(f"警告: {model2_name} 性能对比特征包含无穷值，正在清理...")
        features2 = np.nan_to_num(features2, posinf=1e6, neginf=-1e6)
    
    # 计算每个类别在每个模型上的聚合度
    model1_scores = {}
    model2_scores = {}
    
    for category in all_categories:
        # 模型1的聚合度
        mask1 = [cat == category for cat in categories1]
        if any(mask1) and sum(mask1) > 1:
            category_features1 = features1[mask1]
            distances1 = []
            for i in range(len(category_features1)):
                for j in range(i+1, len(category_features1)):
                    try:
                        dist = np.linalg.norm(category_features1[i] - category_features1[j])
                        if np.isfinite(dist):  # 检查距离是否为有限值
                            distances1.append(dist)
                    except Exception as e:
                        print(f"警告: 计算{model1_name}类别{category}距离时出错: {e}")
                        continue
            model1_scores[category] = np.mean(distances1) if distances1 else 0.0
        else:
            model1_scores[category] = 0.0
        
        # 模型2的聚合度
        mask2 = [cat == category for cat in categories2]
        if any(mask2) and sum(mask2) > 1:
            category_features2 = features2[mask2]
            distances2 = []
            for i in range(len(category_features2)):
                for j in range(i+1, len(category_features2)):
                    try:
                        dist = np.linalg.norm(category_features2[i] - category_features2[j])
                        if np.isfinite(dist):  # 检查距离是否为有限值
                            distances2.append(dist)
                    except Exception as e:
                        print(f"警告: 计算{model2_name}类别{category}距离时出错: {e}")
                        continue
            model2_scores[category] = np.mean(distances2) if distances2 else 0.0
        else:
            model2_scores[category] = 0.0
    
    # 创建柱状图
    plt.figure(figsize=(16, 10))
    
    x = np.arange(len(all_categories))
    width = 0.35
    
    # 获取分数列表
    scores1 = [model1_scores[cat] for cat in all_categories]
    scores2 = [model2_scores[cat] for cat in all_categories]
    
    # 归一化分数（越小越好）
    max_score = max(max(scores1), max(scores2))
    if max_score > 0:
        normalized_scores1 = [1 - (s / max_score) for s in scores1]
        normalized_scores2 = [1 - (s / max_score) for s in scores2]
    else:
        normalized_scores1 = scores1
        normalized_scores2 = scores2
    
    # 绘制柱状图
    bars1 = plt.bar(x - width/2, normalized_scores1, width, label=model1_name, 
                    alpha=0.8, color='skyblue', edgecolor='navy', linewidth=1)
    bars2 = plt.bar(x + width/2, normalized_scores2, width, label=model2_name, 
                    alpha=0.8, color='lightcoral', edgecolor='darkred', linewidth=1)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('身体部位类别', fontsize=14)
    plt.ylabel('归一化聚合度分数 (越高越好)', fontsize=14)
    plt.title('两个模型在不同身体部位上的聚合度对比', fontsize=16)
    plt.xticks(x, all_categories, rotation=45, ha='right')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加性能总结
    avg_score1 = np.mean(normalized_scores1)
    avg_score2 = np.mean(normalized_scores2)
    plt.text(0.02, 0.98, f'{model1_name} 平均分数: {avg_score1:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    plt.text(0.02, 0.92, f'{model2_name} 平均分数: {avg_score2:.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印性能总结
    print(f"\n=== 模型性能总结 ===")
    print(f"{model1_name} 平均聚合度分数: {avg_score1:.4f}")
    print(f"{model2_name} 平均聚合度分数: {avg_score2:.4f}")
    
    if avg_score1 > avg_score2:
        print(f"{model1_name} 整体表现更好 (+{avg_score1 - avg_score2:.4f})")
    else:
        print(f"{model2_name} 整体表现更好 (+{avg_score2 - avg_score1:.4f})")
    
    # 按类别显示详细对比
    print(f"\n各部位详细对比:")
    for category in all_categories:
        score1 = normalized_scores1[all_categories.index(category)]
        score2 = normalized_scores2[all_categories.index(category)]
        diff = score1 - score2
        better_model = model1_name if diff > 0 else model2_name
        print(f"  {category}: {model1_name}({score1:.3f}) vs {model2_name}({score2:.3f}) - {better_model}更好 ({diff:+.3f})")


def main():
    """主函数"""
    print("=== 图像编码器比较分析 ===")
    # 确保中文正常显示
    setup_chinese_font()
    
    try:
        # 配置路径
        base_path = "/media/ps/data-ssd/UltrasoundRAG/clip_caption/data-anatomy/更新路径之后的json数据集/split_with_llm_enhance_refined"
        output_dir = "/media/ps/data-ssd/UltrasoundRAG/CLIP/show_embedding/output/category_analysis"
        
        # 加载数据集
        print("正在加载数据集...")
        image_paths, categories, category_info = load_category_datasets(base_path, max_samples_per_file=500)
        
        if not image_paths:
            print("错误: 没有找到有效的图像路径")
            return
        
        print(f"成功加载 {len(image_paths)} 个图像样本")
        
        # 加载模型配置
        print("\n=== 加载模型 ===")
        try:
            with open('/media/ps/data-ssd/UltrasoundRAG/CLIP/config/qwen/qwen_vl_clip_opt_4.json', 'r') as f:
                qwen_config = json.load(f)
        except Exception as e:
            print(f"错误: 无法读取Qwen配置: {e}")
            return

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def load_qwen_model(cfg, ckpt_path=None):
            model = QwenVLCLIPModel(cfg['model'])
            # 优先在目标设备上初始化空权重，避免meta迁移错误
            if hasattr(model, 'to_empty'):
                try:
                    model.to_empty(device=device)
                except TypeError:
                    model.to_empty()
            else:
                model = model.to(device)

            if ckpt_path:
                try:
                    ckpt = torch.load(ckpt_path, map_location='cpu')
                    state = ckpt.get('model_state_dict', ckpt)
                    try:
                        model.load_state_dict(state, strict=False, assign=True)
                    except TypeError:
                        model.load_state_dict(state, strict=False)
                    print("QwenVLCLIP 权重加载完成:", ckpt_path)
                except Exception as e:
                    raise RuntimeError(f"加载权重失败: {e}")

            # 确保在目标设备
            try:
                model = model.to(device)
            except NotImplementedError:
                pass
            model.eval()
            return model

        # 顺序加载以节省显存：先预训练 -> 提取 -> 释放；再微调 -> 提取
        try:
            qwen_pre_model = load_qwen_model(qwen_config)
            print("QwenVLCLIP 预训练模型加载完成")
        except Exception as e:
            print(f"错误: 无法创建Qwen预训练模型: {e}")
            return
        
        # 提取特征
        print("\n=== 提取特征 ===")
        # Qwen 预训练特征
        print("正在提取 QwenVLCLIP-Pretrained 特征...")
        try:
            qwen_pre_features, qwen_pre_categories = extract_features_by_category(
                qwen_pre_model, image_paths, categories,
                batch_size=8, num_workers=4, image_size=588, is_sam=False
            )
            print(f"QwenVLCLIP-Pretrained 特征提取完成，形状: {qwen_pre_features.shape}")
        except Exception as e:
            print(f"错误: QwenVLCLIP-Pretrained 特征提取失败: {e}")
            return

        # 释放预训练模型显存
        try:
            del qwen_pre_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Qwen 微调后特征
        print("正在提取 QwenVLCLIP-Finetuned 特征...")
        try:
            qwen_ft_model = load_qwen_model(
                qwen_config,
                ckpt_path='/media/ps/data-ssd/UltrasoundRAG/CLIP/output/qwen_vl_clip_6_ultrasound_loss/checkpoint_epoch_15.pth'
            )
            qwen_ft_features, qwen_ft_categories = extract_features_by_category(
                qwen_ft_model, image_paths, categories,
                batch_size=8, num_workers=4, image_size=588, is_sam=False
            )
            print(f"QwenVLCLIP-Finetuned 特征提取完成，形状: {qwen_ft_features.shape}")
        except Exception as e:
            print(f"错误: QwenVLCLIP-Finetuned 特征提取失败: {e}")
            return
        finally:
            try:
                del qwen_ft_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        print(f"Qwen-Pretrained 特征形状: {qwen_pre_features.shape}")
        print(f"Qwen-Finetuned 特征形状: {qwen_ft_features.shape}")
        
        # 验证特征数据
        if qwen_pre_features.size == 0 or qwen_ft_features.size == 0:
            print("错误: 特征数组为空")
            return
        
        # 分析聚类质量
        print("\n=== 分析聚类质量 ===")
        
        try:
            qwen_pre_silhouette, qwen_pre_category_scores = analyze_clustering_quality(qwen_pre_features, qwen_pre_categories)
            print(f"QwenVLCLIP-Pretrained 聚类质量分析完成")
        except Exception as e:
            print(f"警告: QwenVLCLIP-Pretrained 聚类质量分析失败: {e}")
            qwen_pre_silhouette, qwen_pre_category_scores = 0.0, {}

        try:
            qwen_ft_silhouette, qwen_ft_category_scores = analyze_clustering_quality(qwen_ft_features, qwen_ft_categories)
            print(f"QwenVLCLIP-Finetuned 聚类质量分析完成")
        except Exception as e:
            print(f"警告: QwenVLCLIP-Finetuned 聚类质量分析失败: {e}")
            qwen_ft_silhouette, qwen_ft_category_scores = 0.0, {}
        
        # 比较两个模型
        print(f"\n=== 模型比较（Qwen 训练前 vs 训练后）===")
        print(f"Qwen-Pretrained 整体轮廓系数: {qwen_pre_silhouette:.4f}")
        print(f"Qwen-Finetuned 整体轮廓系数: {qwen_ft_silhouette:.4f}")
        
        if qwen_ft_silhouette > qwen_pre_silhouette:
            print(f"Qwen-Finetuned 在类别聚合度上表现更好 (+{qwen_ft_silhouette - qwen_pre_silhouette:.4f})")
        else:
            print(f"Qwen-Pretrained 在类别聚合度上表现更好 (+{qwen_pre_silhouette - qwen_ft_silhouette:.4f})")
        
        # 生成可视化
        print("\n=== 生成可视化图表 ===")
        try:
            visualize_feature_comparison(
                qwen_pre_features, qwen_ft_features,
                qwen_pre_categories, qwen_ft_categories,
                "QwenVLCLIP-Pretrained", "QwenVLCLIP-Finetuned", output_dir
            )
            print("可视化图表生成完成")
        except Exception as e:
            print(f"警告: 可视化图表生成失败: {e}")
        
        # 保存特征和结果
        print("\n=== 保存分析结果 ===")
        try:
            results = {
                'qwen_pretrained_silhouette': qwen_pre_silhouette,
                'qwen_finetuned_silhouette': qwen_ft_silhouette,
                'qwen_pretrained_category_scores': dict(qwen_pre_category_scores),
                'qwen_finetuned_category_scores': dict(qwen_ft_category_scores),
                'category_info': dict(category_info),
                'feature_shapes': {
                    'qwen_pretrained': qwen_pre_features.shape,
                    'qwen_finetuned': qwen_ft_features.shape
                },
                'analysis_timestamp': str(datetime.datetime.now())
            }
            
            with open(os.path.join(output_dir, 'analysis_results.json'), 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"分析结果已保存到: {output_dir}")
        except Exception as e:
            print(f"警告: 保存分析结果失败: {e}")
        
        print("\n分析完成！")
        
    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
