"""
可视化工具
提供各种模型性能可视化功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import cv2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap


class VisualizationTools:
    """可视化工具类"""
    
    def __init__(self, model, output_dir: Path):
        self.model = model
        self.output_dir = output_dir
        self.viz_dir = output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def visualize_attention_maps(self, 
                               dataloader, 
                               max_batches: int = 10,
                               save_plots: bool = True) -> Dict[str, Any]:
        """可视化注意力图"""
        print("可视化注意力图...")
        
        attention_dir = self.viz_dir / "attention_maps"
        attention_dir.mkdir(exist_ok=True)
        
        results = {
            'attention_quality_scores': [],
            'attention_consistency': [],
            'attention_coverage': []
        }
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                    
                images, texts, masks = batch
                
                # 获取模型输出
                if hasattr(self.model.core, 'use_attention_head') and self.model.core.use_attention_head:
                    result = self.model.core.forward(images, texts, masks=masks)
                    attention_maps = result.get('attention_map')
                    
                    if attention_maps is not None:
                        # 分析每个样本的注意力图
                        for j in range(len(images)):
                            attention_map = attention_maps[j, 0].cpu().numpy()  # [H, W]
                            image = images[j].permute(1, 2, 0).cpu().numpy()
                            text = texts[j]
                            mask = masks[j, 0].cpu().numpy() if masks is not None else None
                            
                            # 可视化单个样本
                            self._visualize_single_attention(
                                image, attention_map, text, mask, 
                                attention_dir / f"batch_{i}_sample_{j}.png"
                            )
                            
                            # 计算注意力质量指标
                            quality_score = self._calculate_attention_quality(attention_map, mask)
                            results['attention_quality_scores'].append(quality_score)
                            
                            # 计算注意力一致性
                            consistency = self._calculate_attention_consistency(attention_map)
                            results['attention_consistency'].append(consistency)
                            
                            # 计算注意力覆盖度
                            coverage = self._calculate_attention_coverage(attention_map)
                            results['attention_coverage'].append(coverage)
        
        # 生成统计图表
        if save_plots:
            self._plot_attention_statistics(results, attention_dir)
        
        return results
    
    def _visualize_single_attention(self, 
                                  image: np.ndarray, 
                                  attention_map: np.ndarray,
                                  text: str,
                                  mask: Optional[np.ndarray],
                                  save_path: Path):
        """可视化单个样本的注意力图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 原始图像
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 注意力图
        im1 = axes[0, 1].imshow(attention_map, cmap='hot', alpha=0.8)
        axes[0, 1].set_title('Attention Map')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # 注意力叠加
        axes[1, 0].imshow(image)
        axes[1, 0].imshow(attention_map, cmap='hot', alpha=0.6)
        axes[1, 0].set_title('Attention Overlay')
        axes[1, 0].axis('off')
        
        # 真实mask（如果有）
        if mask is not None:
            axes[1, 1].imshow(mask, cmap='gray')
            axes[1, 1].set_title('Ground Truth Mask')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Ground Truth Mask', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Ground Truth Mask')
        axes[1, 1].axis('off')
        
        # 添加文本信息
        fig.suptitle(f'Text: {text[:50]}...', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _calculate_attention_quality(self, attention_map: np.ndarray, mask: Optional[np.ndarray]) -> float:
        """计算注意力质量分数"""
        if mask is None:
            # 如果没有真实mask，使用注意力图的集中度作为质量指标
            return np.std(attention_map) / (np.mean(attention_map) + 1e-8)
        
        # 计算注意力图与真实mask的IoU
        attention_binary = (attention_map > np.percentile(attention_map, 70)).astype(np.uint8)
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        intersection = np.logical_and(attention_binary, mask_binary).sum()
        union = np.logical_or(attention_binary, mask_binary).sum()
        
        return intersection / (union + 1e-8)
    
    def _calculate_attention_consistency(self, attention_map: np.ndarray) -> float:
        """计算注意力一致性（局部平滑度）"""
        # 计算梯度的L2范数作为一致性指标
        grad_x = np.gradient(attention_map, axis=1)
        grad_y = np.gradient(attention_map, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return 1.0 / (np.mean(gradient_magnitude) + 1e-8)
    
    def _calculate_attention_coverage(self, attention_map: np.ndarray) -> float:
        """计算注意力覆盖度"""
        threshold = np.percentile(attention_map, 50)
        coverage = (attention_map > threshold).sum() / attention_map.size
        return coverage
    
    def _plot_attention_statistics(self, results: Dict[str, Any], save_dir: Path):
        """绘制注意力统计图表"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 注意力质量分布
        axes[0].hist(results['attention_quality_scores'], bins=20, alpha=0.7)
        axes[0].set_title('Attention Quality Distribution')
        axes[0].set_xlabel('Quality Score')
        axes[0].set_ylabel('Frequency')
        
        # 注意力一致性分布
        axes[1].hist(results['attention_consistency'], bins=20, alpha=0.7)
        axes[1].set_title('Attention Consistency Distribution')
        axes[1].set_xlabel('Consistency Score')
        axes[1].set_ylabel('Frequency')
        
        # 注意力覆盖度分布
        axes[2].hist(results['attention_coverage'], bins=20, alpha=0.7)
        axes[2].set_title('Attention Coverage Distribution')
        axes[2].set_xlabel('Coverage Ratio')
        axes[2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(save_dir / "attention_statistics.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_feature_space(self, 
                              dataloader, 
                              max_batches: int = 20,
                              save_plots: bool = True) -> Dict[str, Any]:
        """可视化特征空间"""
        print("可视化特征空间...")
        
        space_dir = self.viz_dir / "feature_space"
        space_dir.mkdir(exist_ok=True)
        
        # 收集特征
        image_features = []
        text_features = []
        labels = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                    
                images, texts = batch
                img_feat, txt_feat = self.model(images, texts)
                
                image_features.append(img_feat.cpu())
                text_features.append(txt_feat.cpu())
                labels.extend(['image'] * len(img_feat) + ['text'] * len(txt_feat))
        
        # 合并特征
        image_features = torch.cat(image_features, dim=0).numpy()
        text_features = torch.cat(text_features, dim=0).numpy()
        all_features = np.vstack([image_features, text_features])
        
        results = {}
        
        # t-SNE可视化
        if save_plots:
            tsne_result = self._plot_tsne(all_features, labels, space_dir)
            results['tsne'] = tsne_result
        
        # UMAP可视化
        if save_plots:
            umap_result = self._plot_umap(all_features, labels, space_dir)
            results['umap'] = umap_result
        
        # PCA可视化
        if save_plots:
            pca_result = self._plot_pca(all_features, labels, space_dir)
            results['pca'] = pca_result
        
        # 特征相似度热图
        if save_plots:
            self._plot_similarity_heatmap(image_features, text_features, space_dir)
        
        return results
    
    def _plot_tsne(self, features: np.ndarray, labels: List[str], save_dir: Path) -> Dict[str, Any]:
        """绘制t-SNE图"""
        print("计算t-SNE...")
        
        # 如果特征太多，先进行PCA降维
        if features.shape[0] > 1000:
            pca = PCA(n_components=50)
            features = pca.fit_transform(features)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embedding = tsne.fit_transform(features)
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))
        
        # 分别绘制图像和文本特征
        image_mask = np.array(labels) == 'image'
        text_mask = np.array(labels) == 'text'
        
        plt.scatter(embedding[image_mask, 0], embedding[image_mask, 1], 
                   c='blue', alpha=0.6, label='Image Features', s=20)
        plt.scatter(embedding[text_mask, 0], embedding[text_mask, 1], 
                   c='red', alpha=0.6, label='Text Features', s=20)
        
        plt.title('t-SNE Visualization of Feature Space')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / "tsne_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'embedding': embedding,
            'image_points': np.sum(image_mask),
            'text_points': np.sum(text_mask)
        }
    
    def _plot_umap(self, features: np.ndarray, labels: List[str], save_dir: Path) -> Dict[str, Any]:
        """绘制UMAP图"""
        print("计算UMAP...")
        
        # 如果特征太多，先进行PCA降维
        if features.shape[0] > 1000:
            pca = PCA(n_components=50)
            features = pca.fit_transform(features)
        
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(features)
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))
        
        # 分别绘制图像和文本特征
        image_mask = np.array(labels) == 'image'
        text_mask = np.array(labels) == 'text'
        
        plt.scatter(embedding[image_mask, 0], embedding[image_mask, 1], 
                   c='blue', alpha=0.6, label='Image Features', s=20)
        plt.scatter(embedding[text_mask, 0], embedding[text_mask, 1], 
                   c='red', alpha=0.6, label='Text Features', s=20)
        
        plt.title('UMAP Visualization of Feature Space')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / "umap_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'embedding': embedding,
            'image_points': np.sum(image_mask),
            'text_points': np.sum(text_mask)
        }
    
    def _plot_pca(self, features: np.ndarray, labels: List[str], save_dir: Path) -> Dict[str, Any]:
        """绘制PCA图"""
        print("计算PCA...")
        
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(features)
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))
        
        # 分别绘制图像和文本特征
        image_mask = np.array(labels) == 'image'
        text_mask = np.array(labels) == 'text'
        
        plt.scatter(embedding[image_mask, 0], embedding[image_mask, 1], 
                   c='blue', alpha=0.6, label='Image Features', s=20)
        plt.scatter(embedding[text_mask, 0], embedding[text_mask, 1], 
                   c='red', alpha=0.6, label='Text Features', s=20)
        
        plt.title('PCA Visualization of Feature Space')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / "pca_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'embedding': embedding,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'image_points': np.sum(image_mask),
            'text_points': np.sum(text_mask)
        }
    
    def _plot_similarity_heatmap(self, 
                               image_features: np.ndarray, 
                               text_features: np.ndarray, 
                               save_dir: Path):
        """绘制相似度热图"""
        print("计算相似度热图...")
        
        # 计算相似度矩阵
        similarity_matrix = np.matmul(image_features, text_features.T)
        
        # 绘制热图
        plt.figure(figsize=(12, 10))
        
        # 只显示前50x50的子矩阵（如果太大）
        if similarity_matrix.shape[0] > 50 or similarity_matrix.shape[1] > 50:
            similarity_subset = similarity_matrix[:50, :50]
        else:
            similarity_subset = similarity_matrix
        
        sns.heatmap(similarity_subset, 
                   cmap='viridis', 
                   cbar=True,
                   square=True,
                   xticklabels=False,
                   yticklabels=False)
        
        plt.title('Image-Text Similarity Matrix')
        plt.xlabel('Text Features')
        plt.ylabel('Image Features')
        
        plt.tight_layout()
        plt.savefig(save_dir / "similarity_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_training_curves(self, log_file: str, save_plots: bool = True):
        """可视化训练曲线"""
        print("可视化训练曲线...")
        
        curves_dir = self.viz_dir / "training_curves"
        curves_dir.mkdir(exist_ok=True)
        
        # 这里需要解析日志文件
        # 由于日志格式可能不同，这里提供一个框架
        # 实际实现需要根据具体的日志格式进行调整
        
        if save_plots:
            # 示例：绘制损失曲线
            plt.figure(figsize=(12, 8))
            
            # 这里需要从日志文件中提取数据
            # 暂时使用示例数据
            epochs = range(1, 101)
            train_loss = np.random.exponential(2.0, 100) * np.exp(-np.array(epochs) * 0.02)
            val_loss = train_loss + np.random.normal(0, 0.1, 100)
            
            plt.plot(epochs, train_loss, label='Training Loss', alpha=0.8)
            plt.plot(epochs, val_loss, label='Validation Loss', alpha=0.8)
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(curves_dir / "loss_curves.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    def visualize_gradient_flow(self, dataloader, max_batches: int = 5, save_plots: bool = True):
        """可视化梯度流"""
        print("可视化梯度流...")
        
        grad_dir = self.viz_dir / "gradient_flow"
        grad_dir.mkdir(exist_ok=True)
        
        # 收集梯度信息
        gradient_norms = {
            'image_projection': [],
            'text_projection': [],
            'attention_layer': [],
            'fusion_layer': []
        }
        
        # 注册hook来收集梯度
        def hook_fn(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    grad_norm = grad_output[0].norm().item()
                    gradient_norms[name].append(grad_norm)
            return hook
        
        # 注册hooks
        hooks = []
        for name, module in self.model.named_modules():
            if 'image_projection' in name:
                hook = module.register_backward_hook(hook_fn('image_projection'))
                hooks.append(hook)
            elif 'text_projection' in name:
                hook = module.register_backward_hook(hook_fn('text_projection'))
                hooks.append(hook)
            elif 'attn' in name:
                hook = module.register_backward_hook(hook_fn('attention_layer'))
                hooks.append(hook)
            elif 'fusion' in name:
                hook = module.register_backward_hook(hook_fn('fusion_layer'))
                hooks.append(hook)
        
        # 前向和反向传播
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
                
            images, texts = batch
            img_feat, txt_feat = self.model(images, texts)
            
            # 计算损失并反向传播
            loss = torch.nn.functional.mse_loss(img_feat, txt_feat)
            loss.backward()
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        # 绘制梯度分布
        if save_plots:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, (layer, grads) in enumerate(gradient_norms.items()):
                if grads:
                    axes[i].hist(grads, bins=20, alpha=0.7)
                    axes[i].set_title(f'{layer} Gradient Norms')
                    axes[i].set_xlabel('Gradient Norm')
                    axes[i].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(grad_dir / "gradient_distribution.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        return gradient_norms
