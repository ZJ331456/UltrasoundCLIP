"""
特征分析工具
分析模型特征的质量、分布和相似度
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.stats as stats


class FeatureAnalyzer:
    """特征分析器类"""
    
    def __init__(self, model, output_dir: Path):
        self.model = model
        self.output_dir = output_dir
        self.analysis_dir = output_dir / "feature_analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def analyze_features(self, 
                        dataloader, 
                        max_batches: int = 50,
                        save_plots: bool = True) -> Dict[str, Any]:
        """分析特征质量"""
        print("分析特征质量...")
        
        results = {}
        
        # 收集特征
        image_features, text_features = self._collect_features(dataloader, max_batches)
        
        # 1. 相似度分析
        similarity_analysis = self._analyze_similarity(image_features, text_features, save_plots)
        results.update(similarity_analysis)
        
        # 2. 特征分布分析
        distribution_analysis = self._analyze_distribution(image_features, text_features, save_plots)
        results.update(distribution_analysis)
        
        # 3. 特征多样性分析
        diversity_analysis = self._analyze_diversity(image_features, text_features, save_plots)
        results.update(diversity_analysis)
        
        # 4. 特征聚类分析
        clustering_analysis = self._analyze_clustering(image_features, text_features, save_plots)
        results.update(clustering_analysis)
        
        # 5. 特征相关性分析
        correlation_analysis = self._analyze_correlation(image_features, text_features, save_plots)
        results.update(correlation_analysis)
        
        return results
    
    def _collect_features(self, dataloader, max_batches: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """收集特征"""
        image_features = []
        text_features = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                    
                images, texts = batch
                img_feat, txt_feat = self.model(images, texts)
                
                image_features.append(img_feat.cpu())
                text_features.append(txt_feat.cpu())
        
        return torch.cat(image_features, dim=0), torch.cat(text_features, dim=0)
    
    def _analyze_similarity(self, 
                          image_features: torch.Tensor, 
                          text_features: torch.Tensor,
                          save_plots: bool = True) -> Dict[str, Any]:
        """分析特征相似度"""
        print("分析特征相似度...")
        
        # 转换为numpy
        img_feat = image_features.numpy()
        txt_feat = text_features.numpy()
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(img_feat, txt_feat)
        
        # 提取正负样本相似度
        batch_size = min(len(img_feat), len(txt_feat))
        positive_sims = np.diag(similarity_matrix[:batch_size, :batch_size])
        
        # 负样本相似度（非对角线）
        negative_sims = []
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    negative_sims.append(similarity_matrix[i, j])
        
        negative_sims = np.array(negative_sims)
        
        # 计算分离度指标
        separation = np.mean(positive_sims) - np.mean(negative_sims)
        separation_std = np.std(positive_sims) + np.std(negative_sims)
        
        # 计算排名统计
        ranks = []
        for i in range(batch_size):
            # 对于每个图像，计算其文本的排名
            img_scores = similarity_matrix[i, :batch_size]
            rank = np.sum(img_scores > img_scores[i]) + 1
            ranks.append(rank)
        
        ranks = np.array(ranks)
        
        results = {
            'similarity_separation': separation,
            'similarity_std': separation_std,
            'positive_sim_mean': np.mean(positive_sims),
            'positive_sim_std': np.std(positive_sims),
            'negative_sim_mean': np.mean(negative_sims),
            'negative_sim_std': np.std(negative_sims),
            'mean_rank': np.mean(ranks),
            'median_rank': np.median(ranks),
            'rank_std': np.std(ranks)
        }
        
        # 绘制相似度分布图
        if save_plots:
            self._plot_similarity_distribution(positive_sims, negative_sims, separation)
            self._plot_rank_distribution(ranks)
        
        return results
    
    def _plot_similarity_distribution(self, 
                                    positive_sims: np.ndarray, 
                                    negative_sims: np.ndarray,
                                    separation: float):
        """绘制相似度分布图"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 直方图
        axes[0].hist(positive_sims, bins=50, alpha=0.7, label='Positive', density=True, color='blue')
        axes[0].hist(negative_sims, bins=50, alpha=0.7, label='Negative', density=True, color='red')
        axes[0].set_xlabel('Cosine Similarity')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Similarity Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 箱线图
        axes[1].boxplot([positive_sims, negative_sims], labels=['Positive', 'Negative'])
        axes[1].set_ylabel('Cosine Similarity')
        axes[1].set_title(f'Similarity Box Plot (Separation: {separation:.4f})')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "similarity_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_rank_distribution(self, ranks: np.ndarray):
        """绘制排名分布图"""
        plt.figure(figsize=(10, 6))
        
        plt.hist(ranks, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(ranks), color='red', linestyle='--', 
                   label=f'Mean Rank: {np.mean(ranks):.2f}')
        plt.axvline(np.median(ranks), color='green', linestyle='--', 
                   label=f'Median Rank: {np.median(ranks):.2f}')
        
        plt.xlabel('Rank')
        plt.ylabel('Frequency')
        plt.title('Rank Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "rank_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _analyze_distribution(self, 
                            image_features: torch.Tensor, 
                            text_features: torch.Tensor,
                            save_plots: bool = True) -> Dict[str, Any]:
        """分析特征分布"""
        print("分析特征分布...")
        
        img_feat = image_features.numpy()
        txt_feat = text_features.numpy()
        
        # 计算统计量
        img_stats = {
            'mean': np.mean(img_feat),
            'std': np.std(img_feat),
            'min': np.min(img_feat),
            'max': np.max(img_feat),
            'skewness': stats.skew(img_feat.flatten()),
            'kurtosis': stats.kurtosis(img_feat.flatten())
        }
        
        txt_stats = {
            'mean': np.mean(txt_feat),
            'std': np.std(txt_feat),
            'min': np.min(txt_feat),
            'max': np.max(txt_feat),
            'skewness': stats.skew(txt_feat.flatten()),
            'kurtosis': stats.kurtosis(txt_feat.flatten())
        }
        
        # 计算L2范数分布
        img_norms = np.linalg.norm(img_feat, axis=1)
        txt_norms = np.linalg.norm(txt_feat, axis=1)
        
        results = {
            'image_stats': img_stats,
            'text_stats': txt_stats,
            'image_norm_mean': np.mean(img_norms),
            'image_norm_std': np.std(img_norms),
            'text_norm_mean': np.mean(txt_norms),
            'text_norm_std': np.std(txt_norms)
        }
        
        # 绘制分布图
        if save_plots:
            self._plot_feature_distribution(img_feat, txt_feat, img_stats, txt_stats)
            self._plot_norm_distribution(img_norms, txt_norms)
        
        return results
    
    def _plot_feature_distribution(self, 
                                 img_feat: np.ndarray, 
                                 txt_feat: np.ndarray,
                                 img_stats: Dict[str, float],
                                 txt_stats: Dict[str, float]):
        """绘制特征分布图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 图像特征分布
        axes[0, 0].hist(img_feat.flatten(), bins=100, alpha=0.7, density=True)
        axes[0, 0].set_title(f'Image Feature Distribution\nMean: {img_stats["mean"]:.4f}, Std: {img_stats["std"]:.4f}')
        axes[0, 0].set_xlabel('Feature Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 文本特征分布
        axes[0, 1].hist(txt_feat.flatten(), bins=100, alpha=0.7, density=True)
        axes[0, 1].set_title(f'Text Feature Distribution\nMean: {txt_stats["mean"]:.4f}, Std: {txt_stats["std"]:.4f}')
        axes[0, 1].set_xlabel('Feature Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 对比分布
        axes[0, 2].hist(img_feat.flatten(), bins=100, alpha=0.5, label='Image', density=True)
        axes[0, 2].hist(txt_feat.flatten(), bins=100, alpha=0.5, label='Text', density=True)
        axes[0, 2].set_title('Feature Distribution Comparison')
        axes[0, 2].set_xlabel('Feature Value')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Q-Q图
        from scipy.stats import probplot
        probplot(img_feat.flatten(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Image Features Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        probplot(txt_feat.flatten(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Text Features Q-Q Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 散点图（降维后）
        if img_feat.shape[1] > 2:
            pca = PCA(n_components=2)
            img_2d = pca.fit_transform(img_feat)
            txt_2d = pca.fit_transform(txt_feat)
        else:
            img_2d = img_feat
            txt_2d = txt_feat
        
        axes[1, 2].scatter(img_2d[:, 0], img_2d[:, 1], alpha=0.6, s=20, label='Image')
        axes[1, 2].scatter(txt_2d[:, 0], txt_2d[:, 1], alpha=0.6, s=20, label='Text')
        axes[1, 2].set_title('2D Feature Space (PCA)')
        axes[1, 2].set_xlabel('PC1')
        axes[1, 2].set_ylabel('PC2')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "feature_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_norm_distribution(self, img_norms: np.ndarray, txt_norms: np.ndarray):
        """绘制L2范数分布图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].hist(img_norms, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_title(f'Image Feature L2 Norms\nMean: {np.mean(img_norms):.4f}, Std: {np.std(img_norms):.4f}')
        axes[0].set_xlabel('L2 Norm')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(txt_norms, bins=50, alpha=0.7, edgecolor='black')
        axes[1].set_title(f'Text Feature L2 Norms\nMean: {np.mean(txt_norms):.4f}, Std: {np.std(txt_norms):.4f}')
        axes[1].set_xlabel('L2 Norm')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "norm_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _analyze_diversity(self, 
                         image_features: torch.Tensor, 
                         text_features: torch.Tensor,
                         save_plots: bool = True) -> Dict[str, Any]:
        """分析特征多样性"""
        print("分析特征多样性...")
        
        img_feat = image_features.numpy()
        txt_feat = text_features.numpy()
        
        # 计算特征间相似度
        img_similarity = cosine_similarity(img_feat)
        txt_similarity = cosine_similarity(txt_feat)
        
        # 排除对角线
        mask = ~np.eye(img_similarity.shape[0], dtype=bool)
        img_off_diag = img_similarity[mask]
        txt_off_diag = txt_similarity[mask]
        
        # 计算多样性指标
        img_diversity = 1 - np.mean(img_off_diag)
        txt_diversity = 1 - np.mean(txt_off_diag)
        
        # 计算特征空间的有效维度
        img_effective_dim = self._calculate_effective_dimension(img_feat)
        txt_effective_dim = self._calculate_effective_dimension(txt_feat)
        
        results = {
            'image_diversity': img_diversity,
            'text_diversity': txt_diversity,
            'image_effective_dimension': img_effective_dim,
            'text_effective_dimension': txt_effective_dim,
            'image_similarity_mean': np.mean(img_off_diag),
            'text_similarity_mean': np.mean(txt_off_diag)
        }
        
        # 绘制多样性分析图
        if save_plots:
            self._plot_diversity_analysis(img_off_diag, txt_off_diag, img_diversity, txt_diversity)
        
        return results
    
    def _calculate_effective_dimension(self, features: np.ndarray) -> float:
        """计算有效维度"""
        # 使用PCA计算有效维度
        pca = PCA()
        pca.fit(features)
        
        # 计算累积解释方差比
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        
        # 找到解释95%方差所需的维度
        effective_dim = np.argmax(cumsum >= 0.95) + 1
        
        return effective_dim
    
    def _plot_diversity_analysis(self, 
                               img_off_diag: np.ndarray, 
                               txt_off_diag: np.ndarray,
                               img_diversity: float,
                               txt_diversity: float):
        """绘制多样性分析图"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 图像特征相似度分布
        axes[0].hist(img_off_diag, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_title(f'Image Feature Similarity Distribution\nDiversity: {img_diversity:.4f}')
        axes[0].set_xlabel('Cosine Similarity')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # 文本特征相似度分布
        axes[1].hist(txt_off_diag, bins=50, alpha=0.7, edgecolor='black')
        axes[1].set_title(f'Text Feature Similarity Distribution\nDiversity: {txt_diversity:.4f}')
        axes[1].set_xlabel('Cosine Similarity')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "diversity_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _analyze_clustering(self, 
                          image_features: torch.Tensor, 
                          text_features: torch.Tensor,
                          save_plots: bool = True) -> Dict[str, Any]:
        """分析特征聚类"""
        print("分析特征聚类...")
        
        img_feat = image_features.numpy()
        txt_feat = text_features.numpy()
        
        # 使用K-means聚类
        n_clusters = min(10, len(img_feat) // 5)  # 动态确定聚类数
        
        img_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        txt_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        img_labels = img_kmeans.fit_predict(img_feat)
        txt_labels = txt_kmeans.fit_predict(txt_feat)
        
        # 计算聚类质量指标
        img_inertia = img_kmeans.inertia_
        txt_inertia = txt_kmeans.inertia_
        
        # 计算轮廓系数
        from sklearn.metrics import silhouette_score
        img_silhouette = silhouette_score(img_feat, img_labels)
        txt_silhouette = silhouette_score(txt_feat, txt_labels)
        
        results = {
            'image_clusters': n_clusters,
            'text_clusters': n_clusters,
            'image_inertia': img_inertia,
            'text_inertia': txt_inertia,
            'image_silhouette': img_silhouette,
            'text_silhouette': txt_silhouette
        }
        
        # 绘制聚类结果
        if save_plots:
            self._plot_clustering_results(img_feat, txt_feat, img_labels, txt_labels)
        
        return results
    
    def _plot_clustering_results(self, 
                               img_feat: np.ndarray, 
                               txt_feat: np.ndarray,
                               img_labels: np.ndarray,
                               txt_labels: np.ndarray):
        """绘制聚类结果"""
        # 降维到2D进行可视化
        pca = PCA(n_components=2)
        img_2d = pca.fit_transform(img_feat)
        txt_2d = pca.fit_transform(txt_feat)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 图像特征聚类
        scatter = axes[0].scatter(img_2d[:, 0], img_2d[:, 1], c=img_labels, cmap='tab10', alpha=0.7)
        axes[0].set_title('Image Features Clustering')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[0])
        
        # 文本特征聚类
        scatter = axes[1].scatter(txt_2d[:, 0], txt_2d[:, 1], c=txt_labels, cmap='tab10', alpha=0.7)
        axes[1].set_title('Text Features Clustering')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "clustering_results.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _analyze_correlation(self, 
                           image_features: torch.Tensor, 
                           text_features: torch.Tensor,
                           save_plots: bool = True) -> Dict[str, Any]:
        """分析特征相关性"""
        print("分析特征相关性...")
        
        img_feat = image_features.numpy()
        txt_feat = text_features.numpy()
        
        # 计算特征间的相关性
        img_corr = np.corrcoef(img_feat.T)
        txt_corr = np.corrcoef(txt_feat.T)
        
        # 计算平均相关性
        mask = ~np.eye(img_corr.shape[0], dtype=bool)
        img_avg_corr = np.mean(img_corr[mask])
        txt_avg_corr = np.mean(txt_corr[mask])
        
        # 计算高相关性特征对的比例
        img_high_corr = np.sum(np.abs(img_corr[mask]) > 0.8) / len(img_corr[mask])
        txt_high_corr = np.sum(np.abs(txt_corr[mask]) > 0.8) / len(txt_corr[mask])
        
        results = {
            'image_avg_correlation': img_avg_corr,
            'text_avg_correlation': txt_avg_corr,
            'image_high_correlation_ratio': img_high_corr,
            'text_high_correlation_ratio': txt_high_corr
        }
        
        # 绘制相关性热图
        if save_plots:
            self._plot_correlation_heatmap(img_corr, txt_corr)
        
        return results
    
    def _plot_correlation_heatmap(self, img_corr: np.ndarray, txt_corr: np.ndarray):
        """绘制相关性热图"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 只显示部分特征的相关性（如果特征太多）
        max_features = 50
        if img_corr.shape[0] > max_features:
            img_corr_subset = img_corr[:max_features, :max_features]
            txt_corr_subset = txt_corr[:max_features, :max_features]
        else:
            img_corr_subset = img_corr
            txt_corr_subset = txt_corr
        
        # 图像特征相关性
        sns.heatmap(img_corr_subset, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   ax=axes[0],
                   cbar_kws={'shrink': 0.8})
        axes[0].set_title('Image Features Correlation Matrix')
        
        # 文本特征相关性
        sns.heatmap(txt_corr_subset, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   ax=axes[1],
                   cbar_kws={'shrink': 0.8})
        axes[1].set_title('Text Features Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "correlation_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
