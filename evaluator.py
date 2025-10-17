"""
整合的CLIP模型评估器
支持标准检索指标和细粒度分析
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import os


class Evaluator:
    """整合的CLIP模型评估器 - 优化性能版本"""
    
    def __init__(self, model: torch.nn.Module, device: torch.device, eval_batch_size: int = 32768):
        """
        初始化评估器
        Args:
            model: 模型
            device: 计算设备
            eval_batch_size: 批处理大小（参考CN-CLIP优化）
        """
        # 安全地将模型迁移到目标设备
        try:
            self.model = model.to(device)
        except NotImplementedError:
            # 处理包含meta tensor的模型
            if hasattr(model, 'to_empty'):
                try:
                    model.to_empty(device=device)
                except TypeError:
                    model.to_empty()
                self.model = model.to(device)
            else:
                # 回退：逐模块移动
                for m in model.modules():
                    if hasattr(m, 'to'):
                        m.to(device)
                self.model = model
        self.device = device
    
        # CN-CLIP性能优化参数
        self.eval_batch_size = eval_batch_size
        self.use_fp16 = torch.cuda.is_available()  # 使用半精度加速
    
    def evaluate(self, dataloader: DataLoader, verbose: bool = True) -> Dict[str, float]:
        """
        标准检索评估
        Args:
            dataloader: 数据加载器
            verbose: 是否显示详细信息
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        # 提取所有特征
        all_image_features = []
        all_text_features = []
        
        if verbose:
            print("提取特征中...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Feature extraction", disable=not verbose):
                images = batch['images'].to(self.device)
                
                # 处理文本输入
                if hasattr(self.model, 'config') and 'qwen' in str(type(self.model)).lower():
                    # 对于Qwen模型使用原始文本
                    if 'texts' in batch:
                        text_inputs = batch['texts']
                    elif 'text' in batch:
                        text_data = batch['text']
                        text_inputs = [text_data] if isinstance(text_data, str) else text_data
                    else:
                        raise KeyError(f"Batch中找不到文本数据，可用键: {list(batch.keys())}")
                else:
                    # 使用tokenized结果
                    text_tokens = batch['text_tokens']
                    if isinstance(text_tokens, dict):
                        text_inputs = {k: v.to(self.device) for k, v in text_tokens.items()}
                    else:
                        text_inputs = text_tokens.to(self.device)
                
                # 提取特征
                image_features, text_features = self.model(images, text_inputs)
                
                all_image_features.append(image_features.cpu())
                all_text_features.append(text_features.cpu())
        
        # 拼接所有特征
        image_features = torch.cat(all_image_features, dim=0)
        text_features = torch.cat(all_text_features, dim=0)
        
        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 计算检索指标
        metrics = self._compute_retrieval_metrics(image_features, text_features, verbose)
        
        # 7. 如果模型支持attention map，评估attention相关指标
        if hasattr(self.model, 'core') and hasattr(self.model.core, 'experiment_config'):
            if self.model.core.experiment_config.get('use_attention_head', False):
                attention_metrics = self._evaluate_attention_metrics(dataloader, verbose)
                if attention_metrics:
                    metrics.update(attention_metrics)
        
        return metrics
    
    def _compute_retrieval_metrics(self, 
                                  image_features: torch.Tensor, 
                                  text_features: torch.Tensor,
                                  verbose: bool = True) -> Dict[str, float]:
        """
        计算检索指标 - 参考CN-CLIP的高性能方法
        使用批处理方式计算，避免OOM，提高速度
        """
        n_samples = image_features.size(0)
        
        # 数据类型优化 (参考CN-CLIP)
        if self.use_fp16 and image_features.dtype != torch.float16:
            image_features = image_features.half()
            text_features = text_features.half()
        
        if verbose:
            print(f"样本数量: {n_samples}")
            print(f"图像特征形状: {image_features.shape}")
            print(f"文本特征形状: {text_features.shape}")
            print(f"使用批处理大小: {self.eval_batch_size}")
        
        # 1. 图像到文本检索 (I2T) - 批处理方式
        if verbose:
            print("计算图像→文本检索(I2T)...")
        
        i2t_predictions = self._compute_i2t_predictions_batched(
            image_features, text_features, verbose
        )
        
        # 2. 文本到图像检索 (T2I) - 批处理方式  
        if verbose:
            print("计算文本→图像检索(T2I)...")
        
        t2i_predictions = self._compute_t2i_predictions_batched(
            image_features, text_features, verbose
        )
        
        # 3. 计算Recall@K指标 (参考CN-CLIP的计算方式)
        i2t_metrics = self._calculate_recall_at_k_cnclip_style(i2t_predictions, n_samples, "I2T")
        t2i_metrics = self._calculate_recall_at_k_cnclip_style(t2i_predictions, n_samples, "T2I")
        
        # 4. 合并指标
        metrics = {}
        for k in [1, 5, 10]:
            metrics[f'i2t_r{k}'] = i2t_metrics[f'r{k}']
            metrics[f't2i_r{k}'] = t2i_metrics[f'r{k}']
        
        # 5. 计算平均指标 (参考CN-CLIP)
        metrics['mean_r1'] = (metrics['i2t_r1'] + metrics['t2i_r1']) / 2
        metrics['mean_r5'] = (metrics['i2t_r5'] + metrics['t2i_r5']) / 2
        metrics['mean_r10'] = (metrics['i2t_r10'] + metrics['t2i_r10']) / 2
        
        # 6. 添加排名统计
        i2t_ranks = [pred.index(i) + 1 if i in pred else len(pred) + 1 for i, pred in enumerate(i2t_predictions)]
        t2i_ranks = [pred.index(i) + 1 if i in pred else len(pred) + 1 for i, pred in enumerate(t2i_predictions)]
        
        metrics['i2t_mean_rank'] = np.mean(i2t_ranks)
        metrics['t2i_mean_rank'] = np.mean(t2i_ranks)
        metrics['i2t_median_rank'] = np.median(i2t_ranks)
        metrics['t2i_median_rank'] = np.median(t2i_ranks)
        
        # 注意：attention指标需要在evaluate方法中单独计算，因为需要dataloader
        
        if verbose:
            print(f"I2T平均排名: {metrics['i2t_mean_rank']:.2f}, 中位数排名: {metrics['i2t_median_rank']:.2f}")
            print(f"T2I平均排名: {metrics['t2i_mean_rank']:.2f}, 中位数排名: {metrics['t2i_median_rank']:.2f}")
        
        return metrics
    
    def _compute_i2t_predictions_batched(self, image_features: torch.Tensor, 
                                        text_features: torch.Tensor, 
                                        verbose: bool = True) -> List[List[int]]:
        """
        批处理方式计算I2T预测 - 参考CN-CLIP的性能优化
        """
        n_samples = image_features.size(0)
        i2t_predictions = []
        
        # 批处理计算，避免OOM
        for i in tqdm(range(n_samples), desc="I2T Batched", disable=not verbose):
            query_image = image_features[i:i+1]  # [1, dim]
            
            # 分批计算相似度，避免内存溢出
            all_scores = []
            for start_idx in range(0, text_features.size(0), self.eval_batch_size):
                end_idx = min(start_idx + self.eval_batch_size, text_features.size(0))
                text_batch = text_features[start_idx:end_idx]  # [batch_size, dim]
                
                # 计算批次相似度: [1, batch_size]
                batch_scores = torch.matmul(query_image, text_batch.T)
                all_scores.append(batch_scores.squeeze(0).cpu())
            
            # 合并所有分数
            scores = torch.cat(all_scores, dim=0)  # [n_texts]
            
            # 获取TopK
            _, sorted_indices = torch.sort(scores, descending=True)
            top_k_text_ids = sorted_indices[:10].tolist()
            i2t_predictions.append(top_k_text_ids)
        
        return i2t_predictions
    
    def _compute_t2i_predictions_batched(self, image_features: torch.Tensor, 
                                        text_features: torch.Tensor, 
                                        verbose: bool = True) -> List[List[int]]:
        """
        批处理方式计算T2I预测 - 参考CN-CLIP的性能优化
        """
        n_samples = text_features.size(0)
        t2i_predictions = []
        
        # 批处理计算，避免OOM
        for i in tqdm(range(n_samples), desc="T2I Batched", disable=not verbose):
            query_text = text_features[i:i+1]  # [1, dim]
            
            # 分批计算相似度，避免内存溢出
            all_scores = []
            for start_idx in range(0, image_features.size(0), self.eval_batch_size):
                end_idx = min(start_idx + self.eval_batch_size, image_features.size(0))
                image_batch = image_features[start_idx:end_idx]  # [batch_size, dim]
                
                # 计算批次相似度: [batch_size, 1] -> [batch_size]
                batch_scores = torch.matmul(image_batch, query_text.T).squeeze(1)
                all_scores.append(batch_scores.cpu())
            
            # 合并所有分数
            scores = torch.cat(all_scores, dim=0)  # [n_images]
            
            # 获取TopK
            _, sorted_indices = torch.sort(scores, descending=True)
            top_k_image_ids = sorted_indices[:10].tolist()
            t2i_predictions.append(top_k_image_ids)
        
        return t2i_predictions
    
    def _calculate_recall_at_k_cnclip_style(self, predictions: List[List[int]], n_samples: int, task_type: str) -> Dict[str, float]:
        """
        按照CN-CLIP的标准方式计算Recall@K指标
        
        Args:
            predictions: 每个查询的TopK预测结果 [[top_k_ids], ...]
            n_samples: 总样本数
            task_type: "I2T" 或 "T2I"
        
        Returns:
            Recall@K指标字典
        """
        r1_stat, r5_stat, r10_stat = 0, 0, 0
        
        for query_id in range(n_samples):
            # 获取该查询的TopK预测结果
            top_k_pred_ids = predictions[query_id]
            
            # 对于图像-文本对数据，正确答案就是相同索引
            ground_truth_id = query_id
            
            # 检查正确答案是否在TopK预测中（参考CN-CLIP的逻辑）
            if ground_truth_id in top_k_pred_ids[:1]:  # Top-1
                r1_stat += 1
            if ground_truth_id in top_k_pred_ids[:5]:  # Top-5
                r5_stat += 1
            if ground_truth_id in top_k_pred_ids[:10]: # Top-10
                r10_stat += 1
        
        # 计算Recall@K比例
        r1 = r1_stat / n_samples
        r5 = r5_stat / n_samples
        r10 = r10_stat / n_samples
        
        return {
            'r1': r1,    # Recall@1
            'r5': r5,    # Recall@5
            'r10': r10,  # Recall@10
        }
    
    def _calculate_recall_at_k(self, ranks: List[int]) -> Dict[str, float]:
        """
        传统的基于排名的Recall@K计算（保留兼容性）
        Recall@K = 排名 <= K 的样本数量 / 总样本数量
        """
        ranks = np.array(ranks)
        
        return {
            'r1': (ranks <= 1).mean(),    # Recall@1
            'r5': (ranks <= 5).mean(),    # Recall@5
            'r10': (ranks <= 10).mean(),  # Recall@10
        }
    
    def evaluate_finegrained(self, dataloader: DataLoader, 
                           save_dir: Optional[str] = None,
                           max_batches: int = 100) -> Dict:
        """
        细粒度性能评估
        """
        self.model.eval()
        
        # 收集预测结果
        all_similarities = []
        all_predictions = []
        all_labels = []
        text_descriptions = []
        
        # 解剖部位和病理关键词
        anatomical_parts = ['breast', 'thyroid', 'liver', 'kidney', 'heart', 'abdomen', 'fetal']
        pathology_terms = ['cyst', 'mass', 'nodule', 'lesion', 'tumor', 'normal', 'healthy']
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Fine-grained evaluation")):
                if batch_idx >= max_batches:
                    break
                
                images = batch['images'].to(self.device)
                
                # 处理文本输入
                if hasattr(self.model, 'config') and 'qwen' in str(type(self.model)).lower():
                    texts = batch.get('texts', batch.get('text', []))
                else:
                    texts = batch.get('texts', batch.get('text', []))
                    if not isinstance(texts, list):
                        texts = [texts] if isinstance(texts, str) else []
                
                if not texts:
                    continue
                
                # 前向传播
                try:
                    if hasattr(self.model, 'config') and 'qwen' in str(type(self.model)).lower():
                        text_inputs = texts
                    else:
                        text_tokens = batch['text_tokens']
                        if isinstance(text_tokens, dict):
                            text_inputs = {k: v.to(self.device) for k, v in text_tokens.items()}
                        else:
                            text_inputs = text_tokens.to(self.device)
                    
                    image_features, text_features = self.model(images, text_inputs)
                    
                    # 计算相似度
                    similarity_matrix = torch.matmul(image_features, text_features.T)
                    
                    # 收集结果
                    all_similarities.append(similarity_matrix.cpu())
                    predictions = torch.argmax(similarity_matrix, dim=1)
                    labels = torch.arange(len(images))
                    
                    all_predictions.extend(predictions.cpu().tolist())
                    all_labels.extend(labels.tolist())
                    text_descriptions.extend(texts)
                    
                except Exception as e:
                    print(f"批次 {batch_idx} 处理失败: {e}")
                    continue
        
        if not all_similarities:
            print("警告: 没有有效的批次数据")
            return {}
        
        # 计算整体指标
        try:
            # 尝试连接相似度矩阵，如果失败则使用替代方案
            all_similarities_tensor = torch.cat(all_similarities, dim=0)
        except RuntimeError as e:
            print(f"警告: 无法连接相似度矩阵，使用第一个批次: {e}")
            all_similarities_tensor = all_similarities[0]
        
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        # Top-k准确率（安全计算）
        try:
            top3_accuracy = self._compute_topk_accuracy(all_similarities_tensor, 3)
            top5_accuracy = self._compute_topk_accuracy(all_similarities_tensor, 5)
        except Exception as e:
            print(f"警告: Top-k准确率计算失败: {e}")
            top3_accuracy = 0.0
            top5_accuracy = 0.0
        
        # 细粒度分析
        anatomical_accuracy = self._analyze_category_recognition(
            text_descriptions, all_predictions, all_labels, anatomical_parts, "anatomical"
        )
        pathology_accuracy = self._analyze_category_recognition(
            text_descriptions, all_predictions, all_labels, pathology_terms, "pathology"
        )
        
        # 相似度分布分析（安全版本）
        try:
            similarity_stats = self._analyze_similarity_distribution(all_similarities_tensor)
        except Exception as e:
            print(f"警告: 相似度分布分析失败: {e}")
            similarity_stats = {
                "positive_mean": 0.0, "positive_std": 0.0,
                "negative_mean": 0.0, "negative_std": 0.0,
                "separation": 0.0, "note": "Analysis failed"
            }
        
        # 生成报告
        report = {
            "overall_metrics": {
                "accuracy": float(accuracy),
                "top3_accuracy": float(top3_accuracy),
                "top5_accuracy": float(top5_accuracy),
                "total_samples": len(all_predictions)
            },
            "finegrained_metrics": {
                "anatomical_recognition": anatomical_accuracy,
                "pathology_recognition": pathology_accuracy
            },
            "similarity_analysis": similarity_stats
        }
        
        # 保存结果和可视化
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存数值结果
            with open(f"{save_dir}/finegrained_evaluation.json", "w", encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # 生成可视化（安全版本）
            try:
                self._plot_similarity_distribution(all_similarities_tensor, save_dir)
            except Exception as e:
                print(f"警告: 相似度分布图绘制失败: {e}")
            
            try:
                self._plot_confusion_matrix(all_predictions, all_labels, save_dir)
            except Exception as e:
                print(f"警告: 混淆矩阵绘制失败: {e}")
        
        return report
    
    def _compute_topk_accuracy(self, similarities: torch.Tensor, k: int) -> float:
        """计算Top-k准确率"""
        if similarities.size(0) == 0:
            return 0.0
            
        # 检查是否为方阵
        if similarities.size(0) != similarities.size(1):
            # 非方阵情况，计算基本准确率
            batch_size = similarities.size(0)
            n_features = similarities.size(1)
            if k > n_features:
                k = n_features
            
            # 假设标签是对应的索引（如果可能的话）
            if batch_size <= n_features:
                labels = torch.arange(batch_size)
                _, topk_indices = torch.topk(similarities, k, dim=1)
                correct = (topk_indices == labels.unsqueeze(1)).any(dim=1)
                return correct.float().mean().item()
            else:
                return 0.0  # 无法计算准确率
        
        # 方阵情况，标准计算
        batch_size = similarities.size(0)
        labels = torch.arange(batch_size)
        
        _, topk_indices = torch.topk(similarities, k, dim=1)
        correct = (topk_indices == labels.unsqueeze(1)).any(dim=1)
        return correct.float().mean().item()
    
    def _analyze_category_recognition(self, texts: List[str], predictions: List[int], 
                                    labels: List[int], categories: List[str], 
                                    category_type: str) -> Dict[str, float]:
        """分析特定类别的识别准确率"""
        category_accuracy = {}
        
        for category in categories:
            # 找到包含该类别的文本索引
            category_indices = [
                i for i, text in enumerate(texts) 
                if category.lower() in text.lower()
            ]
            
            if category_indices:
                category_predictions = [predictions[i] for i in category_indices]
                category_labels = [labels[i] for i in category_indices]
                accuracy = np.mean(np.array(category_predictions) == np.array(category_labels))
                category_accuracy[category] = {
                    "accuracy": float(accuracy),
                    "count": len(category_indices)
                }
        
        return category_accuracy
    
    def _analyze_similarity_distribution(self, similarities: torch.Tensor) -> Dict:
        """分析相似度分布"""
        # 检查相似度矩阵是否为方阵
        if similarities.size(0) != similarities.size(1):
            # 非方阵的情况，无法分析对角线，返回基本统计
            all_similarities = similarities.flatten().numpy()
            return {
                "positive_mean": float(np.mean(all_similarities)),
                "positive_std": float(np.std(all_similarities)),
                "positive_min": float(np.min(all_similarities)),
                "positive_max": float(np.max(all_similarities)),
                "negative_mean": float(np.mean(all_similarities)),
                "negative_std": float(np.std(all_similarities)),
                "negative_min": float(np.min(all_similarities)),
                "negative_max": float(np.max(all_similarities)),
                "separation": 0.0,
                "note": "Non-square similarity matrix, limited analysis"
            }
        
        # 方阵的情况，进行完整分析
        # 对角线元素（正样本）
        positive_similarities = torch.diag(similarities).numpy()
        
        # 非对角线元素（负样本）
        mask = ~torch.eye(similarities.size(0), dtype=torch.bool)
        negative_similarities = similarities[mask].numpy()
        
        return {
            "positive_mean": float(np.mean(positive_similarities)),
            "positive_std": float(np.std(positive_similarities)),
            "positive_min": float(np.min(positive_similarities)),
            "positive_max": float(np.max(positive_similarities)),
            "negative_mean": float(np.mean(negative_similarities)),
            "negative_std": float(np.std(negative_similarities)),
            "negative_min": float(np.min(negative_similarities)),
            "negative_max": float(np.max(negative_similarities)),
            "separation": float(np.mean(positive_similarities) - np.mean(negative_similarities))
        }
    
    def _plot_similarity_distribution(self, similarities: torch.Tensor, save_dir: str):
        """绘制相似度分布图"""
        # 检查相似度矩阵是否为方阵
        if similarities.size(0) != similarities.size(1):
            # 非方阵的情况，只绘制整体分布
            all_similarities = similarities.flatten().numpy()
            
            plt.figure(figsize=(8, 6))
            plt.hist(all_similarities, bins=50, alpha=0.7, label='所有相似度', color='blue', density=True)
            plt.xlabel('相似度分数')
            plt.ylabel('密度')
            plt.title('相似度分布（非方阵数据）')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/similarity_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # 方阵的情况，绘制正负样本对比
        positive_similarities = torch.diag(similarities).numpy()
        mask = ~torch.eye(similarities.size(0), dtype=torch.bool)
        negative_similarities = similarities[mask].numpy()
        
        plt.figure(figsize=(12, 6))
        
        # 左图：直方图
        plt.subplot(1, 2, 1)
        plt.hist(positive_similarities, bins=50, alpha=0.7, label='正样本对', color='green', density=True)
        plt.hist(negative_similarities, bins=50, alpha=0.7, label='负样本对', color='red', density=True)
        plt.xlabel('相似度分数')
        plt.ylabel('密度')
        plt.title('相似度分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 右图：箱线图
        plt.subplot(1, 2, 2)
        data = [positive_similarities, negative_similarities]
        labels = ['正样本对', '负样本对']
        plt.boxplot(data, labels=labels)
        plt.ylabel('相似度分数')
        plt.title('相似度分布箱线图')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/similarity_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, predictions: List[int], labels: List[int], save_dir: str):
        """绘制混淆矩阵（采样显示）"""
        if not predictions or not labels:
            print("警告: 预测或标签列表为空，跳过混淆矩阵绘制")
            return
            
        try:
            # 限制显示的类别数量
            max_pred = max(predictions) if predictions else 0
            max_label = max(labels) if labels else 0
            max_classes = min(20, max(max_pred, max_label) + 1)
            
            # 过滤数据
            filtered_preds = [p for p in predictions if p < max_classes]
            filtered_labels = [l for l in labels if l < max_classes]
            
            if len(filtered_preds) > 0 and len(filtered_labels) > 0:
                cm = confusion_matrix(filtered_labels, filtered_preds)
                
                plt.figure(figsize=(12, 10))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': '样本数量'})
                plt.title(f'混淆矩阵 (前{max_classes}个类别)')
                plt.ylabel('真实标签')
                plt.xlabel('预测标签')
                plt.tight_layout()
                plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                print("警告: 过滤后数据为空，跳过混淆矩阵绘制")
        except Exception as e:
            print(f"绘制混淆矩阵时出错: {e}")
            print("跳过混淆矩阵绘制")
    
    def print_metrics(self, metrics: Dict[str, float]):
        """打印评估指标 - 参考CN-CLIP的输出格式"""
        print("\n" + "="*60)
        print("CLIP模型评估结果 (参考CN-CLIP标准)")
        print("="*60)
        
        if 'i2t_r1' in metrics:
            # 1. 双向检索结果
            print(f"📊 检索性能指标:")
            print(f"  图像→文本检索 (I2T):")
            print(f"    Recall@1:  {metrics['i2t_r1']:.4f} ({metrics['i2t_r1']*100:.2f}%)")
            print(f"    Recall@5:  {metrics['i2t_r5']:.4f} ({metrics['i2t_r5']*100:.2f}%)")
            print(f"    Recall@10: {metrics['i2t_r10']:.4f} ({metrics['i2t_r10']*100:.2f}%)")
            
            print(f"\n  文本→图像检索 (T2I):")
            print(f"    Recall@1:  {metrics['t2i_r1']:.4f} ({metrics['t2i_r1']*100:.2f}%)")
            print(f"    Recall@5:  {metrics['t2i_r5']:.4f} ({metrics['t2i_r5']*100:.2f}%)")
            print(f"    Recall@10: {metrics['t2i_r10']:.4f} ({metrics['t2i_r10']*100:.2f}%)")
            
            # 2. 平均性能 (参考CN-CLIP的mean_recall计算)
            print(f"\n🎯 综合性能 (CN-CLIP标准):")
            print(f"    Mean Recall@1:  {metrics['mean_r1']:.4f} ({metrics['mean_r1']*100:.2f}%)")
            print(f"    Mean Recall@5:  {metrics['mean_r5']:.4f} ({metrics['mean_r5']*100:.2f}%)")
            print(f"    Mean Recall@10: {metrics['mean_r10']:.4f} ({metrics['mean_r10']*100:.2f}%)")
            
            # 3. CN-CLIP式综合得分
            cn_clip_score = (metrics['mean_r1'] + metrics['mean_r5'] + metrics['mean_r10']) / 3.0 * 100
            print(f"\n🏆 CN-CLIP综合得分: {cn_clip_score:.2f}")
            
            # 4. 排名统计
            if 'i2t_mean_rank' in metrics:
                print(f"\n📈 排名统计:")
                print(f"    I2T平均排名: {metrics['i2t_mean_rank']:.2f}")
                print(f"    T2I平均排名: {metrics['t2i_mean_rank']:.2f}")
                print(f"    I2T中位数排名: {metrics['i2t_median_rank']:.2f}")
                print(f"    T2I中位数排名: {metrics['t2i_median_rank']:.2f}")
        
        print("="*60)
    
    def _evaluate_attention_metrics(self, dataloader: DataLoader, verbose: bool = True) -> Dict[str, float]:
        """评估attention相关指标"""
        if verbose:
            print("评估attention指标...")
        
        attention_metrics = {
            'attention_coverage': [],  # attention覆盖mask的比例
            'attention_inside_ratio': [],  # attention在mask内的比例
            'attention_sparsity': [],  # attention的稀疏度
            'attention_entropy': [],  # attention的熵
        }
        
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Attention evaluation", disable=not verbose):
                images = batch['images'].to(self.device)
                masks = batch.get('mask')  # 可能没有mask
                
                # 处理文本输入
                if hasattr(self.model, 'config') and 'qwen' in str(type(self.model)).lower():
                    if 'texts' in batch:
                        text_inputs = batch['texts']
                    elif 'text' in batch:
                        text_data = batch['text']
                        text_inputs = [text_data] if isinstance(text_data, str) else text_data
                    else:
                        continue
                else:
                    text_tokens = batch['text_tokens']
                    if isinstance(text_tokens, dict):
                        text_inputs = {k: v.to(self.device) for k, v in text_tokens.items()}
                    else:
                        text_inputs = text_tokens.to(self.device)
                
                # 前向传播获取attention map
                self.model.eval()
                result = self.model.core.forward(images, text_inputs, masks=masks)
                
                if 'attention_map' in result:
                    attention_map = result['attention_map']  # [B, 1, H, W]
                    
                    # 计算attention指标
                    B, _, H, W = attention_map.shape
                    
                    for i in range(B):
                        att_map = attention_map[i, 0]  # [H, W]
                        
                        # 稀疏度：激活区域的比例
                        threshold = 0.1
                        active_ratio = (att_map > threshold).float().mean().item()
                        attention_metrics['attention_sparsity'].append(active_ratio)
                        
                        # 熵：衡量分布的集中程度
                        att_flat = att_map.flatten()
                        att_prob = F.softmax(att_flat, dim=0)
                        entropy = -(att_prob * torch.log(att_prob + 1e-8)).sum().item()
                        attention_metrics['attention_entropy'].append(entropy)
                        
                        # 如果有mask，计算覆盖度和inside ratio
                        if masks is not None and i < masks.shape[0]:
                            mask = masks[i].to(self.device)
                            if mask.shape[-2:] != (H, W):
                                mask = F.interpolate(mask.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
                            mask = mask[0] > 0.5  # 二值化
                            
                            if mask.sum() > 0:
                                # 覆盖度：attention和mask的IoU
                                att_binary = att_map > threshold
                                intersection = (att_binary & mask).float().sum()
                                union = (att_binary | mask).float().sum()
                                iou = (intersection / (union + 1e-8)).item()
                                attention_metrics['attention_coverage'].append(iou)
                                
                                # Inside ratio：attention权重在mask内的比例
                                inside_weight = (att_map * mask).sum()
                                total_weight = att_map.sum()
                                inside_ratio = (inside_weight / (total_weight + 1e-8)).item()
                                attention_metrics['attention_inside_ratio'].append(inside_ratio)
                    
                    total_samples += B
        
        # 计算平均值
        metrics = {}
        for key, values in attention_metrics.items():
            if values:
                metrics[f'avg_{key}'] = np.mean(values)
                metrics[f'std_{key}'] = np.std(values)
        
        metrics['attention_evaluated_samples'] = total_samples
        
        if verbose:
            print(f"评估了 {total_samples} 个样本的attention指标")
            if 'avg_attention_coverage' in metrics:
                print(f"  Attention覆盖度: {metrics['avg_attention_coverage']:.4f} ± {metrics['std_attention_coverage']:.4f}")
            if 'avg_attention_inside_ratio' in metrics:
                print(f"  Attention内部比例: {metrics['avg_attention_inside_ratio']:.4f} ± {metrics['std_attention_inside_ratio']:.4f}")
            print(f"  Attention稀疏度: {metrics.get('avg_attention_sparsity', 0):.4f} ± {metrics.get('std_attention_sparsity', 0):.4f}")
            print(f"  Attention熵: {metrics.get('avg_attention_entropy', 0):.4f} ± {metrics.get('std_attention_entropy', 0):.4f}")
        
        return metrics
    
    def print_finegrained_metrics(self, report: Dict):
        """打印细粒度评估结果"""
        print("\n" + "="*50)
        print("细粒度评估结果")
        print("="*50)
        
        overall = report.get('overall_metrics', {})
        print(f"整体准确率: {overall.get('accuracy', 0):.4f}")
        print(f"Top-3准确率: {overall.get('top3_accuracy', 0):.4f}")
        print(f"Top-5准确率: {overall.get('top5_accuracy', 0):.4f}")
        print(f"总样本数: {overall.get('total_samples', 0)}")
        
        # 解剖部位识别
        anatomical = report.get('finegrained_metrics', {}).get('anatomical_recognition', {})
        if anatomical:
            print(f"\n解剖部位识别:")
            for part, metrics in anatomical.items():
                print(f"  {part}: {metrics['accuracy']:.4f} ({metrics['count']}样本)")
        
        # 病理识别
        pathology = report.get('finegrained_metrics', {}).get('pathology_recognition', {})
        if pathology:
            print(f"\n病理特征识别:")
            for term, metrics in pathology.items():
                print(f"  {term}: {metrics['accuracy']:.4f} ({metrics['count']}样本)")
        
        # 相似度分析
        sim_stats = report.get('similarity_analysis', {})
        if sim_stats:
            print(f"\n相似度分析:")
            print(f"  正样本对: {sim_stats.get('positive_mean', 0):.4f} ± {sim_stats.get('positive_std', 0):.4f}")
            print(f"  负样本对: {sim_stats.get('negative_mean', 0):.4f} ± {sim_stats.get('negative_std', 0):.4f}")
            print(f"  分离度: {sim_stats.get('separation', 0):.4f}")
        
        print("="*50)
