"""
评估工具函数
提供各种评估指标的计算功能
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import average_precision_score


def calculate_retrieval_metrics(image_features: torch.Tensor, 
                              text_features: torch.Tensor) -> Dict[str, float]:
    """
    计算检索指标
    
    Args:
        image_features: 图像特征 [N, D]
        text_features: 文本特征 [N, D]
    
    Returns:
        包含各种检索指标的字典
    """
    # 确保特征已归一化
    image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
    
    # 计算相似度矩阵
    similarity_matrix = torch.matmul(image_features, text_features.T)
    
    # 获取批次大小
    batch_size = min(image_features.size(0), text_features.size(0))
    
    # 计算图像到文本检索指标
    i2t_metrics = _calculate_direction_metrics(similarity_matrix, batch_size)
    
    # 计算文本到图像检索指标
    t2i_metrics = _calculate_direction_metrics(similarity_matrix.T, batch_size)
    
    # 合并指标
    metrics = {}
    for key in i2t_metrics:
        metrics[f'i2t_{key}'] = i2t_metrics[key]
        metrics[f't2i_{key}'] = t2i_metrics[key]
        metrics[key] = (i2t_metrics[key] + t2i_metrics[key]) / 2
    
    return metrics


def _calculate_direction_metrics(similarity_matrix: torch.Tensor, 
                               batch_size: int) -> Dict[str, float]:
    """计算单向检索指标"""
    metrics = {}
    
    # 计算排名
    ranks = []
    for i in range(batch_size):
        scores = similarity_matrix[i]
        sorted_indices = torch.argsort(scores, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    
    # R@1, R@5, R@10
    metrics['r1'] = np.mean(ranks <= 1)
    metrics['r5'] = np.mean(ranks <= 5)
    metrics['r10'] = np.mean(ranks <= 10)
    
    # 平均排名和中位数排名
    metrics['mean_rank'] = np.mean(ranks)
    metrics['median_rank'] = np.median(ranks)
    
    # MRR (Mean Reciprocal Rank)
    metrics['mrr'] = np.mean(1.0 / ranks)
    
    # mAP@10
    metrics['mAP@10'] = _calculate_map_at_k(similarity_matrix, batch_size, k=10)
    
    return metrics


def _calculate_map_at_k(similarity_matrix: torch.Tensor, 
                       batch_size: int, 
                       k: int = 10) -> float:
    """计算mAP@k"""
    map_scores = []
    
    for i in range(batch_size):
        scores = similarity_matrix[i].cpu().numpy()
        sorted_indices = np.argsort(scores)[::-1]
        
        # 创建标签（正样本为1，负样本为0）
        labels = np.zeros(len(scores))
        labels[i] = 1  # 正样本
        
        # 计算AP@k
        ap = _calculate_ap_at_k(labels, sorted_indices, k)
        map_scores.append(ap)
    
    return np.mean(map_scores)


def _calculate_ap_at_k(labels: np.ndarray, 
                      sorted_indices: np.ndarray, 
                      k: int) -> float:
    """计算AP@k"""
    # 取前k个结果
    top_k_indices = sorted_indices[:k]
    top_k_labels = labels[top_k_indices]
    
    # 计算精确率
    precision_at_k = np.cumsum(top_k_labels) / np.arange(1, k + 1)
    
    # 计算AP
    if np.sum(top_k_labels) == 0:
        return 0.0
    
    ap = np.sum(precision_at_k * top_k_labels) / np.sum(top_k_labels)
    return ap


def calculate_similarity_metrics(image_features: torch.Tensor, 
                               text_features: torch.Tensor) -> Dict[str, float]:
    """计算相似度相关指标"""
    # 确保特征已归一化
    image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
    
    # 计算相似度矩阵
    similarity_matrix = torch.matmul(image_features, text_features.T)
    
    batch_size = min(image_features.size(0), text_features.size(0))
    
    # 正样本相似度
    positive_sims = torch.diag(similarity_matrix[:batch_size, :batch_size])
    
    # 负样本相似度
    negative_sims = []
    for i in range(batch_size):
        for j in range(batch_size):
            if i != j:
                negative_sims.append(similarity_matrix[i, j])
    
    negative_sims = torch.stack(negative_sims) if negative_sims else torch.tensor([])
    
    # 计算指标
    metrics = {
        'positive_sim_mean': positive_sims.mean().item(),
        'positive_sim_std': positive_sims.std().item(),
        'negative_sim_mean': negative_sims.mean().item() if len(negative_sims) > 0 else 0.0,
        'negative_sim_std': negative_sims.std().item() if len(negative_sims) > 0 else 0.0,
        'similarity_separation': (positive_sims.mean() - negative_sims.mean()).item() if len(negative_sims) > 0 else 0.0
    }
    
    return metrics


def calculate_feature_diversity(features: torch.Tensor) -> Dict[str, float]:
    """计算特征多样性指标"""
    # 计算特征间相似度
    similarity_matrix = torch.matmul(features, features.T)
    
    # 排除对角线
    mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)
    off_diagonal_sims = similarity_matrix[mask]
    
    # 计算多样性指标
    diversity = 1 - off_diagonal_sims.mean().item()
    diversity_std = off_diagonal_sims.std().item()
    
    # 计算有效维度
    effective_dim = _calculate_effective_dimension(features)
    
    return {
        'diversity': diversity,
        'diversity_std': diversity_std,
        'effective_dimension': effective_dim
    }


def _calculate_effective_dimension(features: torch.Tensor) -> float:
    """计算有效维度"""
    # 使用SVD计算有效维度
    U, S, V = torch.svd(features)
    
    # 计算累积解释方差比
    explained_variance_ratio = S**2 / (S**2).sum()
    cumsum = torch.cumsum(explained_variance_ratio, dim=0)
    
    # 找到解释95%方差所需的维度
    effective_dim = torch.argmax(cumsum >= 0.95).item() + 1
    
    return effective_dim


def calculate_attention_quality(attention_maps: torch.Tensor, 
                              ground_truth_masks: torch.Tensor = None) -> Dict[str, float]:
    """计算注意力质量指标"""
    if attention_maps is None:
        return {}
    
    batch_size = attention_maps.size(0)
    quality_scores = []
    
    for i in range(batch_size):
        attention_map = attention_maps[i, 0]  # [H, W]
        
        # 计算注意力集中度
        attention_std = attention_map.std().item()
        attention_mean = attention_map.mean().item()
        concentration = attention_std / (attention_mean + 1e-8)
        
        # 计算注意力覆盖度
        threshold = attention_map.quantile(0.5).item()
        coverage = (attention_map > threshold).float().mean().item()
        
        # 计算注意力平滑度
        grad_x = torch.diff(attention_map, dim=1)
        grad_y = torch.diff(attention_map, dim=0)
        smoothness = 1.0 / (grad_x.std().item() + grad_y.std().item() + 1e-8)
        
        # 如果有真实mask，计算IoU
        iou = 0.0
        if ground_truth_masks is not None:
            gt_mask = ground_truth_masks[i, 0]  # [H, W]
            attention_binary = (attention_map > threshold).float()
            gt_binary = (gt_mask > 0.5).float()
            
            intersection = (attention_binary * gt_binary).sum().item()
            union = (attention_binary + gt_binary).clamp(0, 1).sum().item()
            iou = intersection / (union + 1e-8)
        
        quality_score = {
            'concentration': concentration,
            'coverage': coverage,
            'smoothness': smoothness,
            'iou': iou
        }
        quality_scores.append(quality_score)
    
    # 计算平均质量指标
    avg_quality = {}
    for key in quality_scores[0].keys():
        avg_quality[key] = np.mean([score[key] for score in quality_scores])
    
    return avg_quality


def calculate_gradient_metrics(model, dataloader, max_batches: int = 5) -> Dict[str, float]:
    """计算梯度相关指标"""
    gradient_norms = []
    gradient_ratios = []
    
    # 注册hook来收集梯度
    def hook_fn(module, grad_input, grad_output):
        if grad_output[0] is not None:
            grad_norm = grad_output[0].norm().item()
            gradient_norms.append(grad_norm)
    
    # 注册hooks
    hooks = []
    for name, module in model.named_modules():
        if any(key in name for key in ['projection', 'attention', 'fusion']):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook)
    
    # 前向和反向传播
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
            
        images, texts = batch
        img_feat, txt_feat = model(images, texts)
        
        # 计算损失并反向传播
        loss = torch.nn.functional.mse_loss(img_feat, txt_feat)
        loss.backward()
    
    # 移除hooks
    for hook in hooks:
        hook.remove()
    
    if not gradient_norms:
        return {}
    
    gradient_norms = np.array(gradient_norms)
    
    # 计算梯度指标
    metrics = {
        'gradient_norm_mean': np.mean(gradient_norms),
        'gradient_norm_std': np.std(gradient_norms),
        'gradient_norm_max': np.max(gradient_norms),
        'gradient_norm_min': np.min(gradient_norms),
        'gradient_norm_median': np.median(gradient_norms)
    }
    
    # 计算梯度比率（最大/最小）
    if metrics['gradient_norm_min'] > 0:
        metrics['gradient_ratio'] = metrics['gradient_norm_max'] / metrics['gradient_norm_min']
    else:
        metrics['gradient_ratio'] = float('inf')
    
    return metrics


def calculate_memory_usage(model) -> Dict[str, float]:
    """计算模型内存使用情况"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 估算内存使用（MB）
    param_memory = total_params * 4 / (1024 * 1024)  # 假设float32
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'parameter_memory_mb': param_memory,
        'non_trainable_parameters': total_params - trainable_params
    }


def calculate_inference_time(model, dataloader, max_batches: int = 10) -> Dict[str, float]:
    """计算推理时间"""
    import time
    
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
                
            images, texts = batch
            
            # 预热
            if i == 0:
                for _ in range(3):
                    _ = model(images, texts)
            
            # 计时
            start_time = time.time()
            _ = model(images, texts)
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean_inference_time': np.mean(times),
        'std_inference_time': np.std(times),
        'min_inference_time': np.min(times),
        'max_inference_time': np.max(times),
        'median_inference_time': np.median(times)
    }
