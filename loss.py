"""
损失函数模块
包含各种CLIP训练损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CLIPLoss(nn.Module):
    """标准CLIP损失函数"""

    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        计算CLIP损失
        Args:
            image_features: 图像特征 [batch_size, embed_dim]
            text_features: 文本特征 [batch_size, embed_dim]
        Returns:
            损失值
        """
        # 计算相似度矩阵
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        
        # 创建标签
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)
        
        # 计算损失（支持标签平滑）
        if self.label_smoothing > 0:
            loss_i2t = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
            loss_t2i = F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)
        else:
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2


class HardNegativeLoss(nn.Module):
    """硬负样本损失函数"""
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.2, k_hard: int = 5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.k_hard = k_hard
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        计算硬负样本损失
        """
        batch_size = image_features.size(0)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(image_features, text_features.T) / self.temperature
        
        # 创建标签
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        # 图像到文本损失
        i2t_loss = self._compute_hard_negative_loss(similarity_matrix, labels)
        
        # 文本到图像损失
        t2i_loss = self._compute_hard_negative_loss(similarity_matrix.T, labels)
        
        return (i2t_loss + t2i_loss) / 2
    
    def _compute_hard_negative_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算单向硬负样本损失"""
        batch_size = logits.size(0)
        
        # 获取正样本分数
        positive_scores = logits[torch.arange(batch_size), labels]
        
        # 创建负样本mask
        negative_mask = torch.ones_like(logits, dtype=torch.bool)
        negative_mask[torch.arange(batch_size), labels] = False
        
        # 获取负样本分数
        negative_scores = logits[negative_mask].view(batch_size, -1)
        
        # 选择最难的k个负样本
        hard_negatives, _ = torch.topk(negative_scores, self.k_hard, dim=1)
        
        # 计算损失
        pos_exp = torch.exp(positive_scores.unsqueeze(1))
        neg_exp = torch.exp(hard_negatives)
        
        loss = -torch.log(pos_exp / (pos_exp + neg_exp.sum(dim=1, keepdim=True)))
        
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal损失函数"""
    
    def __init__(self, temperature: float = 0.07, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """计算Focal损失"""
        # 计算相似度矩阵
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)
        
        # 计算标准交叉熵损失
        ce_loss_i2t = F.cross_entropy(logits, labels, reduction='none')
        ce_loss_t2i = F.cross_entropy(logits.T, labels, reduction='none')
        
        # 计算概率
        pt_i2t = torch.exp(-ce_loss_i2t)
        pt_t2i = torch.exp(-ce_loss_t2i)
        
        # 计算Focal损失
        focal_loss_i2t = self.alpha * (1 - pt_i2t) ** self.gamma * ce_loss_i2t
        focal_loss_t2i = self.alpha * (1 - pt_t2i) ** self.gamma * ce_loss_t2i
        
        return (focal_loss_i2t.mean() + focal_loss_t2i.mean()) / 2


class SoftTargetLoss(nn.Module):
    """软目标损失函数 - 基于特征相似度的软标签"""
    
    def __init__(self, temperature: float = 0.07, soft_temperature: float = 0.5, alpha: float = 0.8):
        super().__init__()
        self.temperature = temperature
        self.soft_temperature = soft_temperature
        self.alpha = alpha  # 硬标签和软标签的权重平衡
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        计算软目标损失
        Args:
            image_features: 图像特征 [batch_size, embed_dim]
            text_features: 文本特征 [batch_size, embed_dim]
        """
        batch_size = image_features.size(0)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(image_features, text_features.T) / self.temperature
        
        # 创建硬标签
        hard_labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        # 创建软标签 - 基于特征相似度
        with torch.no_grad():
            # 图像间的相似度
            img_sim = torch.matmul(image_features, image_features.T) / self.soft_temperature
            img_sim = F.softmax(img_sim, dim=1)
            
            # 文本间的相似度
            text_sim = torch.matmul(text_features, text_features.T) / self.soft_temperature
            text_sim = F.softmax(text_sim, dim=1)
            
            # 组合软标签
            soft_labels_i2t = (img_sim + text_sim) / 2
            soft_labels_t2i = soft_labels_i2t.T
        
        # 计算硬标签损失
        hard_loss_i2t = F.cross_entropy(similarity_matrix, hard_labels)
        hard_loss_t2i = F.cross_entropy(similarity_matrix.T, hard_labels)
        
        # 计算软标签损失 (KL散度)
        # soft_probs_i2t = F.log_softmax(similarity_matrix, dim=1)
        # soft_probs_t2i = F.log_softmax(similarity_matrix.T, dim=1)
        
        # soft_loss_i2t = F.kl_div(soft_probs_i2t, soft_labels_i2t, reduction='batchmean')
        # soft_loss_t2i = F.kl_div(soft_probs_t2i, soft_labels_t2i, reduction='batchmean')
        soft_loss_i2t = F.kl_div(F.log_softmax(similarity_matrix, dim=1), soft_labels_i2t, reduction='batchmean')
        soft_loss_t2i = F.kl_div(F.log_softmax(similarity_matrix.T, dim=1), soft_labels_t2i, reduction='batchmean')

        # 组合损失
        total_loss_i2t = self.alpha * hard_loss_i2t + (1 - self.alpha) * soft_loss_i2t
        total_loss_t2i = self.alpha * hard_loss_t2i + (1 - self.alpha) * soft_loss_t2i
        
        return (total_loss_i2t + total_loss_t2i) / 2


class RankingLoss(nn.Module):
    """排序损失函数 - 更好地区分样本间的差距"""
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.3, num_negatives: int = 5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.num_negatives = num_negatives
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        计算排序损失
        """
        batch_size = image_features.size(0)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(image_features, text_features.T) / self.temperature
        
        # 获取正样本分数
        positive_scores = similarity_matrix.diag()
        
        loss = 0.0
        count = 0
        
        for i in range(batch_size):
            # 获取负样本（除了正样本外的所有样本）
            negative_mask = torch.ones(batch_size, dtype=torch.bool, device=similarity_matrix.device)
            negative_mask[i] = False
            negative_scores = similarity_matrix[i][negative_mask]
            
            # 选择最相似的几个负样本
            if negative_scores.size(0) > self.num_negatives:
                negative_scores, _ = torch.topk(negative_scores, self.num_negatives)
            
            # 计算排序损失：正样本应该比负样本高出margin
            for neg_score in negative_scores:
                ranking_loss = torch.clamp(self.margin - positive_scores[i] + neg_score, min=0)
                loss += ranking_loss
                count += 1
        
        return loss / max(count, 1)


class ContrastiveLoss(nn.Module):
    """对比学习损失 - 考虑样本间的细粒度差异"""
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.5, 
                 use_similarity_weights: bool = True):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.use_similarity_weights = use_similarity_weights
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        计算对比学习损失
        """
        batch_size = image_features.size(0)
        
        # 归一化特征
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(image_features, text_features.T) / self.temperature
        
        if self.use_similarity_weights:
            # 计算样本间的相似度权重
            with torch.no_grad():
                img_sim = torch.matmul(image_features, image_features.T)
                text_sim = torch.matmul(text_features, text_features.T)
                
                # 创建权重矩阵：相似度越高，权重越大
                weights = (img_sim + text_sim) / 2
                weights = torch.exp(weights / self.temperature)
                
                # 对角线置零（不考虑自己和自己的相似度）
                weights.fill_diagonal_(0)
        else:
            weights = torch.ones_like(similarity_matrix)
            weights.fill_diagonal_(0)
        
        # 计算加权对比损失
        positive_pairs = similarity_matrix.diag()
        
        # 图像到文本损失
        i2t_loss = 0
        for i in range(batch_size):
            pos_score = positive_pairs[i]
            neg_scores = similarity_matrix[i]
            neg_weights = weights[i]
            
            # 加权负样本损失
            weighted_neg_exp = torch.sum(neg_weights * torch.exp(neg_scores))
            i2t_loss += -torch.log(torch.exp(pos_score) / (torch.exp(pos_score) + weighted_neg_exp))
        
        # 文本到图像损失
        t2i_loss = 0
        for i in range(batch_size):
            pos_score = positive_pairs[i]
            neg_scores = similarity_matrix[:, i]
            neg_weights = weights[:, i]
            
            # 加权负样本损失
            weighted_neg_exp = torch.sum(neg_weights * torch.exp(neg_scores))
            t2i_loss += -torch.log(torch.exp(pos_score) / (torch.exp(pos_score) + weighted_neg_exp))
        
        return (i2t_loss + t2i_loss) / (2 * batch_size)


class AdaptiveMarginLoss(nn.Module):
    """自适应边界损失 - 根据样本难度动态调整边界"""
    
    def __init__(self, temperature: float = 0.07, base_margin: float = 0.3, 
                 margin_scale: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.base_margin = base_margin
        self.margin_scale = margin_scale
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        计算自适应边界损失
        """
        batch_size = image_features.size(0)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(image_features, text_features.T) / self.temperature
        
        # 获取正样本分数
        positive_scores = similarity_matrix.diag()
        
        # 计算样本难度（基于正负样本分数差）
        with torch.no_grad():
            difficulties = []
            for i in range(batch_size):
                neg_scores = torch.cat([similarity_matrix[i, :i], similarity_matrix[i, i+1:]])
                max_neg_score = torch.max(neg_scores)
                difficulty = torch.clamp(max_neg_score - positive_scores[i] + self.base_margin, min=0)
                difficulties.append(difficulty)
            difficulties = torch.stack(difficulties)
            
            # 自适应边界
            adaptive_margins = self.base_margin + self.margin_scale * difficulties
        
        # 计算损失
        loss = 0
        for i in range(batch_size):
            neg_scores = torch.cat([similarity_matrix[i, :i], similarity_matrix[i, i+1:]])
            margin_loss = torch.clamp(adaptive_margins[i] - positive_scores[i] + torch.max(neg_scores), min=0)
            loss += margin_loss
        
        return loss / batch_size


class HybridLoss(nn.Module):
    """混合损失函数"""
    
    def __init__(self, temperature: float = 0.07, clip_weight: float = 0.7, 
                 hard_weight: float = 0.3, margin: float = 0.2, k_hard: int = 5):
        super().__init__()
        self.clip_weight = clip_weight
        self.hard_weight = hard_weight
        
        self.clip_loss = CLIPLoss(temperature)
        self.hard_loss = HardNegativeLoss(temperature, margin, k_hard)
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """计算混合损失"""
        clip_loss = self.clip_loss(image_features, text_features)
        hard_loss = self.hard_loss(image_features, text_features)
        
        return self.clip_weight * clip_loss + self.hard_weight * hard_loss


class UltrasoundFinegrainedLoss(nn.Module):
    """
    优化版超声图像细节识别损失函数
    增强对比学习效果，提高正负样本区分度
    """
    
    def __init__(self, temperature: float = 0.05,  # 降低温度增加尖锐度
                 margin: float = 0.3,  # 增加边界距离
                 hard_ratio: float = 0.5,  # 增加硬负样本比例
                 focal_gamma: float = 3.0,  # 增加焦点损失强度
                 similarity_threshold: float = 0.75,  # 降低相似度阈值
                 contrastive_scale: float = 2.0,  # 新增：对比强度缩放
                 dynamic_temp_target: float = 0.6,  # 动态温度目标正样本相似度
                 dynamic_temp_gain: float = 2.0,    # 动态温度调节强度
                 semi_hard_window: float = 0.05):   # 半难负样本窗口大小（相对正样本）
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.hard_ratio = hard_ratio
        self.focal_gamma = focal_gamma
        self.similarity_threshold = similarity_threshold
        self.contrastive_scale = contrastive_scale
        self.dynamic_temp_target = dynamic_temp_target
        self.dynamic_temp_gain = dynamic_temp_gain
        self.semi_hard_window = semi_hard_window
        
        # 基础CLIP损失
        self.clip_loss = CLIPLoss(temperature=temperature)
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        优化版损失计算，增强对比学习效果
        """
        batch_size = image_features.shape[0]
        device = image_features.device
        
        # 预处理：增强特征对比度
        image_features = self._enhance_features(image_features)
        text_features = self._enhance_features(text_features)
        
        # 验证特征有效性
        if not self._validate_features(image_features, text_features):
            # 如果特征无效，返回安全的基础损失
            return self._compute_safe_fallback_loss(image_features, text_features)
        
        # 计算各个损失组件，添加异常处理
        loss_components = {}
        
        try:
            # 1. 增强版CLIP损失
            enhanced_clip_loss = self._compute_enhanced_clip_loss(image_features, text_features)
            loss_components['clip'] = enhanced_clip_loss if self._is_valid_loss(enhanced_clip_loss) else torch.tensor(0.0, device=device, dtype=image_features.dtype)
            
            # 2. 硬负样本挖掘损失（加强版）
            hard_negative_loss = self._compute_hard_negative_loss(image_features, text_features)
            loss_components['hard_neg'] = hard_negative_loss if self._is_valid_loss(hard_negative_loss) else torch.tensor(0.0, device=device, dtype=image_features.dtype)
            
            # 3. 三元组边界损失（新增）
            triplet_loss = self._compute_triplet_loss(image_features, text_features)
            loss_components['triplet'] = triplet_loss if self._is_valid_loss(triplet_loss) else torch.tensor(0.0, device=device, dtype=image_features.dtype)
            
            # 4. 温度自适应焦点损失
            adaptive_focal_loss = self._compute_adaptive_focal_loss(image_features, text_features)
            loss_components['focal'] = adaptive_focal_loss if self._is_valid_loss(adaptive_focal_loss) else torch.tensor(0.0, device=device, dtype=image_features.dtype)
            
            # 5. 对比正则化损失（新增）
            contrastive_reg_loss = self._compute_contrastive_regularization(image_features, text_features)
            loss_components['contrastive'] = contrastive_reg_loss if self._is_valid_loss(contrastive_reg_loss) else torch.tensor(0.0, device=device, dtype=image_features.dtype)
            
        except Exception as e:
            print(f"WARNING: 损失计算出现异常: {e}")
            return self._compute_safe_fallback_loss(image_features, text_features)
        
        # 优化版损失组合（增强对比学习权重）
        total_loss = (0.3 * loss_components['clip'] + 
                     0.25 * loss_components['hard_neg'] + 
                     0.2 * loss_components['triplet'] +
                     0.15 * loss_components['focal'] + 
                     0.1 * loss_components['contrastive'])
        
        # 最终验证
        if not self._is_valid_loss(total_loss):
            print("WARNING: 总损失包含NaN/Inf，使用安全回退损失")
            return self._compute_safe_fallback_loss(image_features, text_features)
        
        return total_loss
    
    def _validate_features(self, image_features: torch.Tensor, text_features: torch.Tensor) -> bool:
        """验证特征的有效性"""
        return (not torch.isnan(image_features).any() and 
                not torch.isinf(image_features).any() and
                not torch.isnan(text_features).any() and 
                not torch.isinf(text_features).any())
    
    def _is_valid_loss(self, loss: torch.Tensor) -> bool:
        """检查损失值是否有效"""
        return not (torch.isnan(loss).any() or torch.isinf(loss).any())
    
    def _compute_safe_fallback_loss(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """计算安全的回退损失（简单的CLIP损失）"""
        # 重新归一化特征
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        # 简单的相似度计算
        sim = torch.matmul(image_features, text_features.T)
        sim = torch.clamp(sim, min=-10.0, max=10.0)
        
        # 使用固定温度
        logits = sim / 0.07
        labels = torch.arange(sim.size(0), device=sim.device, dtype=torch.long)
        
        # 计算交叉熵损失
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def _enhance_features(self, features: torch.Tensor) -> torch.Tensor:
        """增强特征对比度 - 数值稳定版本"""
        # 首先检查输入是否有异常值
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("WARNING: 输入特征中发现NaN/Inf值，使用标准归一化")
            features = torch.where(torch.isnan(features) | torch.isinf(features), 
                                 torch.zeros_like(features), features)
        
        # L2归一化前添加数值稳定性
        features_norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        # 避免除零，添加小的epsilon
        features_norm = torch.clamp(features_norm, min=1e-8)
        features = features / features_norm
        
        # 更保守的对比度增强
        scale_factor = min(self.contrastive_scale, 1.5)  # 降低缩放因子
        features = features * scale_factor
        
        # 最终检查并裁剪极端值
        features = torch.clamp(features, min=-10.0, max=10.0)
        
        return features
    
    def _compute_enhanced_clip_loss(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """增强版CLIP损失：动态温度 + 正样本对角增强 - 数值稳定版本"""
        # 计算相似度矩阵，添加数值稳定性检查
        sim = torch.matmul(image_features, text_features.T)
        
        # 检查相似度矩阵是否有异常值
        if torch.isnan(sim).any() or torch.isinf(sim).any():
            print("WARNING: 相似度矩阵包含NaN/Inf值，使用简单CLIP损失")
            labels = torch.arange(sim.size(0), device=sim.device, dtype=torch.long)
            return F.cross_entropy(sim / self.temperature, labels)
        
        batch_size = sim.size(0)
        labels = torch.arange(batch_size, device=sim.device)
        pos_sim = sim[labels, labels]

        # 更保守的动态温度计算
        pos_mean = pos_sim.mean().detach()
        # 严格限制pos_mean范围
        pos_mean_clamped = torch.clamp(pos_mean, min=-1.0, max=1.0)
        
        # 使用更稳定的温度调节机制
        temp_adjustment = torch.tanh(self.dynamic_temp_gain * (pos_mean_clamped - self.dynamic_temp_target))
        dynamic_temp = self.temperature * (1.0 + 0.1 * temp_adjustment)  # 进一步减小温度变化范围
        
        # 严格限制温度范围，确保数值稳定
        dynamic_temp = torch.clamp(dynamic_temp, min=0.01, max=0.5)

        # 计算logits，添加更严格的数值限制
        logits = sim / dynamic_temp
        logits = torch.clamp(logits, min=-20.0, max=20.0)  # 更保守的范围
        
        # 更温和的对角线增强
        diagonal_mask = torch.eye(batch_size, device=logits.device).bool()
        logits[diagonal_mask] *= 1.1  # 从1.2降低到1.1
        
        # 计算损失
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def _compute_triplet_loss(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """三元组边界损失，强化正负样本间距"""
        batch_size = image_features.shape[0]
        device = image_features.device
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(image_features, text_features.T)
        
        triplet_losses = []
        for i in range(batch_size):
            # 正样本相似度
            positive_sim = similarity_matrix[i, i]
            
            # 负样本相似度（除了正样本）
            negative_mask = torch.ones(batch_size, device=similarity_matrix.device, dtype=torch.bool)
            negative_mask[i] = False
            negative_sims = similarity_matrix[i][negative_mask]
            
            # 半难负样本：挑选接近正样本但仍低于正样本的负样本
            window_low = (positive_sim - self.semi_hard_window).item()
            semi_hard_pool = negative_sims[(negative_sims <= positive_sim) & (negative_sims >= window_low)]
            if semi_hard_pool.numel() > 0:
                hard_negative_sim = semi_hard_pool.max()
            else:
                hard_negative_sim = negative_sims.max()

            # 难度自适应margin
            difficulty = (hard_negative_sim - positive_sim).clamp(min=0).detach()
            adaptive_margin = self.margin * (1.0 + 0.5 * difficulty)
            triplet_loss = F.relu(hard_negative_sim - positive_sim + adaptive_margin)
            triplet_losses.append(triplet_loss)
        
        return torch.stack(triplet_losses).mean()
    
    def _compute_adaptive_focal_loss(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """自适应焦点损失，重点关注难分样本"""
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device, dtype=torch.long)
        
        # 计算概率
        probs = F.softmax(logits, dim=1)
        correct_probs = probs[torch.arange(batch_size, device=labels.device), labels]
        
        # 焦点权重：难样本权重更高
        focal_weights = (1 - correct_probs).pow(self.focal_gamma)
        
        # 加权交叉熵损失
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        focal_loss = focal_weights * ce_loss
        
        return focal_loss.mean()
    
    def _compute_contrastive_regularization(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """对比正则化损失，防止特征坍塌"""
        batch_size = image_features.shape[0]
        
        # 计算特征内部相似度，鼓励多样性
        img_sim = torch.matmul(image_features, image_features.T)
        txt_sim = torch.matmul(text_features, text_features.T)
        
        # 排除对角线（自相似度）
        mask = ~torch.eye(batch_size, device=image_features.device, dtype=torch.bool)
        
        # 惩罚过高的内部相似度（防止特征坍塌）
        img_reg = F.relu(img_sim[mask] - 0.5).mean()
        txt_reg = F.relu(txt_sim[mask] - 0.5).mean()
        
        return (img_reg + txt_reg) / 2
    
    def _compute_hard_negative_loss(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """增强的硬负样本损失 - 专为超声图像特征区分度优化 - 数值稳定版本"""
        batch_size = image_features.shape[0]
        device = image_features.device
        eps = 1e-8
        
        # L2归一化特征，确保数值稳定性
        image_features_norm = F.normalize(image_features, p=2, dim=1)
        text_features_norm = F.normalize(text_features, p=2, dim=1)
        
        # 数值稳定性检查
        if torch.isnan(image_features_norm).any() or torch.isinf(image_features_norm).any():
            image_features_norm = F.normalize(image_features, p=2, dim=1)
        if torch.isnan(text_features_norm).any() or torch.isinf(text_features_norm).any():
            text_features_norm = F.normalize(text_features, p=2, dim=1)
        
        # 计算相似度矩阵
        sim = torch.matmul(image_features_norm, text_features_norm.T)
        labels = torch.arange(batch_size, device=device)
        pos_sim = sim[labels, labels]

        # 增强的动态温度调节 - 数值稳定版本
        pos_mean = pos_sim.mean().detach().item()
        neg_similarities = sim[~torch.eye(batch_size, dtype=torch.bool, device=device)]
        neg_mean = neg_similarities.mean().detach().item()
        similarity_gap = pos_mean - neg_mean  # float
        
        # 根据正负样本差距动态调整温度
        if similarity_gap < 0.1:  # 正负样本过于相近
            temp_factor = 0.3  # 激进降温
        elif similarity_gap < 0.2:
            temp_factor = 0.5
        elif similarity_gap < 0.3:
            temp_factor = 0.7
        else:
            temp_factor = 1.0
            
        # 使用Python的限幅以避免向torch.clamp传入float
        dynamic_temp = self.temperature * temp_factor  # float
        dynamic_temp = max(0.01, min(0.2, dynamic_temp))
        logits = sim / dynamic_temp  # float会自动广播
        # 数值稳定性：限制logits范围
        logits = torch.clamp(logits, min=-50.0, max=50.0)

        # 增强的半难负样本挖掘
        hard_negatives_loss = 0.0
        contrastive_loss = 0.0
        
        for i in range(batch_size):
            scores = logits[i]
            pos_score = scores[i]
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask[i] = False
            neg_scores = scores[mask]

            # 多层次难度采样
            # 1. 超难负样本：相似度非常接近正样本
            super_hard_threshold = pos_score - 0.1
            super_hard_mask = neg_scores >= super_hard_threshold
            
            # 2. 难负样本：在半难窗口内
            window_low = pos_score - self.semi_hard_window
            hard_mask = (neg_scores < super_hard_threshold) & (neg_scores >= window_low)
            
            # 3. 中等负样本：较为困难但不在半难窗口
            medium_threshold = pos_score - 0.3
            medium_mask = (neg_scores < window_low) & (neg_scores >= medium_threshold)
            
            # 收集不同难度的负样本
            selected_negs = []
            weights = []
            
            # 超难负样本（权重最高）
            if super_hard_mask.any():
                super_hard_negs = neg_scores[super_hard_mask]
                num_super = min(len(super_hard_negs), max(1, int(0.2 * batch_size)))
                top_super, _ = torch.topk(super_hard_negs, num_super)
                selected_negs.append(top_super)
                weights.extend([3.0] * len(top_super))
            
            # 难负样本
            if hard_mask.any():
                hard_negs = neg_scores[hard_mask]
                num_hard = min(len(hard_negs), max(1, int(0.3 * batch_size)))
                top_hard, _ = torch.topk(hard_negs, num_hard)
                selected_negs.append(top_hard)
                weights.extend([2.0] * len(top_hard))
            
            # 中等负样本（随机采样）
            if medium_mask.any():
                medium_negs = neg_scores[medium_mask]
                num_medium = min(len(medium_negs), max(1, int(0.2 * batch_size)))
                if len(medium_negs) > num_medium:
                    indices = torch.randperm(len(medium_negs))[:num_medium]
                    medium_selected = medium_negs[indices]
                else:
                    medium_selected = medium_negs
                selected_negs.append(medium_selected)
                weights.extend([1.0] * len(medium_selected))
            
            # 如果没有找到足够的困难负样本，补充最相似的负样本
            if not selected_negs or sum(len(negs) for negs in selected_negs) < 2:
                k_fallback = max(2, int(0.2 * batch_size))
                top_negs, _ = torch.topk(neg_scores, min(k_fallback, len(neg_scores)))
                selected_negs.append(top_negs)
                weights.extend([1.5] * len(top_negs))
            
            # 合并所有选中的负样本
            if selected_negs:
                all_selected_negs = torch.cat(selected_negs)
                sample_weights = torch.tensor(weights, device=image_features.device, dtype=torch.float32)
                
                # 加权边际损失（稳定版）
                margin_losses = F.relu(all_selected_negs - pos_score + self.margin)
                weighted_margin_loss = (margin_losses * sample_weights).mean()
                
                # 稳定的对比损失：log-sum-exp 技巧避免exp溢出
                max_val = torch.max(torch.cat([pos_score.unsqueeze(0), all_selected_negs]))
                pos_scaled = torch.exp(pos_score - max_val)
                neg_scaled = torch.exp(all_selected_negs - max_val) * sample_weights
                denom = pos_scaled + neg_scaled.sum() + eps
                contrastive_sample_loss = -torch.log((pos_scaled / denom).clamp_min(eps))
                
                hard_negatives_loss += weighted_margin_loss
                contrastive_loss += contrastive_sample_loss

        # 组合损失
        hard_negatives_loss = hard_negatives_loss / batch_size
        contrastive_loss = contrastive_loss / batch_size
        
        # 总损失：边际损失 + 对比损失
        total_loss = 0.6 * hard_negatives_loss + 0.4 * contrastive_loss
        
        return total_loss
    
    def _compute_margin_aware_loss(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """计算边界感知损失，强调细节差异"""
        batch_size = image_features.shape[0]
        
        # 计算特征间的L2距离
        img_distances = torch.cdist(image_features, image_features, p=2)
        text_distances = torch.cdist(text_features, text_features, p=2)
        
        # 找到相似的样本对（距离小但应该有不同语义）
        margin_loss = 0
        count = 0
        
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                img_dist = img_distances[i, j]
                text_dist = text_distances[i, j]
                
                # 如果图像相似但文本差异大，增加惩罚
                if img_dist < self.similarity_threshold and text_dist > self.similarity_threshold:
                    # 希望相似图像的文本表示也相似
                    margin_loss += torch.clamp(text_dist - img_dist - self.margin, min=0)
                    count += 1
                elif img_dist > self.similarity_threshold and text_dist < self.similarity_threshold:
                    # 希望不同图像的文本表示也不同
                    margin_loss += torch.clamp(img_dist - text_dist - self.margin, min=0)
                    count += 1
        
        return margin_loss / max(count, 1)
    
    def _compute_focal_loss(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """计算焦点损失，专注于难样本"""
        similarity_matrix = torch.matmul(image_features, text_features.T) / self.temperature
        batch_size = similarity_matrix.shape[0]
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        # 计算标准交叉熵
        log_probs_i2t = F.log_softmax(similarity_matrix, dim=1)
        log_probs_t2i = F.log_softmax(similarity_matrix.T, dim=1)
        
        ce_loss_i2t = F.nll_loss(log_probs_i2t, labels, reduction='none')
        ce_loss_t2i = F.nll_loss(log_probs_t2i, labels, reduction='none')
        
        # 计算概率
        pt_i2t = torch.exp(-ce_loss_i2t)
        pt_t2i = torch.exp(-ce_loss_t2i)
        
        # 应用焦点权重
        focal_weight_i2t = (1 - pt_i2t) ** self.focal_gamma
        focal_weight_t2i = (1 - pt_t2i) ** self.focal_gamma
        
        focal_loss_i2t = focal_weight_i2t * ce_loss_i2t
        focal_loss_t2i = focal_weight_t2i * ce_loss_t2i
        
        return (focal_loss_i2t.mean() + focal_loss_t2i.mean()) / 2


class UltrasoundSimpleLoss(nn.Module):
    """
    超声图像简单高效损失函数
    支持硬标签和软标签两种模式
    基于CLIP损失 + 软标签学习 + 简单hard negative
    """
    
    def __init__(self, temperature: float = 0.1, 
                 label_smoothing: float = 0.1,
                 hard_negative_ratio: float = 0.3,
                 margin: float = 0.2,
                 use_soft_label: bool = True,
                 soft_label_temp: float = 2.0,
                 soft_label_weight: float = 0.6):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.hard_negative_ratio = hard_negative_ratio
        self.margin = margin
        self.use_soft_label = use_soft_label
        self.soft_label_temp = soft_label_temp
        self.soft_label_weight = soft_label_weight
        
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        支持软标签的损失计算
        Args:
            image_features: 图像特征 [batch_size, embed_dim]
            text_features: 文本特征 [batch_size, embed_dim]
        Returns:
            loss: 总损失
        """
        batch_size = image_features.size(0)
        device = image_features.device
        
        # 确保特征已归一化
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(image_features, text_features.T) / self.temperature
        
        # 创建硬标签
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        
        if self.use_soft_label:
            # 软标签模式：结合硬标签和软标签
            total_loss = self._compute_soft_hard_combined_loss(sim_matrix, labels, image_features, text_features)
        else:
            # 硬标签模式：原来的实现
            total_loss = self._compute_hard_label_loss(sim_matrix, labels)
        
        return total_loss
    
    def _compute_soft_hard_combined_loss(self, sim_matrix: torch.Tensor, labels: torch.Tensor, 
                                       image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """计算软硬标签结合的损失"""
        batch_size = sim_matrix.size(0)
        
        # 1. 硬标签损失（带标签平滑）
        if self.label_smoothing > 0:
            loss_i2t = F.cross_entropy(sim_matrix, labels, label_smoothing=self.label_smoothing)
            loss_t2i = F.cross_entropy(sim_matrix.T, labels, label_smoothing=self.label_smoothing)
        else:
            loss_i2t = F.cross_entropy(sim_matrix, labels)
            loss_t2i = F.cross_entropy(sim_matrix.T, labels)
        
        hard_loss = (loss_i2t + loss_t2i) / 2
        
        # 2. 软标签损失
        soft_loss = self._compute_soft_label_loss(sim_matrix, image_features, text_features)
        
        # 3. 简单的hard negative损失
        hard_neg_loss = self._compute_simple_hard_negative_loss(sim_matrix, labels)
        
        # 4. 组合损失：软标签 + 硬标签 + hard negative
        total_loss = (self.soft_label_weight * soft_loss + 
                     (1 - self.soft_label_weight) * hard_loss + 
                     0.1 * hard_neg_loss)
        
        return total_loss
    
    def _compute_soft_label_loss(self, sim_matrix: torch.Tensor, 
                                image_features: torch.Tensor, 
                                text_features: torch.Tensor) -> torch.Tensor:
        """计算软标签损失（参考CLIP_breast的实现）"""
        # 统一到FP32，提高数值稳定性
        img = image_features.float()
        txt = text_features.float()
        
        # 教师分布：基于模态内相似度构造软标签
        with torch.no_grad():
            image_sim = torch.matmul(img, img.T)                    # [B, B]
            text_sim = torch.matmul(txt, txt.T)                     # [B, B]
            # 组合软标签，应用温度控制
            targets = F.softmax((image_sim + text_sim) / 2.0 * self.soft_label_temp, dim=-1)
        
        # 学生预测：跨模态相似度
        logits_i2t = sim_matrix * self.temperature  # 恢复原始logits
        logits_t2i = sim_matrix.T * self.temperature
        
        # 软标签交叉熵
        loss_i = self._soft_cross_entropy(logits_i2t, targets)
        loss_t = self._soft_cross_entropy(logits_t2i, targets.T)
        
        return (loss_i + loss_t) / 2
    
    def _soft_cross_entropy(self, predicted_logits: torch.Tensor, 
                           target_prob: torch.Tensor, 
                           reduction: str = 'mean') -> torch.Tensor:
        """软标签交叉熵实现（参考CLIP_breast）"""
        # 统一到FP32，提高数值稳定性
        predicted_logits = predicted_logits.float()
        target_prob = target_prob.float()
        
        # log_softmax本身是数值稳定实现
        log_prob = F.log_softmax(predicted_logits, dim=-1)      # [B, B]
        loss = -(target_prob * log_prob).sum(dim=-1)            # [B]
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def _compute_hard_label_loss(self, sim_matrix: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算硬标签损失（原来的实现）"""
        # 1. 基础CLIP损失（带标签平滑）
        if self.label_smoothing > 0:
            loss_i2t = F.cross_entropy(sim_matrix, labels, label_smoothing=self.label_smoothing)
            loss_t2i = F.cross_entropy(sim_matrix.T, labels, label_smoothing=self.label_smoothing)
        else:
            loss_i2t = F.cross_entropy(sim_matrix, labels)
            loss_t2i = F.cross_entropy(sim_matrix.T, labels)
        
        clip_loss = (loss_i2t + loss_t2i) / 2
        
        # 2. 简单的hard negative损失
        hard_neg_loss = self._compute_simple_hard_negative_loss(sim_matrix, labels)
        
        # 3. 组合损失（简单加权）
        total_loss = 0.8 * clip_loss + 0.2 * hard_neg_loss
        
        return total_loss
    
    def _compute_simple_hard_negative_loss(self, sim_matrix: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算简单的hard negative损失"""
        batch_size = sim_matrix.size(0)
        
        # 获取正样本相似度
        pos_sim = sim_matrix[torch.arange(batch_size), labels]
        
        loss = 0.0
        for i in range(batch_size):
            # 获取负样本相似度（排除正样本）
            neg_mask = torch.ones(batch_size, dtype=torch.bool, device=sim_matrix.device)
            neg_mask[i] = False
            neg_sims = sim_matrix[i][neg_mask]
            
            # 选择最难的几个负样本
            num_hard = max(1, int(self.hard_negative_ratio * (batch_size - 1)))
            hard_negs, _ = torch.topk(neg_sims, num_hard)
            
            # 计算margin loss
            for hard_neg in hard_negs:
                margin_loss = F.relu(hard_neg - pos_sim[i] + self.margin)
                loss += margin_loss
        
        return loss / batch_size


class SimplifiedClipStableLoss(nn.Module):
    """
    简化稳定的CLIP损失函数 - 对照组实验专用
    基于成功CLIP实现的简化版本，数值稳定性增强
    """
    
    def __init__(self, 
                 temperature: float = 0.07,
                 label_smoothing: float = 0.1,
                 numerical_stability: dict = None):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        
        # 数值稳定性配置
        if numerical_stability is None:
            numerical_stability = {}
        
        self.feature_clamp_range = numerical_stability.get('feature_clamp_range', [-10.0, 10.0])
        self.similarity_clamp_range = numerical_stability.get('similarity_clamp_range', [-50.0, 50.0])
        self.safe_normalize_eps = numerical_stability.get('safe_normalize_eps', 1e-8)
        self.gradient_clip_loss = numerical_stability.get('gradient_clip_loss', True)
        
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        计算简化稳定的CLIP损失
        """
        batch_size = image_features.size(0)
        device = image_features.device
        
        # 安全归一化和裁剪
        image_features = self._safe_normalize_and_clamp(image_features)
        text_features = self._safe_normalize_and_clamp(text_features)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(image_features, text_features.T) / self.temperature
        
        # 裁剪相似度防止数值溢出
        similarity_matrix = torch.clamp(similarity_matrix, 
                                      min=self.similarity_clamp_range[0],
                                      max=self.similarity_clamp_range[1])
        
        # 创建标签
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        
        # 计算损失
        if self.label_smoothing > 0:
            loss_i2t = F.cross_entropy(similarity_matrix, labels, 
                                     label_smoothing=self.label_smoothing)
            loss_t2i = F.cross_entropy(similarity_matrix.T, labels, 
                                     label_smoothing=self.label_smoothing)
        else:
            loss_i2t = F.cross_entropy(similarity_matrix, labels)
            loss_t2i = F.cross_entropy(similarity_matrix.T, labels)
        
        total_loss = (loss_i2t + loss_t2i) / 2
        
        # 损失值裁剪（如果启用）
        if self.gradient_clip_loss:
            total_loss = torch.clamp(total_loss, min=0.0, max=10.0)
        
        # 检查损失有效性
        if not torch.isfinite(total_loss):
            print("WARNING: SimplifiedClipStableLoss computed non-finite loss, using fallback")
            return torch.tensor(1.0, device=device, requires_grad=True)
        
        return total_loss
    
    def _safe_normalize_and_clamp(self, features: torch.Tensor) -> torch.Tensor:
        """安全的归一化和裁剪"""
        # 先裁剪极端值
        features = torch.clamp(features, 
                             min=self.feature_clamp_range[0],
                             max=self.feature_clamp_range[1])
        
        # 安全L2归一化
        norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=self.safe_normalize_eps)
        features = features / norm
        
        return features


class UltrasoundAdvancedLoss(nn.Module):
    """
    高级超声图像损失函数
    包含自适应温度、多尺度损失、语义一致性和特征多样性
    """
    
    def __init__(self, 
                 temperature: float = 0.05,
                 use_adaptive_temperature: bool = True,
                 adaptive_temp_config: dict = None,
                 hard_negative_mining: dict = None,
                 multi_scale_loss: dict = None,
                 semantic_consistency: dict = None,
                 feature_diversity: dict = None,
                 label_smoothing: float = 0.2,
                 numerical_stability: dict = None):
        super().__init__()
        
        self.base_temperature = temperature
        self.current_temperature = temperature
        self.use_adaptive_temperature = use_adaptive_temperature
        self.label_smoothing = label_smoothing
        
        # 自适应温度配置
        if adaptive_temp_config is None:
            adaptive_temp_config = {}
        self.min_temp = adaptive_temp_config.get('min_temp', 0.03)
        self.max_temp = adaptive_temp_config.get('max_temp', 0.15)
        self.adaptation_rate = adaptive_temp_config.get('adaptation_rate', 0.1)
        
        # 硬负样本挖掘配置
        if hard_negative_mining is None:
            hard_negative_mining = {}
        self.hard_mining_enabled = hard_negative_mining.get('enabled', True)
        self.hard_ratio = hard_negative_mining.get('ratio', 0.4)
        self.margin = hard_negative_mining.get('margin', 0.3)
        self.semi_hard_ratio = hard_negative_mining.get('semi_hard_ratio', 0.6)
        
        # 多尺度损失配置
        if multi_scale_loss is None:
            multi_scale_loss = {}
        self.multi_scale_enabled = multi_scale_loss.get('enabled', True)
        self.scales = multi_scale_loss.get('scales', [0.5, 0.75, 1.0])
        self.scale_weights = multi_scale_loss.get('weights', [0.2, 0.3, 0.5])
        
        # 语义一致性配置
        if semantic_consistency is None:
            semantic_consistency = {}
        self.semantic_enabled = semantic_consistency.get('enabled', True)
        self.semantic_weight = semantic_consistency.get('weight', 0.15)
        
        # 特征多样性配置
        if feature_diversity is None:
            feature_diversity = {}
        self.diversity_enabled = feature_diversity.get('enabled', True)
        self.diversity_weight = feature_diversity.get('diversity_weight', 0.1)
        self.regularization_strength = feature_diversity.get('regularization_strength', 0.05)
        
        # 数值稳定性配置
        if numerical_stability is None:
            numerical_stability = {}
        self.feature_clamp_range = numerical_stability.get('feature_clamp_range', [-5.0, 5.0])
        self.similarity_clamp_range = numerical_stability.get('similarity_clamp_range', [-20.0, 20.0])
        self.safe_normalize_eps = numerical_stability.get('safe_normalize_eps', 1e-8)
        
        # 训练步数计数器
        self.training_step = 0
        
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """计算高级损失"""
        self.training_step += 1
        batch_size = image_features.size(0)
        device = image_features.device
        
        # 安全归一化
        image_features = self._safe_normalize(image_features)
        text_features = self._safe_normalize(text_features)
        
        # 自适应温度调整
        if self.use_adaptive_temperature:
            self._update_adaptive_temperature(image_features, text_features)
        
        total_loss = 0.0
        loss_components = {}
        
        # 1. 基础CLIP损失 (主要损失)
        clip_loss = self._compute_clip_loss(image_features, text_features)
        loss_components['clip'] = clip_loss
        total_loss += 0.6 * clip_loss
        
        # 2. 硬负样本挖掘损失
        if self.hard_mining_enabled:
            hard_neg_loss = self._compute_hard_negative_loss(image_features, text_features)
            loss_components['hard_negative'] = hard_neg_loss
            total_loss += 0.2 * hard_neg_loss
        
        # 3. 多尺度损失
        if self.multi_scale_enabled:
            multi_scale_loss = self._compute_multi_scale_loss(image_features, text_features)
            loss_components['multi_scale'] = multi_scale_loss
            total_loss += 0.1 * multi_scale_loss
        
        # 4. 语义一致性损失 (大幅降低权重)
        if self.semantic_enabled:
            semantic_loss = self._compute_semantic_consistency(image_features, text_features)
            loss_components['semantic'] = semantic_loss
            total_loss += 0.05 * semantic_loss  # 从0.15降低到0.05
        
        # 5. 特征多样性损失 (大幅降低权重)
        if self.diversity_enabled:
            diversity_loss = self._compute_feature_diversity(image_features, text_features)
            loss_components['diversity'] = diversity_loss
            total_loss += 0.05 * diversity_loss  # 从0.1降低到0.05
        
        # 检查损失有效性
        if not torch.isfinite(total_loss):
            print(f"WARNING: UltrasoundAdvancedLoss非有限值，使用fallback")
            return self._compute_fallback_loss(image_features, text_features)
        
        return total_loss
    
    def _safe_normalize(self, features: torch.Tensor) -> torch.Tensor:
        """安全归一化"""
        features = torch.clamp(features, min=self.feature_clamp_range[0], max=self.feature_clamp_range[1])
        norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=self.safe_normalize_eps)
        return features / norm
    
    def _update_adaptive_temperature(self, image_features: torch.Tensor, text_features: torch.Tensor):
        """自适应温度调整"""
        with torch.no_grad():
            # 计算特征相似度分布
            similarity = torch.matmul(image_features, text_features.T)
            pos_sim = similarity.diag().mean()
            neg_sim = similarity[~torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)].mean()
            
            # 根据正负样本差距调整温度
            gap = pos_sim - neg_sim
            if gap < 0.1:  # 差距太小，降低温度增强区分度
                target_temp = self.min_temp
            elif gap > 0.5:  # 差距太大，提高温度防止过度锐化
                target_temp = self.max_temp
            else:
                # 线性插值
                target_temp = self.min_temp + (self.max_temp - self.min_temp) * (gap - 0.1) / 0.4
            
            # 平滑更新
            self.current_temperature = (1 - self.adaptation_rate) * self.current_temperature + self.adaptation_rate * target_temp
            self.current_temperature = max(self.min_temp, min(self.max_temp, self.current_temperature))
    
    def _compute_clip_loss(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """计算CLIP损失"""
        similarity = torch.matmul(image_features, text_features.T) / self.current_temperature
        similarity = torch.clamp(similarity, min=self.similarity_clamp_range[0], max=self.similarity_clamp_range[1])
        
        batch_size = similarity.size(0)
        labels = torch.arange(batch_size, device=similarity.device, dtype=torch.long)
        
        if self.label_smoothing > 0:
            loss_i2t = F.cross_entropy(similarity, labels, label_smoothing=self.label_smoothing)
            loss_t2i = F.cross_entropy(similarity.T, labels, label_smoothing=self.label_smoothing)
        else:
            loss_i2t = F.cross_entropy(similarity, labels)
            loss_t2i = F.cross_entropy(similarity.T, labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def _compute_hard_negative_loss(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """计算硬负样本损失"""
        batch_size = image_features.size(0)
        similarity = torch.matmul(image_features, text_features.T)
        
        total_loss = 0.0
        for i in range(batch_size):
            pos_sim = similarity[i, i]
            
            # 获取负样本
            neg_mask = torch.ones(batch_size, dtype=torch.bool, device=similarity.device)
            neg_mask[i] = False
            neg_sims = similarity[i][neg_mask]
            
            # 硬负样本和半难负样本
            hard_threshold = pos_sim - self.margin
            hard_negs = neg_sims[neg_sims >= hard_threshold]
            
            if len(hard_negs) > 0:
                num_hard = max(1, int(self.hard_ratio * len(neg_sims)))
                selected_hard = torch.topk(hard_negs, min(num_hard, len(hard_negs)))[0]
                
                # 计算损失
                for hard_neg in selected_hard:
                    loss = F.relu(hard_neg - pos_sim + self.margin)
                    total_loss += loss
        
        return total_loss / batch_size
    
    def _compute_multi_scale_loss(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """计算多尺度损失"""
        total_loss = 0.0
        
        for scale, weight in zip(self.scales, self.scale_weights):
            if scale == 1.0:
                # 原始尺度
                scaled_img_feat = image_features
                scaled_txt_feat = text_features
            else:
                # 缩放特征
                feat_dim = image_features.size(-1)
                scaled_dim = int(feat_dim * scale)
                
                # 简单的特征缩放（通过线性层模拟）
                scaled_img_feat = image_features[:, :scaled_dim]
                scaled_txt_feat = text_features[:, :scaled_dim]
                
                # 重新归一化
                scaled_img_feat = F.normalize(scaled_img_feat, p=2, dim=-1)
                scaled_txt_feat = F.normalize(scaled_txt_feat, p=2, dim=-1)
            
            # 计算该尺度的损失
            scale_loss = self._compute_clip_loss(scaled_img_feat, scaled_txt_feat)
            total_loss += weight * scale_loss
        
        return total_loss
    
    def _compute_semantic_consistency(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """计算语义一致性损失 - 修复版"""
        batch_size = image_features.size(0)
        
        # 计算模态内相似性，添加数值稳定性
        img_sim = torch.matmul(image_features, image_features.T)
        txt_sim = torch.matmul(text_features, text_features.T)
        
        # 裁剪相似度矩阵防止数值爆炸
        img_sim = torch.clamp(img_sim, min=-10.0, max=10.0)
        txt_sim = torch.clamp(txt_sim, min=-10.0, max=10.0)
        
        # 只计算对角线外的相似性差异，避免对角线自相似度的影响
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=image_features.device)
        img_off_diag = img_sim[mask]
        txt_off_diag = txt_sim[mask]
        
        # 使用更稳定的损失计算
        consistency_loss = F.mse_loss(img_off_diag, txt_off_diag)
        
        # 进一步缩放损失值
        return consistency_loss * 0.1
    
    def _compute_feature_diversity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """计算特征多样性损失，防止特征坍塌 - 修复版"""
        batch_size = image_features.size(0)
        
        # 使用更简单稳定的多样性损失：计算特征间的平均相似度
        # 避免协方差矩阵计算的不稳定性
        
        # 计算图像特征间的相似度
        img_sim = torch.matmul(image_features, image_features.T)
        img_sim = torch.clamp(img_sim, min=-10.0, max=10.0)
        
        # 计算文本特征间的相似度
        txt_sim = torch.matmul(text_features, text_features.T)
        txt_sim = torch.clamp(txt_sim, min=-10.0, max=10.0)
        
        # 排除对角线，计算非对角线元素的平均相似度
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=image_features.device)
        img_off_diag = img_sim[mask]
        txt_off_diag = txt_sim[mask]
        
        # 多样性损失：惩罚过高的特征间相似度
        img_diversity_loss = F.relu(img_off_diag - 0.5).mean()
        txt_diversity_loss = F.relu(txt_off_diag - 0.5).mean()
        
        return (img_diversity_loss + txt_diversity_loss) / 2
    
    def _compute_fallback_loss(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """回退损失函数"""
        similarity = torch.matmul(image_features, text_features.T) / 0.07
        batch_size = similarity.size(0)
        labels = torch.arange(batch_size, device=similarity.device, dtype=torch.long)
        
        loss_i2t = F.cross_entropy(similarity, labels)
        loss_t2i = F.cross_entropy(similarity.T, labels)
        
        return (loss_i2t + loss_t2i) / 2


class LossFactory:
    """损失函数工厂类"""
    
    @staticmethod
    def create_loss(loss_type: str, **kwargs) -> nn.Module:
        """
        创建损失函数
        Args:
            loss_type: 损失函数类型
            **kwargs: 损失函数参数
        Returns:
            损失函数实例
        """
        if loss_type == 'clip':
            return CLIPLoss(**kwargs)
        elif loss_type == 'hard_negative':
            return HardNegativeLoss(**kwargs)
        elif loss_type == 'focal':
            return FocalLoss(**kwargs)
        elif loss_type == 'hybrid':
            return HybridLoss(**kwargs)
        elif loss_type == 'soft_target':
            return SoftTargetLoss(**kwargs)
        elif loss_type == 'ranking':
            return RankingLoss(**kwargs)
        elif loss_type == 'contrastive':
            return ContrastiveLoss(**kwargs)
        elif loss_type == 'adaptive_margin':
            return AdaptiveMarginLoss(**kwargs)
        elif loss_type == 'ultrasound_finegrained':
            return UltrasoundFinegrainedLoss(**kwargs)
        elif loss_type == 'ultrasound_simple':
            return UltrasoundSimpleLoss(**kwargs)
        elif loss_type == 'simplified_clip_stable':
            return SimplifiedClipStableLoss(**kwargs)
        elif loss_type == 'ultrasound_advanced':
            return UltrasoundAdvancedLoss(**kwargs)
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}")
