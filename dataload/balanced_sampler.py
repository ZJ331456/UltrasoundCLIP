"""
平衡批次采样器
确保每个batch包含多个类别的样本，避免类别不平衡问题
"""

import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import torch
from torch.utils.data import Sampler
import numpy as np

logger = logging.getLogger(__name__)


class BalancedBatchSampler(Sampler):
    """
    平衡批次采样器
    
    确保每个batch包含指定数量的不同类别样本，避免某些大类主导训练过程。
    支持小类过采样和数据增强策略。
    """
    
    def __init__(self, 
                 dataset,
                 batch_size: int = 64,
                 num_classes_per_batch: int = 8,
                 samples_per_class: Optional[int] = None,
                 oversample_small_classes: bool = True,
                 max_oversample_ratio: float = 3.0,
                 shuffle: bool = True,
                 drop_last: bool = True,
                 class_key: str = 'class',
                 text_key: str = 'text'):
        """
        初始化平衡批次采样器
        
        Args:
            dataset: 数据集对象
            batch_size: 批次大小
            num_classes_per_batch: 每个batch包含的类别数量
            samples_per_class: 每个类别的样本数量（如果为None，则自动计算）
            oversample_small_classes: 是否对小类进行过采样
            max_oversample_ratio: 最大过采样比例
            shuffle: 是否打乱数据
            drop_last: 是否丢弃最后一个不完整的batch
            class_key: 类别信息的键名
            text_key: 文本信息的键名（用于提取类别）
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes_per_batch = num_classes_per_batch
        self.oversample_small_classes = oversample_small_classes
        self.max_oversample_ratio = max_oversample_ratio
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.class_key = class_key
        self.text_key = text_key
        
        # 自动计算每个类别的样本数量
        if samples_per_class is None:
            self.samples_per_class = batch_size // num_classes_per_batch
        else:
            self.samples_per_class = int(samples_per_class)
        
        # 确保参数合理性
        if self.samples_per_class * num_classes_per_batch > batch_size:
            logger.warning(f"每个类别样本数({self.samples_per_class}) × 类别数({num_classes_per_batch}) > batch_size({batch_size})")
            self.samples_per_class = batch_size // num_classes_per_batch
            logger.info(f"自动调整每个类别样本数为: {self.samples_per_class}")
        
        # 确保samples_per_class至少为1
        if self.samples_per_class < 1:
            self.samples_per_class = 1
            logger.warning(f"samples_per_class调整为1")
        
        # 分析数据集中的类别分布
        self.class_indices = self._analyze_class_distribution()
        self.class_names = list(self.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        # 计算类别统计信息
        self.class_counts = {cls: len(indices) for cls, indices in self.class_indices.items()}
        self.min_class_count = min(self.class_counts.values())
        self.max_class_count = max(self.class_counts.values())
        
        # 生成平衡的样本索引
        self.balanced_indices = self._generate_balanced_indices()
        
        logger.info(f"平衡批次采样器初始化完成:")
        logger.info(f"  - 总类别数: {self.num_classes}")
        logger.info(f"  - 每个batch类别数: {self.num_classes_per_batch}")
        logger.info(f"  - 每个类别样本数: {self.samples_per_class}")
        logger.info(f"  - 批次大小: {self.batch_size}")
        logger.info(f"  - 类别分布: 最小{self.min_class_count}, 最大{self.max_class_count}")
        logger.info(f"  - 过采样: {'启用' if oversample_small_classes else '禁用'}")
    
    def _analyze_class_distribution(self) -> Dict[str, List[int]]:
        """分析数据集中的类别分布"""
        class_indices = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            try:
                # 获取样本数据
                sample = self.dataset[idx]
                
                # 提取类别信息
                class_name = self._extract_class_name(sample)
                
                if class_name:
                    class_indices[class_name].append(idx)
                else:
                    # 如果无法提取类别，使用默认类别
                    class_indices['unknown'].append(idx)
                    
            except Exception as e:
                logger.warning(f"分析样本 {idx} 时出错: {e}")
                class_indices['error'].append(idx)
        
        # 过滤掉样本数太少的类别（少于samples_per_class的类别）
        filtered_class_indices = {}
        for class_name, indices in class_indices.items():
            if len(indices) >= self.samples_per_class:
                filtered_class_indices[class_name] = indices
            else:
                logger.warning(f"类别 '{class_name}' 样本数({len(indices)})少于要求({self.samples_per_class})，跳过")
        
        if not filtered_class_indices:
            logger.error("没有找到足够的类别样本，回退到随机采样")
            # 回退策略：将所有样本归为一个类别
            all_indices = list(range(len(self.dataset)))
            filtered_class_indices['all'] = all_indices
        
        return filtered_class_indices
    
    def _extract_class_name(self, sample: Dict[str, Any]) -> Optional[str]:
        """从样本中提取类别名称"""
        # 方法1: 直接从class_key获取
        if self.class_key in sample:
            return str(sample[self.class_key])
        
        # 方法2: 从文本中提取类别（基于超声图像的常见类别）
        if self.text_key in sample:
            text = sample[self.text_key].lower()
            
            # 定义超声图像常见类别关键词
            class_keywords = {
                'brachial_plexus': ['brachial plexus', '臂丛神经', '臂丛'],
                'carotid_artery': ['carotid artery', '颈动脉', 'carotid'],
                'thyroid': ['thyroid', '甲状腺', 'thyroid gland'],
                'liver': ['liver', '肝脏', 'hepatic'],
                'kidney': ['kidney', '肾脏', 'renal'],
                'heart': ['heart', '心脏', 'cardiac', 'cardio'],
                'lung': ['lung', '肺部', 'pulmonary'],
                'breast': ['breast', '乳腺', 'mammary'],
                'ovary': ['ovary', '卵巢', 'ovarian'],
                'uterus': ['uterus', '子宫', 'uterine'],
                'prostate': ['prostate', '前列腺', 'prostatic'],
                'bladder': ['bladder', '膀胱', 'vesical'],
                'pancreas': ['pancreas', '胰腺', 'pancreatic'],
                'spleen': ['spleen', '脾脏', 'splenic'],
                'gallbladder': ['gallbladder', '胆囊', 'gall bladder'],
                'aorta': ['aorta', '主动脉', 'aortic'],
                'vena_cava': ['vena cava', '下腔静脉', 'inferior vena cava'],
                'portal_vein': ['portal vein', '门静脉', 'portal'],
                'bile_duct': ['bile duct', '胆管', 'biliary'],
                'lymph_node': ['lymph node', '淋巴结', 'lymphatic'],
            }
            
            # 匹配类别
            for class_name, keywords in class_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        return class_name
            
            # 如果没有匹配到，尝试提取第一个有意义的词
            words = text.split()
            if words:
                # 返回第一个词作为类别（简单策略）
                return words[0][:20]  # 限制长度
        
        return None
    
    def _generate_balanced_indices(self) -> List[int]:
        """生成平衡的样本索引"""
        balanced_indices = []
        
        # 计算需要多少个完整的batch
        total_samples_needed = len(self.dataset)
        if self.drop_last:
            # 计算能生成多少个完整的batch
            num_batches = total_samples_needed // self.batch_size
            total_samples_needed = num_batches * self.batch_size
        
        # 为每个类别准备样本索引
        class_sample_pools = {}
        for class_name, indices in self.class_indices.items():
            if self.oversample_small_classes:
                # 过采样小类
                target_count = max(
                    self.samples_per_class * (total_samples_needed // self.batch_size),
                    self.min_class_count * self.max_oversample_ratio
                )
                
                # 重复采样直到达到目标数量
                oversampled_indices = []
                while len(oversampled_indices) < target_count:
                    remaining = int(target_count - len(oversampled_indices))
                    if remaining <= 0:
                        break
                    oversampled_indices.extend(
                        random.choices(indices, k=min(remaining, len(indices)))
                    )
                
                class_sample_pools[class_name] = oversampled_indices
            else:
                class_sample_pools[class_name] = indices.copy()
        
        # 生成平衡的batch
        batch_count = 0
        while len(balanced_indices) < total_samples_needed:
            batch_indices = []
            
            # 随机选择类别
            if self.shuffle:
                selected_classes = random.sample(
                    self.class_names, 
                    min(self.num_classes_per_batch, len(self.class_names))
                )
            else:
                # 循环选择类别
                start_idx = batch_count % len(self.class_names)
                selected_classes = []
                for i in range(min(self.num_classes_per_batch, len(self.class_names))):
                    class_idx = (start_idx + i) % len(self.class_names)
                    selected_classes.append(self.class_names[class_idx])
            
            # 从每个选中的类别中采样
            for class_name in selected_classes:
                if class_name in class_sample_pools and class_sample_pools[class_name]:
                    # 从该类别的样本池中随机选择
                    class_samples = random.sample(
                        class_sample_pools[class_name],
                        min(self.samples_per_class, len(class_sample_pools[class_name]))
                    )
                    batch_indices.extend(class_samples)
            
            # 如果batch不够满，用随机样本填充
            while len(batch_indices) < self.batch_size and len(balanced_indices) + len(batch_indices) < total_samples_needed:
                # 随机选择一个类别
                random_class = random.choice(self.class_names)
                if random_class in class_sample_pools and class_sample_pools[random_class]:
                    random_sample = random.choice(class_sample_pools[random_class])
                    batch_indices.append(random_sample)
                else:
                    # 如果所有类别都用完了，随机选择任何样本
                    random_sample = random.randint(0, len(self.dataset) - 1)
                    batch_indices.append(random_sample)
            
            # 打乱batch内的顺序
            if self.shuffle:
                random.shuffle(batch_indices)
            
            balanced_indices.extend(batch_indices)
            batch_count += 1
        
        # 截断到需要的长度
        balanced_indices = balanced_indices[:total_samples_needed]
        
        logger.info(f"生成平衡索引完成: {len(balanced_indices)} 个样本，{len(balanced_indices) // self.batch_size} 个batch")
        
        return balanced_indices
    
    def __iter__(self):
        """迭代器，返回平衡的样本索引"""
        if self.shuffle:
            # 重新生成平衡索引（每次epoch都重新平衡）
            self.balanced_indices = self._generate_balanced_indices()
        
        return iter(self.balanced_indices)
    
    def __len__(self):
        """返回总样本数"""
        return len(self.balanced_indices)
    
    def get_class_distribution_stats(self) -> Dict[str, Any]:
        """获取类别分布统计信息"""
        return {
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'class_counts': self.class_counts,
            'min_class_count': self.min_class_count,
            'max_class_count': self.max_class_count,
            'class_imbalance_ratio': self.max_class_count / self.min_class_count if self.min_class_count > 0 else float('inf'),
            'samples_per_class': self.samples_per_class,
            'num_classes_per_batch': self.num_classes_per_batch,
            'total_samples': len(self.balanced_indices),
            'num_batches': len(self.balanced_indices) // self.batch_size
        }


class StratifiedBatchSampler(Sampler):
    """
    分层批次采样器
    
    确保每个batch中各类别的比例与整体数据集中的比例相近，
    但限制每个batch的类别数量。
    """
    
    def __init__(self, 
                 dataset,
                 batch_size: int = 64,
                 num_classes_per_batch: int = 8,
                 shuffle: bool = True,
                 drop_last: bool = True,
                 class_key: str = 'class',
                 text_key: str = 'text'):
        """
        初始化分层批次采样器
        
        Args:
            dataset: 数据集对象
            batch_size: 批次大小
            num_classes_per_batch: 每个batch包含的类别数量
            shuffle: 是否打乱数据
            drop_last: 是否丢弃最后一个不完整的batch
            class_key: 类别信息的键名
            text_key: 文本信息的键名
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes_per_batch = num_classes_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.class_key = class_key
        self.text_key = text_key
        
        # 分析类别分布
        self.class_indices = self._analyze_class_distribution()
        self.class_proportions = self._calculate_class_proportions()
        
        # 生成分层样本索引
        self.stratified_indices = self._generate_stratified_indices()
        
        logger.info(f"分层批次采样器初始化完成:")
        logger.info(f"  - 总类别数: {len(self.class_indices)}")
        logger.info(f"  - 每个batch类别数: {self.num_classes_per_batch}")
        logger.info(f"  - 类别比例: {self.class_proportions}")
    
    def _analyze_class_distribution(self) -> Dict[str, List[int]]:
        """分析数据集中的类别分布"""
        class_indices = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                class_name = self._extract_class_name(sample)
                
                if class_name:
                    class_indices[class_name].append(idx)
                else:
                    class_indices['unknown'].append(idx)
                    
            except Exception as e:
                logger.warning(f"分析样本 {idx} 时出错: {e}")
                class_indices['error'].append(idx)
        
        return dict(class_indices)
    
    def _extract_class_name(self, sample: Dict[str, Any]) -> Optional[str]:
        """从样本中提取类别名称"""
        # 复用BalancedBatchSampler的逻辑
        if self.class_key in sample:
            return str(sample[self.class_key])
        
        if self.text_key in sample:
            text = sample[self.text_key].lower()
            
            # 简化的类别提取逻辑
            class_keywords = {
                'brachial_plexus': ['brachial plexus', '臂丛神经'],
                'carotid_artery': ['carotid artery', '颈动脉'],
                'thyroid': ['thyroid', '甲状腺'],
                'liver': ['liver', '肝脏'],
                'kidney': ['kidney', '肾脏'],
                'heart': ['heart', '心脏'],
                'lung': ['lung', '肺部'],
                'breast': ['breast', '乳腺'],
            }
            
            for class_name, keywords in class_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        return class_name
            
            # 返回第一个词作为类别
            words = text.split()
            if words:
                return words[0][:20]
        
        return None
    
    def _calculate_class_proportions(self) -> Dict[str, float]:
        """计算各类别的比例"""
        total_samples = sum(len(indices) for indices in self.class_indices.values())
        proportions = {}
        
        for class_name, indices in self.class_indices.items():
            proportions[class_name] = len(indices) / total_samples
        
        return proportions
    
    def _generate_stratified_indices(self) -> List[int]:
        """生成分层的样本索引"""
        stratified_indices = []
        
        # 计算需要多少个batch
        total_samples = len(self.dataset)
        if self.drop_last:
            num_batches = total_samples // self.batch_size
        else:
            num_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        # 为每个batch生成样本
        for batch_idx in range(num_batches):
            batch_indices = []
            
            # 根据类别比例选择样本
            for class_name, proportion in self.class_proportions.items():
                if class_name not in self.class_indices:
                    continue
                
                # 计算该类在这个batch中应该有多少样本
                class_samples_in_batch = int(self.batch_size * proportion)
                class_samples_in_batch = min(class_samples_in_batch, len(self.class_indices[class_name]))
                class_samples_in_batch = max(1, class_samples_in_batch)  # 至少1个样本
                
                if class_samples_in_batch > 0:
                    # 随机选择样本
                    selected_samples = random.sample(
                        self.class_indices[class_name],
                        class_samples_in_batch
                    )
                    batch_indices.extend(selected_samples)
            
            # 如果batch不够满，随机填充
            while len(batch_indices) < self.batch_size:
                # 随机选择一个类别
                random_class = random.choice(list(self.class_indices.keys()))
                if self.class_indices[random_class]:
                    random_sample = random.choice(self.class_indices[random_class])
                    batch_indices.append(random_sample)
            
            # 打乱batch内的顺序
            if self.shuffle:
                random.shuffle(batch_indices)
            
            stratified_indices.extend(batch_indices)
        
        return stratified_indices
    
    def __iter__(self):
        """迭代器"""
        if self.shuffle:
            self.stratified_indices = self._generate_stratified_indices()
        
        return iter(self.stratified_indices)
    
    def __len__(self):
        """返回总样本数"""
        return len(self.stratified_indices)
