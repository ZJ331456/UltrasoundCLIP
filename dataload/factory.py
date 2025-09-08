"""
数据加载器工厂
统一管理数据集和数据加载器的创建
"""

import os
import logging
from typing import Dict, Optional, Any
from torch.utils.data import DataLoader

from .json_dataset import JSONDataset
from .jsonl_dataset import JSONLDataset
from .ultrasound_dataset import UltrasoundDataset  # 新增：导入超声数据集
from .transforms import TransformFactory
from .collator import DataCollator

logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """数据加载器工厂类"""
    
    @staticmethod
    def create_datasets(
        config: Dict[str, Any],
        max_samples: Optional[int] = None,
        validate_files: bool = True,
        cache_size: int = 500
    ) -> Dict[str, Any]:
        """
        根据配置创建数据集
        Args:
            config: 数据配置
            max_samples: 每个数据集的最大样本数
            validate_files: 是否验证文件存在性
            cache_size: 图像缓存大小
        """
        datasets = {}
        
        # 获取配置参数
        data_config = config['data']
        model_config = config['model']
        
        image_size = data_config['image_size']
        model_type = model_config['type']
        
        # 检查是否使用优化的数据增强策略
        use_optimized_augmentation = data_config.get('use_optimized_augmentation', False)
        
        # 根据模型类型选择变换
        if use_optimized_augmentation:
            # 使用优化的医学安全增强策略
            from .transforms import create_medical_safe_transforms
            
            transform_config = {
                'clip': lambda training: create_medical_safe_transforms(
                    image_size, training, "clip", data_config.get('ultrasound_augmentation', {})
                ),
                'sam': lambda training: TransformFactory.get_sam_transforms(1024),
                'qwen_vl_clip': lambda training: create_medical_safe_transforms(
                    image_size, training, "imagenet", data_config.get('ultrasound_augmentation', {})
                ),
                'medclip': lambda training: create_medical_safe_transforms(
                    image_size, training, "imagenet", data_config.get('ultrasound_augmentation', {})
                ),
                'convnext_clip': lambda training: create_medical_safe_transforms(
                    image_size, training, "imagenet", data_config.get('ultrasound_augmentation', {})
                ),
            }
            print(f"    - 使用优化的医学安全增强策略")
        else:
            # 使用标准变换策略
            transform_config = {
                'clip': lambda training: TransformFactory.get_clip_transforms(image_size, training),
                'sam': lambda training: TransformFactory.get_sam_transforms(1024),
                'qwen_vl_clip': lambda training: TransformFactory.get_ultrasound_transforms(
                    image_size, data_config, training
                ),
                'medclip': lambda training: TransformFactory.get_ultrasound_transforms(
                    image_size, data_config, training
                ),
                'convnext_clip': lambda training: TransformFactory.get_ultrasound_transforms(
                    image_size, data_config, training
                ),
            }
        
        # get_transform = transform_config.get(model_type, transform_config['clip'])
        print(f"transform_config:{transform_config}")
        print(f"model_type:{model_type}")
        get_transform = transform_config[model_type]
        
        # 检查是否启用超声图像优化
        use_ultrasound_optimization = data_config.get('ultrasound_optimization', {}).get('enabled', False)
        
        # 根据数据格式创建数据集
        if data_config['use_json_format']:
            datasets = DataLoaderFactory._create_json_datasets(
                data_config, get_transform, max_samples, validate_files, cache_size,
                use_ultrasound_optimization, image_size
            )
        else:
            datasets = DataLoaderFactory._create_jsonl_datasets(
                data_config, get_transform, max_samples, cache_size,
                use_ultrasound_optimization, image_size
            )
        
        return datasets
    
    @staticmethod
    def _create_json_datasets(data_config, get_transform, max_samples, validate_files, cache_size, 
                            use_ultrasound_optimization=False, image_size=224):
        """创建JSON格式数据集"""
        datasets = {}
        
        for split in ['train', 'valid', 'test']:
            json_path = data_config[f'{split}_json']
            if json_path and os.path.exists(json_path):
                is_training = (split == 'train')
                
                # 如果启用超声图像优化，使用专门的超声数据集
                if use_ultrasound_optimization:
                    datasets[split] = UltrasoundDataset(
                        json_path=json_path,
                        max_samples=max_samples,
                        validate_files=validate_files,
                        cache_size=cache_size,
                        target_image_size=image_size,
                        augment=is_training,  # 训练时启用增强
                        ultrasound_specific=True
                    )
                    logger.info(f"使用超声图像优化数据集: {split}")
                else:
                    # 使用原有的通用数据集
                    transform = get_transform(is_training)
                    datasets[split] = JSONDataset(
                        json_path,
                        transform=transform,
                        max_samples=max_samples,
                        validate_files=validate_files,
                        cache_size=cache_size
                    )
        
        return datasets
    
    @staticmethod
    def _create_jsonl_datasets(data_config, get_transform, max_samples, cache_size,
                              use_ultrasound_optimization=False, image_size=224):
        """创建JSONL格式数据集"""
        datasets = {}
        
        for split in ['train', 'valid', 'test']:
            jsonl_path = data_config.get(f'{split}_jsonl')
            if jsonl_path and os.path.exists(jsonl_path):
                is_training = (split == 'train')
                
                # 如果启用超声图像优化，使用专门的超声数据集
                if use_ultrasound_optimization:
                    # 注意：JSONL格式需要转换为JSON格式或修改UltrasoundDataset支持JSONL
                    logger.warning(f"JSONL格式暂不支持超声图像优化，使用通用数据集: {split}")
                    transform = get_transform(is_training)
                    datasets[split] = JSONLDataset(
                        jsonl_path,
                        transform=transform,
                        max_samples=max_samples,
                        cache_size=cache_size
                    )
                else:
                    # 使用原有的通用数据集
                    transform = get_transform(is_training)
                    datasets[split] = JSONLDataset(
                        jsonl_path,
                        transform=transform,
                        max_samples=max_samples,
                        cache_size=cache_size
                    )
        
        return datasets
    
    @staticmethod
    def create_dataloaders(
        datasets: Dict[str, Any],
        config: Dict[str, Any],
        drop_last_train: bool = True,
        batch_size: Optional[int] = None
    ) -> Dict[str, DataLoader]:
        """
        创建数据加载器
        Args:
            datasets: 数据集字典
            config: 配置字典
            drop_last_train: 训练时是否丢弃最后一个不完整的batch
            batch_size: 可选，显式指定的batch_size
        """
        data_config = config['data']

        # 获取配置参数
        # 优先使用传入的batch_size，否则从data配置中获取
        if batch_size is None:
            batch_size = data_config.get('batch_size', 16)  # 默认16
        num_workers = data_config['num_workers']
        max_text_length = data_config['max_text_length']
        tokenizer_type = data_config['tokenizer_type']
        
        # 获取模型路径（用于local_bert类型）
        model_path = None
        if tokenizer_type == 'local_bert':
            model_config = config['model']
            text_encoder_config = model_config['text_encoder']
            model_path = text_encoder_config['model_name']

        # 创建collator
        collator = DataCollator(
            max_text_length=max_text_length,
            tokenizer_type=tokenizer_type,
            model_path=model_path
        )

        dataloaders = {}
        for split, dataset in datasets.items():
            shuffle = (split == 'train')
            drop_last = drop_last_train if split == 'train' else False

            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collator,
                pin_memory=True,  # 假设使用GPU
                drop_last=drop_last,
                persistent_workers=num_workers > 0
            )

        return dataloaders
    
    @staticmethod
    def get_dataset_info(datasets: Dict[str, Any]) -> Dict[str, Any]:
        """获取数据集信息"""
        info = {}
        
        for split, dataset in datasets.items():
            info[split] = {
                'size': len(dataset),
                'type': type(dataset).__name__
            }
            
            # 添加统计信息（如果有）
            if hasattr(dataset, 'get_stats'):
                info[split]['stats'] = dataset.get_stats()
        
        return info
