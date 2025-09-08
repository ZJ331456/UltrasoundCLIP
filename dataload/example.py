"""
数据加载器使用示例
展示如何使用新的数据加载模块
"""

import sys
import os
sys.path.append('/home/zhoujun/CLIP')

from dataload import DataLoaderFactory


def example_usage():
    """使用示例"""
    
    # 示例配置
    config = {
        'data': {
            'use_json_format': True,
            'image_size': 224,
            'batch_size': 16,
            'num_workers': 4,
            'max_text_length': 256,
            'tokenizer_type': 'bert',
            'train_json': '/path/to/train.json',
            'valid_json': '/path/to/valid.json',
            'test_json': '/path/to/test.json'
        },
        'model': {
            'type': 'clip'
        }
    }
    
    print("创建数据集...")
    datasets = DataLoaderFactory.create_datasets(
        config,
        max_samples=100,  # 调试模式，只加载100个样本
        validate_files=False,  # 跳过文件验证以加快速度
        cache_size=50  # 小缓存
    )
    
    print("创建数据加载器...")
    dataloaders = DataLoaderFactory.create_dataloaders(datasets, config)
    
    print("获取数据集信息...")
    info = DataLoaderFactory.get_dataset_info(datasets)
    print(info)
    
    # 测试数据加载
    if 'train' in dataloaders:
        print("测试数据加载...")
        for batch_idx, batch in enumerate(dataloaders['train']):
            print(f"Batch {batch_idx}:")
            print(f"  Images shape: {batch['images'].shape}")
            print(f"  Texts count: {len(batch['texts'])}")
            if batch_idx >= 2:  # 只测试前3个batch
                break


if __name__ == "__main__":
    example_usage()
