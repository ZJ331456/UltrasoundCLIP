#!/usr/bin/env python3
"""
更新的训练脚本 - 支持新的dataload模块
"""

import argparse
import torch
import os
import sys
import glob
import re
import gc

# 添加路径以确保能导入模块
sys.path.append('/home/zhoujun/CLIP')

from config_manager import ConfigManager
from dataload import DataLoaderFactory
from models.model_factory import ModelFactory
from trainer import Trainer


def find_latest_checkpoint(save_dir: str) -> str:
    """
    自动查找最新的checkpoint文件（用于断点续训）
    Args:
        save_dir: 保存目录
    Returns:
        最新checkpoint文件路径，如果没有找到返回None
    """
    if not os.path.exists(save_dir):
        return None
    
    # 优先查找latest_checkpoint.pth（专门用于断点续训）
    latest_checkpoint = os.path.join(save_dir, "latest_checkpoint.pth")
    if os.path.exists(latest_checkpoint):
        return latest_checkpoint
    
    # 如果没有latest_checkpoint.pth，查找其他checkpoint文件
    checkpoint_patterns = [
        os.path.join(save_dir, "checkpoint_epoch_*.pth"),  # 旧格式的epoch checkpoint
        os.path.join(save_dir, "best_model.pth"),          # 最佳模型
        os.path.join(save_dir, "emergency_*.pth"),         # 紧急保存的文件
        os.path.join(save_dir, "*.pth")                    # 通用模式
    ]
    
    checkpoint_files = []
    for pattern in checkpoint_patterns:
        checkpoint_files.extend(glob.glob(pattern))
    
    if not checkpoint_files:
        return None
    
    # 按修改时间排序，返回最新的
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # 过滤掉best_model.pth（通常不用于断点续训，因为它可能不是最新的）
    filtered_files = [f for f in checkpoint_files if "best_model.pth" not in f]
    
    if filtered_files:
        return filtered_files[0]
    else:
        # 如果只有best_model.pth，也可以用于断点续训
        return checkpoint_files[0]


def setup_memory_optimization():
    """设置内存优化"""
    # 启用内存高效注意力
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # 设置内存分配策略
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 设置内存分配器
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,expandable_segments:True'


def main():
    parser = argparse.ArgumentParser(description='CLIP模型训练')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, help='恢复训练的checkpoint路径')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--device', type=str, default='auto', help='设备选择')
    parser.add_argument('--debug', action='store_true', help='调试模式，使用少量样本')
    
    args = parser.parse_args()
    
    # 设置内存优化
    setup_memory_optimization()
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 加载配置
    config_manager = ConfigManager(args.config)
    config_dict = config_manager.config
    print("配置加载完成")
    
    # 创建数据集
    print("创建数据集...")
    
    # 检查超声图片特定配置
    data_config = config_dict['data']
    if 'ultrasound_augmentation' in data_config:
        aug_config = data_config['ultrasound_augmentation']
        print("超声图片数据增强配置:")
        print(f"  - 启用状态: {aug_config['enabled']}")
        print(f"  - 对比度增强: {aug_config['contrast_enhancement']}")
        print(f"  - 降噪处理: {aug_config['noise_reduction']}")
        print(f"  - 边缘增强: {aug_config['edge_enhancement']}")
        print(f"  - 图像标准化: {data_config['image_normalization']}")
    
    # 调试模式下减少数据量
    max_samples = 50 if args.debug else None
    validate_files = not args.debug
    cache_size = 20 if args.debug else 200
    
    datasets = DataLoaderFactory.create_datasets(
        config_dict,
        max_samples=max_samples,
        validate_files=validate_files,
        cache_size=cache_size
    )
    
    batch_size = config_dict['training']['batch_size']
    dataloaders = DataLoaderFactory.create_dataloaders(
        datasets, 
        config_dict, 
        drop_last_train=True,  # 改为True避免最后一个不完整batch
        batch_size=batch_size
    )
    # 获取数据加载器信息
    print("获取数据加载器信息...")
    print(f"训练集 DataLoader: {dataloaders['train']}")
    print(f"验证集 DataLoader: {dataloaders['valid']}")
    train_loader = dataloaders['train']
    val_loader = dataloaders['valid']
    
    # 打印训练和验证集的 batch_size
    print(f"配置文件中的 batch_size: {config_dict['training']['batch_size']}")
    print(f"训练集 DataLoader 的 batch_size: {train_loader.batch_size}")
    if val_loader:
        print(f"验证集 DataLoader 的 batch_size: {val_loader.batch_size}")

    if not train_loader:
        raise ValueError("训练集未找到")
    if not val_loader:
        print("警告: 验证集未找到，将使用训练集进行验证")
        val_loader = train_loader
    
    print(f"训练集大小: {len(datasets['train'])}")
    if 'valid' in datasets:
        print(f"验证集大小: {len(datasets['valid'])}")
    
    # 获取数据集信息
    dataset_info = DataLoaderFactory.get_dataset_info(datasets)
    print("数据集信息:")
    for split, info in dataset_info.items():
        print(f"  {split}: {info}")
    
    # 创建模型
    print("创建模型...")
    model_config = config_dict['model']
    model = ModelFactory.create_model(model_config)
    # print(f"model_config':{model_config}")
    # print(f"pretrained_path:{pretrained_path}")
    # 加载预训练权重（如果指定）
    # if model_config['pretrained_path']:
    # 这个预训练权重是并没有在配置文件里面设置的，所以还是得用get方式读取，读取不到就将值置为None
    if model_config.get('pretrained_path'):
        model = ModelFactory.load_checkpoint(model, model_config['pretrained_path'])
    
    print(f"模型创建完成: {type(model).__name__}")
    
    # 验证分词器
    print(f"分词器词汇表大小: {len(model.tokenizer)}")
    
    # 创建训练器
    training_config = config_dict['training']
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device
    )
    
    # 恢复训练（如果指定）
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        print(f"从指定checkpoint恢复训练: {args.checkpoint}")
    else:
        # 自动查找最新的checkpoint
        latest_checkpoint = find_latest_checkpoint(args.save_dir)
        if latest_checkpoint:
            print(f"自动发现checkpoint: {latest_checkpoint}")
            response = input("是否从该checkpoint恢复训练? (y/n): ").lower().strip()
            if response in ['y', 'yes', '是']:
                trainer.load_checkpoint(latest_checkpoint)
                print(f"从checkpoint恢复训练: {latest_checkpoint}")
            else:
                print("从头开始训练")
        else:
            print("从头开始训练")
    
    # 开始训练
    max_epochs = training_config['max_epochs']
    if args.debug:
        max_epochs = min(max_epochs, 3)  # 调试模式只训练3个epoch
        print(f"调试模式: 只训练 {max_epochs} 个epoch")
    
    # 如果从checkpoint恢复，调整训练epoch数
    if hasattr(trainer, 'current_epoch') and trainer.current_epoch > 0:
        remaining_epochs = max_epochs - trainer.current_epoch
        if remaining_epochs > 0:
            print(f"从epoch {trainer.current_epoch} 继续训练，还需训练 {remaining_epochs} 个epoch")
            print(f"目标总epoch数: {training_config['max_epochs']}")
            max_epochs = remaining_epochs
        else:
            print(f"已训练完成目标epoch数: {trainer.current_epoch}")
            return
    else:
        print(f"从头开始训练，目标epoch数: {training_config['max_epochs']}")
    
    print(f"开始训练，当前阶段epoch数: {max_epochs}")
    
    try:
        trainer.train(max_epochs, args.save_dir)
        print("训练完成!")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        # 尝试保存当前状态
        if hasattr(trainer, 'save_checkpoint'):
            emergency_save_path = os.path.join(args.save_dir, 'emergency_save.pth')
            trainer.save_checkpoint(emergency_save_path, trainer.current_epoch, {'error': str(e)})
            print(f"紧急保存到: {emergency_save_path}")
    finally:
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    main()
