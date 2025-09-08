#!/usr/bin/env python3
"""
配置文件验证脚本
验证对照组实验配置文件的正确性和兼容性
"""

import json
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

def validate_config_file(config_path: str) -> bool:
    """验证单个配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"✅ {os.path.basename(config_path)} - JSON格式正确")
        
        # 检查必要的字段
        required_fields = ['model', 'training', 'data']
        for field in required_fields:
            if field not in config:
                print(f"❌ 缺少必要字段: {field}")
                return False
        
        print(f"✅ {os.path.basename(config_path)} - 必要字段完整")
        
        # 检查训练配置
        training_config = config['training']
        required_training_fields = ['batch_size', 'learning_rate', 'max_epochs', 'optimizer', 'scheduler', 'loss']
        for field in required_training_fields:
            if field not in training_config:
                print(f"❌ 训练配置缺少字段: {field}")
                return False
        
        print(f"✅ {os.path.basename(config_path)} - 训练配置完整")
        
        # 检查损失函数配置
        loss_config = training_config['loss']
        if 'type' not in loss_config:
            print(f"❌ 损失函数配置缺少type字段")
            return False
        
        print(f"✅ {os.path.basename(config_path)} - 损失函数配置正确")
        
        # 检查数据配置
        data_config = config['data']
        required_data_fields = ['image_size', 'train_json', 'valid_json']
        for field in required_data_fields:
            if field not in data_config:
                print(f"❌ 数据配置缺少字段: {field}")
                return False
        
        print(f"✅ {os.path.basename(config_path)} - 数据配置完整")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ {os.path.basename(config_path)} - JSON格式错误: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    except Exception as e:
        print(f"❌ {os.path.basename(config_path)} - 验证失败: {e}")
        return False


def validate_config_compatibility():
    """验证配置文件与代码的兼容性"""
    try:
        # 验证损失函数
        from loss import LossFactory
        
        # 测试简化稳定损失函数
        try:
            loss_fn = LossFactory.create_loss('simplified_clip_stable', temperature=0.07)
            print("✅ SimplifiedClipStableLoss 创建成功")
        except Exception as e:
            print(f"❌ SimplifiedClipStableLoss 创建失败: {e}")
            return False
        
        # 验证数据变换
        from dataload.transforms import MedicalSafeAugmentation, create_medical_safe_transforms
        
        # 测试医学安全增强
        try:
            config = {'enabled': True, 'use_medical_optimized': True}
            aug = MedicalSafeAugmentation(config)
            print("✅ MedicalSafeAugmentation 创建成功")
        except Exception as e:
            print(f"❌ MedicalSafeAugmentation 创建失败: {e}")
            return False
        
        # 测试医学安全变换
        try:
            transforms = create_medical_safe_transforms(224, True, "imagenet", config)
            print("✅ create_medical_safe_transforms 创建成功")
        except Exception as e:
            print(f"❌ create_medical_safe_transforms 创建失败: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        return False


def main():
    """主函数"""
    print("🔧 开始验证对照组实验配置文件...")
    print()
    
    # 配置文件路径
    config_dir = Path(__file__).parent.parent / "config" / "convnext"
    config_files = [
        "convnext_config_new_dataload_updata_data_1.json",  # 基线配置
        "convnext_config_new_dataload_updata_data_1_loss_optimized.json",
        "convnext_config_new_dataload_updata_data_1_augmentation_optimized.json", 
        "convnext_config_new_dataload_updata_data_1_lr_optimized.json",
        "convnext_config_new_dataload_updata_data_1_ema_optimized.json"
    ]
    
    # 验证每个配置文件
    all_valid = True
    for config_file in config_files:
        config_path = config_dir / config_file
        print(f"🔍 验证配置文件: {config_file}")
        
        if validate_config_file(str(config_path)):
            print(f"✅ {config_file} - 验证通过")
        else:
            print(f"❌ {config_file} - 验证失败")
            all_valid = False
        print()
    
    # 验证代码兼容性
    print("🔍 验证代码兼容性...")
    if validate_config_compatibility():
        print("✅ 代码兼容性验证通过")
    else:
        print("❌ 代码兼容性验证失败")
        all_valid = False
    print()
    
    # 输出最终结果
    if all_valid:
        print("🎉 所有配置文件和代码兼容性验证通过！")
        print()
        print("📋 接下来可以运行实验：")
        print("1. 损失函数优化实验：python train.py --config config/convnext/convnext_config_new_dataload_updata_data_1_loss_optimized.json --save_dir checkpoints/loss_optimized")
        print("2. 数据增强优化实验：python train.py --config config/convnext/convnext_config_new_dataload_updata_data_1_augmentation_optimized.json --save_dir checkpoints/augmentation_optimized")
        print("3. 学习率策略优化实验：python train.py --config config/convnext/convnext_config_new_dataload_updata_data_1_lr_optimized.json --save_dir checkpoints/lr_optimized")
        print("4. EMA优化实验：python train.py --config config/convnext/convnext_config_new_dataload_updata_data_1_ema_optimized.json --save_dir checkpoints/ema_optimized")
        return True
    else:
        print("❌ 验证失败，请检查配置文件和代码！")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
