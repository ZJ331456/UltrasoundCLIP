#!/usr/bin/env python3
"""
训练曲线绘制脚本 - 从日志文件动态解析数据
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import os
import re
import argparse

def setup_chinese_font():
    """设置中文字体，提供多种回退选项"""
    # 常见的中文字体列表
    chinese_fonts = [
        'SimHei',           # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'WenQuanYi Micro Hei', # 文泉驿微米黑
        'Source Han Sans CN',  # 思源黑体
        'Noto Sans CJK SC',    # Noto字体
    ]
    
    # 检查可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    print("检查可用中文字体...")
    
    for font in chinese_fonts:
        if font in available_fonts:
            print(f"找到中文字体: {font}")
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return True
    
    # 如果没有找到中文字体，使用英文标签
    print("未找到合适的中文字体，使用英文标签以确保字体显示正常")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return False

def parse_training_log(log_file_path):
    """从日志文件解析训练数据"""
    print(f"解析日志文件: {log_file_path}")
    
    epochs = []
    train_losses = []
    val_losses = []
    val_acc = []  # 这里用于存放 Retrieval R@1 的 avg
    val_top5_acc = []  # 这里用于存放 Retrieval R@5 的(i2t,t2i)平均
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_epoch = None
        # 临时缓存当前 epoch 的指标
        tmp_train_loss = None
        tmp_val_loss = None
        tmp_r1_avg = None
        tmp_r5_avg = None
        
        for line in lines:
            # print(f"line: {line.strip()}")
            line = line.strip()
            
            # 捕获 epoch 开始: 兼容任意总轮数，例如 "Epoch 54/500:"
            if line.startswith("Epoch ") and ":" in line:
                match = re.search(r'^Epoch\s+(\d+)/(\d+):', line)
                if match:
                    # 如果进入新 epoch 前有未写入的数据，先完成写入
                    if current_epoch is not None and tmp_train_loss is not None and tmp_val_loss is not None:
                        epochs.append(current_epoch)
                        train_losses.append(tmp_train_loss)
                        val_losses.append(tmp_val_loss)
                        # r@1 与 r@5 可能为空，缺失时使用 NaN 以便后续处理
                        val_acc.append(tmp_r1_avg if tmp_r1_avg is not None else float('nan'))
                        val_top5_acc.append(tmp_r5_avg if tmp_r5_avg is not None else float('nan'))
                    # 重置并记录新的 epoch
                    current_epoch = int(match.group(1))
                    tmp_train_loss = None
                    tmp_val_loss = None
                    tmp_r1_avg = None
                    tmp_r5_avg = None

            # 训练集损失
            elif "Train Loss:" in line:
                match = re.search(r'Train Loss:\s*([\d.]+)', line)
                if match:
                    tmp_train_loss = float(match.group(1))

            # 验证集损失
            elif "Val Loss:" in line:
                match = re.search(r'Val Loss:\s*([\d.]+)', line)
                if match:
                    tmp_val_loss = float(match.group(1))

            # Retrieval R@1 (i2t/t2i/avg): a / b / c
            elif "Retrieval R@1" in line:
                match = re.search(r'Retrieval R@1\s*\(.*?\):\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)', line)
                if match:
                    tmp_r1_avg = float(match.group(3))

            # Retrieval R@5 (i2t/t2i): a / b  -> 取两者平均
            elif "Retrieval R@5" in line:
                match = re.search(r'Retrieval R@5\s*\(.*?\):\s*([\d.]+)\s*/\s*([\d.]+)', line)
                if match:
                    r5_i2t = float(match.group(1))
                    r5_t2i = float(match.group(2))
                    tmp_r5_avg = (r5_i2t + r5_t2i) / 2.0

            # 一个 epoch 的指标块通常在 LR 行结束，这里完成一次写入
            elif line.startswith("LR:") or line.startswith("学习率:"):
                if current_epoch is not None and tmp_train_loss is not None and tmp_val_loss is not None:
                    epochs.append(current_epoch)
                    train_losses.append(tmp_train_loss)
                    val_losses.append(tmp_val_loss)
                    val_acc.append(tmp_r1_avg if tmp_r1_avg is not None else float('nan'))
                    val_top5_acc.append(tmp_r5_avg if tmp_r5_avg is not None else float('nan'))
                    # 重置，等待下一个 epoch
                    current_epoch = None
                    tmp_train_loss = None
                    tmp_val_loss = None
                    tmp_r1_avg = None
                    tmp_r5_avg = None
            # print(f"当前行: {line}")
            # print(f"当前 epoch: {current_epoch}, 当前训练损失: {current_train_loss}")
        
        if len(epochs) > 0:
            print(f"成功解析 {len(epochs)} 个epoch的数据")
            print(f"训练损失范围: {np.nanmin(train_losses):.4f} - {np.nanmax(train_losses):.4f}")
            print(f"验证损失范围: {np.nanmin(val_losses):.4f} - {np.nanmax(val_losses):.4f}")
            # val_acc 代表 R@1(avg)
            if any([not np.isnan(x) for x in val_acc]):
                print(f"R@1(avg) 范围: {np.nanmin(val_acc):.4f} - {np.nanmax(val_acc):.4f}")
        
        return epochs, train_losses, val_losses, val_acc, val_top5_acc
        
    except FileNotFoundError:
        print(f"错误: 找不到日志文件 {log_file_path}")
        return None
    except Exception as e:
        print(f"解析日志文件时出错: {e}")
        return None

def plot_training_curves(epochs, train_losses, val_losses, val_acc, val_top5_acc, output_path, use_chinese=False):
    """绘制训练曲线图"""
    
    # 根据字体支持情况选择标签
    if use_chinese:
        labels = {
            'train_loss': '训练损失',
            'val_loss': '验证损失', 
            'val_acc': '验证准确率',
            'val_top5': '验证Top-5准确率',
            'train_change': '训练损失变化率',
            'val_change': '验证损失变化率',
            'loss_gap': '验证损失 - 训练损失',
            'titles': {
                'loss_compare': '训练损失 vs 验证损失',
                'accuracy': '验证准确率曲线', 
                'convergence': '损失变化率分析',
                'overfitting': '过拟合分析 (验证损失 - 训练损失)'
            }
        }
    else:
        labels = {
            'train_loss': 'Training Loss',
            'val_loss': 'Validation Loss',
            'val_acc': 'Validation Accuracy', 
            'val_top5': 'Validation Top-5 Accuracy',
            'train_change': 'Training Loss Change Rate',
            'val_change': 'Validation Loss Change Rate', 
            'loss_gap': 'Val Loss - Train Loss',
            'titles': {
                'loss_compare': 'Training Loss vs Validation Loss',
                'accuracy': 'Validation Accuracy Curves',
                'convergence': 'Loss Change Rate Analysis', 
                'overfitting': 'Overfitting Analysis (Val Loss - Train Loss)'
            }
        }

    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 损失曲线对比
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label=labels['train_loss'], marker='o', markersize=4)
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label=labels['val_loss'], marker='s', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(labels['titles']['loss_compare'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, max(epochs))

    # 2. 验证准确率曲线
    ax2.plot(epochs, [acc * 100 for acc in val_acc], 'g-', linewidth=2, label=labels['val_acc'], marker='o', markersize=4)
    ax2.plot(epochs, [acc * 100 for acc in val_top5_acc], 'orange', linewidth=2, label=labels['val_top5'], marker='s', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(labels['titles']['accuracy'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, max(epochs))

    # 3. 训练收敛分析
    # 计算损失减少速度
    train_loss_reduction = np.diff(train_losses)
    val_loss_reduction = np.diff(val_losses)

    ax3.plot(epochs[1:], train_loss_reduction, 'b-', linewidth=2, label=labels['train_change'], marker='o', markersize=3)
    ax3.plot(epochs[1:], val_loss_reduction, 'r-', linewidth=2, label=labels['val_change'], marker='s', markersize=3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Change')
    ax3.set_title(labels['titles']['convergence'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 过拟合分析
    overfitting_gap = np.array(val_losses) - np.array(train_losses)
    ax4.plot(epochs, overfitting_gap, 'purple', linewidth=2, label=labels['loss_gap'], marker='o', markersize=4)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Gap')
    ax4.set_title(labels['titles']['overfitting'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线图已保存到: {output_path}")
    plt.show()

def print_analysis_report(epochs, train_losses, val_losses, val_acc, val_top5_acc, use_chinese=False):
    """打印训练分析报告"""
    print("=" * 60)
    if use_chinese:
        print(" 训练分析报告")
        print("=" * 60)
        
        print(f"\n 基本统计:")
        print(f"  训练轮数: {len(epochs)} epochs")
        print(f"  最终训练损失: {train_losses[-1]:.4f}")
        print(f"  最终验证损失: {val_losses[-1]:.4f}")
        print(f"  最终验证准确率: {val_acc[-1]*100:.2f}%")
        print(f"  最终Top-5准确率: {val_top5_acc[-1]*100:.2f}%")
        
        print(f"\n 性能改进:")
        print(f"  训练损失降幅: {train_losses[0] - train_losses[-1]:.4f} ({((train_losses[0] - train_losses[-1])/train_losses[0]*100):.1f}%)")
        print(f"  验证损失降幅: {val_losses[0] - val_losses[-1]:.4f} ({((val_losses[0] - val_losses[-1])/val_losses[0]*100):.1f}%)")
        print(f"  准确率提升: {(val_acc[-1] - val_acc[0])*100:.2f}% (绝对值)")
        print(f"  Top-5准确率提升: {(val_top5_acc[-1] - val_top5_acc[0])*100:.2f}% (绝对值)")
        
        print(f"\n 潜在问题:")
        # 分析过拟合
        final_gap = val_losses[-1] - train_losses[-1]
        print(f"  最终损失差距: {final_gap:.4f}")
        if final_gap > 0.1:
            print("   存在过拟合风险")
        else:
            print("   过拟合风险较低")
    else:
        print(" Training Analysis Report")
        print("=" * 60)
        
        print(f"\n Basic Statistics:")
        print(f"  Training epochs: {len(epochs)} epochs")
        print(f"  Final training loss: {train_losses[-1]:.4f}")
        print(f"  Final validation loss: {val_losses[-1]:.4f}")
        print(f"  Final validation accuracy: {val_acc[-1]*100:.2f}%")
        print(f"  Final Top-5 accuracy: {val_top5_acc[-1]*100:.2f}%")
        
        print(f"\n Performance Improvement:")
        print(f"  Training loss reduction: {train_losses[0] - train_losses[-1]:.4f} ({((train_losses[0] - train_losses[-1])/train_losses[0]*100):.1f}%)")
        print(f"  Validation loss reduction: {val_losses[0] - val_losses[-1]:.4f} ({((val_losses[0] - val_losses[-1])/val_losses[0]*100):.1f}%)")
        print(f"  Accuracy improvement: {(val_acc[-1] - val_acc[0])*100:.2f}% (absolute)")
        print(f"  Top-5 accuracy improvement: {(val_top5_acc[-1] - val_top5_acc[0])*100:.2f}% (absolute)")
        
        print(f"\n Potential Issues:")
        # 分析过拟合
        final_gap = val_losses[-1] - train_losses[-1]
        print(f"  Final loss gap: {final_gap:.4f}")
        if final_gap > 0.1:
            print("   Overfitting risk detected")
        else:
            print("   Low overfitting risk")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='从日志文件生成训练曲线图')
    parser.add_argument('--log_file', type=str, 
                       default='/media/ps/data-ssd/UltrasoundRAG/CLIP/logs/training_convnext_config_2_combined_best_all_data.log',
                       help='日志文件路径')
    parser.add_argument('--output', type=str,
                       default='/media/ps/data-ssd/UltrasoundRAG/CLIP/png/convnext/training_convnext_config_2_combined_best_all_data.png',
                       help='输出图片路径')
    
    args = parser.parse_args()
    
    # 设置字体 - 在Linux环境下往往缺少中文字体，直接使用英文
    use_chinese = False  # 强制使用英文标签避免字体问题
    print("使用英文标签确保最佳显示效果")
    
    # 从日志文件解析数据
    result = parse_training_log(args.log_file)
    if result is None:
        print("无法解析日志文件，退出")
        exit(1)
    
    epochs, train_losses, val_losses, val_acc, val_top5_acc = result
    
    if len(epochs) == 0:
        print("日志文件中没有找到有效的训练数据")
        exit(1)
    
    # 生成训练曲线图
    plot_training_curves(epochs, train_losses, val_losses, val_acc, val_top5_acc, args.output, use_chinese)
    
    # 打印分析报告
    print_analysis_report(epochs, train_losses, val_losses, val_acc, val_top5_acc, use_chinese)