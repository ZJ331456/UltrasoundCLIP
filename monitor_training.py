#!/usr/bin/env python3
"""
训练进度监控脚本
实时监控训练状态和关键指标
"""

import os
import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import re
from collections import defaultdict

def parse_log_file(log_path: str):
    """解析训练日志文件"""
    if not os.path.exists(log_path):
        print(f"日志文件不存在: {log_path}")
        return None
    
    metrics = defaultdict(list)
    epochs = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析epoch结果
            if "Epoch " in line and "/" in line and ":" in line:
                try:
                    epoch_match = re.search(r'Epoch (\d+)/(\d+):', line)
                    if epoch_match:
                        current_epoch = int(epoch_match.group(1))
                        epochs.append(current_epoch)
                except:
                    continue
            
            # 解析训练损失
            if "Train Loss:" in line:
                try:
                    loss_match = re.search(r'Train Loss: ([\d.]+)', line)
                    if loss_match:
                        metrics['train_loss'].append(float(loss_match.group(1)))
                except:
                    continue
            
            # 解析验证损失
            if "Val Loss:" in line:
                try:
                    loss_match = re.search(r'Val Loss: ([\d.]+)', line)
                    if loss_match:
                        metrics['val_loss'].append(float(loss_match.group(1)))
                except:
                    continue
            
            # 解析验证准确率
            if "Val Acc:" in line:
                try:
                    acc_match = re.search(r'Val Acc: ([\d.]+)', line)
                    if acc_match:
                        metrics['val_acc'].append(float(acc_match.group(1)))
                except:
                    continue
            
            # 解析Top-5准确率
            if "Val Top-5 Acc:" in line:
                try:
                    acc_match = re.search(r'Val Top-5 Acc: ([\d.]+)', line)
                    if acc_match:
                        metrics['val_top5_acc'].append(float(acc_match.group(1)))
                except:
                    continue
            
            # 解析学习率
            if "LR:" in line:
                try:
                    lr_match = re.search(r'LR: ([\d.e-]+)', line)
                    if lr_match:
                        metrics['learning_rate'].append(float(lr_match.group(1)))
                except:
                    continue
    
    return metrics, epochs

def plot_training_curves(metrics, epochs, save_path=None):
    """绘制训练曲线"""
    if not metrics or not epochs:
        print("没有足够的数据绘制曲线")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('训练进度监控', fontsize=16, fontweight='bold')
    
    # 损失曲线
    ax1 = axes[0, 0]
    if 'train_loss' in metrics and 'val_loss' in metrics:
        min_len = min(len(metrics['train_loss']), len(metrics['val_loss']), len(epochs))
        ax1.plot(epochs[:min_len], metrics['train_loss'][:min_len], 'b-', label='训练损失', linewidth=2)
        ax1.plot(epochs[:min_len], metrics['val_loss'][:min_len], 'r-', label='验证损失', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('损失曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2 = axes[0, 1]
    if 'val_acc' in metrics:
        min_len = min(len(metrics['val_acc']), len(epochs))
        ax2.plot(epochs[:min_len], metrics['val_acc'][:min_len], 'g-', label='验证准确率', linewidth=2)
        if 'val_top5_acc' in metrics:
            min_len = min(len(metrics['val_top5_acc']), len(epochs))
            ax2.plot(epochs[:min_len], metrics['val_top5_acc'][:min_len], 'orange', label='Top-5准确率', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('准确率曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 学习率曲线
    ax3 = axes[1, 0]
    if 'learning_rate' in metrics:
        min_len = min(len(metrics['learning_rate']), len(epochs))
        ax3.plot(epochs[:min_len], metrics['learning_rate'][:min_len], 'purple', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('学习率变化')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    
    # 训练状态摘要
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 显示最新指标
    summary_text = "最新训练状态:\n\n"
    if metrics:
        if 'train_loss' in metrics and metrics['train_loss']:
            summary_text += f"训练损失: {metrics['train_loss'][-1]:.4f}\n"
        if 'val_loss' in metrics and metrics['val_loss']:
            summary_text += f"验证损失: {metrics['val_loss'][-1]:.4f}\n"
        if 'val_acc' in metrics and metrics['val_acc']:
            summary_text += f"验证准确率: {metrics['val_acc'][-1]:.4f}\n"
        if 'val_top5_acc' in metrics and metrics['val_top5_acc']:
            summary_text += f"Top-5准确率: {metrics['val_top5_acc'][-1]:.4f}\n"
        if 'learning_rate' in metrics and metrics['learning_rate']:
            summary_text += f"学习率: {metrics['learning_rate'][-1]:.2e}\n"
        
        summary_text += f"\n已完成Epoch: {len(epochs)}"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    
    plt.show()

def monitor_training(log_path: str, refresh_interval: int = 30, output_dir: str = None):
    """实时监控训练进度"""
    print(f"开始监控训练日志: {log_path}")
    print(f"刷新间隔: {refresh_interval}秒")
    print("按 Ctrl+C 停止监控\n")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        while True:
            print(f"\n{'='*50}")
            print(f"更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}")
            
            # 解析日志
            metrics, epochs = parse_log_file(log_path)
            
            if metrics and epochs:
                # 显示最新状态
                print(f"已完成Epoch: {len(epochs)}")
                if 'train_loss' in metrics and metrics['train_loss']:
                    print(f"最新训练损失: {metrics['train_loss'][-1]:.4f}")
                if 'val_loss' in metrics and metrics['val_loss']:
                    print(f"最新验证损失: {metrics['val_loss'][-1]:.4f}")
                if 'val_acc' in metrics and metrics['val_acc']:
                    print(f"最新验证准确率: {metrics['val_acc'][-1]:.4f}")
                if 'val_top5_acc' in metrics and metrics['val_top5_acc']:
                    print(f"最新Top-5准确率: {metrics['val_top5_acc'][-1]:.4f}")
                
                # 保存训练曲线
                if output_dir:
                    plot_path = os.path.join(output_dir, 'training_curves.png')
                    plot_training_curves(metrics, epochs, plot_path)
            else:
                print("暂无训练数据或日志文件不存在")
            
            print(f"\n等待 {refresh_interval} 秒后下次更新...")
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n监控已停止")

def main():
    parser = argparse.ArgumentParser(description='训练进度监控工具')
    parser.add_argument('--log_path', type=str, required=True, help='训练日志文件路径')
    parser.add_argument('--refresh_interval', type=int, default=30, help='刷新间隔（秒）')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--plot_only', action='store_true', help='仅绘制曲线，不实时监控')
    
    args = parser.parse_args()
    
    if args.plot_only:
        # 仅绘制曲线
        metrics, epochs = parse_log_file(args.log_path)
        plot_path = None
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            plot_path = os.path.join(args.output_dir, 'training_curves.png')
        plot_training_curves(metrics, epochs, plot_path)
    else:
        # 实时监控
        monitor_training(args.log_path, args.refresh_interval, args.output_dir)

if __name__ == '__main__':
    main()
