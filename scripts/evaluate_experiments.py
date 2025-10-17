#!/usr/bin/env python3
"""
评估不同实验版本的性能对比脚本
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 添加路径
sys.path.append('/media/ps/data-ssd/UltrasoundRAG/CLIP')

from config_manager import ConfigManager
from eval import main as eval_main
import argparse


def evaluate_model(config_path, checkpoint_path, output_dir):
    """评估单个模型"""
    # 构建评估命令参数
    sys.argv = [
        'eval.py',
        '--config', config_path,
        '--model_path', checkpoint_path,
        '--dataset', 'test',
        '--output_dir', output_dir,
        '--eval_type', 'both'
    ]
    
    # 运行评估
    eval_main()
    
    # 读取评估结果
    results_path = os.path.join(output_dir, 'comprehensive_evaluation.json')
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def create_comparison_table(all_results):
    """创建对比表格"""
    data = []
    
    for version, results in all_results.items():
        metrics = results.get('standard_metrics', {})
        row = {
            '版本': version,
            'I2T_R@1': f"{metrics.get('i2t_r1', 0)*100:.2f}%",
            'T2I_R@1': f"{metrics.get('t2i_r1', 0)*100:.2f}%",
            'Mean_R@1': f"{metrics.get('mean_r1', 0)*100:.2f}%",
            'I2T_R@5': f"{metrics.get('i2t_r5', 0)*100:.2f}%",
            'T2I_R@5': f"{metrics.get('t2i_r5', 0)*100:.2f}%",
            'Mean_R@5': f"{metrics.get('mean_r5', 0)*100:.2f}%",
        }
        
        # 添加attention相关指标（如果有）
        if 'avg_attention_coverage' in metrics:
            row['Attention覆盖度'] = f"{metrics.get('avg_attention_coverage', 0):.4f}"
            row['Attention内部比例'] = f"{metrics.get('avg_attention_inside_ratio', 0):.4f}"
            row['Attention稀疏度'] = f"{metrics.get('avg_attention_sparsity', 0):.4f}"
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def plot_comparison_charts(all_results, save_dir):
    """绘制对比图表"""
    versions = []
    mean_r1_scores = []
    mean_r5_scores = []
    attention_coverage = []
    attention_inside_ratio = []
    
    for version, results in all_results.items():
        metrics = results.get('standard_metrics', {})
        versions.append(version)
        mean_r1_scores.append(metrics.get('mean_r1', 0) * 100)
        mean_r5_scores.append(metrics.get('mean_r5', 0) * 100)
        
        if 'avg_attention_coverage' in metrics:
            attention_coverage.append(metrics.get('avg_attention_coverage', 0))
            attention_inside_ratio.append(metrics.get('avg_attention_inside_ratio', 0))
    
    # 绘制检索性能对比
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    x = range(len(versions))
    width = 0.35
    plt.bar([i - width/2 for i in x], mean_r1_scores, width, label='Mean R@1', alpha=0.8)
    plt.bar([i + width/2 for i in x], mean_r5_scores, width, label='Mean R@5', alpha=0.8)
    plt.xlabel('版本')
    plt.ylabel('Recall (%)')
    plt.title('检索性能对比')
    plt.xticks(x, versions, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制attention指标对比（如果有）
    if attention_coverage:
        plt.subplot(1, 2, 2)
        plt.plot(versions, attention_coverage, 'o-', label='覆盖度', markersize=8)
        plt.plot(versions, attention_inside_ratio, 's-', label='内部比例', markersize=8)
        plt.xlabel('版本')
        plt.ylabel('比例')
        plt.title('Attention指标对比')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='评估实验版本对比')
    parser.add_argument('--checkpoint_base', type=str, 
                       default='/media/ps/data-ssd/UltrasoundRAG/CLIP/checkpoints/experiments',
                       help='Checkpoint基础目录')
    parser.add_argument('--output_dir', type=str,
                       default='/media/ps/data-ssd/UltrasoundRAG/CLIP/results/experiments',
                       help='结果保存目录')
    args = parser.parse_args()
    
    # 定义要评估的版本
    versions = {
        'baseline': {
            'config': 'convnext_config_2_combined_best_all_data_mask.json',
            'checkpoint': 'baseline/best_model.pth'
        },
        'v1_mask_supervision': {
            'config': 'convnext_config_2_combined_best_all_data_mask_v1.json',
            'checkpoint': 'v1_mask_supervision/best_model.pth'
        },
        'v2_gating': {
            'config': 'convnext_config_2_combined_best_all_data_mask_v2.json',
            'checkpoint': 'v2_gating/best_model.pth'
        },
        'v3_multiscale': {
            'config': 'convnext_config_2_combined_best_all_data_mask_v3.json',
            'checkpoint': 'v3_multiscale/best_model.pth'
        },
        'v4_sparse_attention': {
            'config': 'convnext_config_2_combined_best_all_data_mask_v4.json',
            'checkpoint': 'v4_sparse_attention/best_model.pth'
        }
    }
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 评估所有版本
    all_results = {}
    config_base = '/media/ps/data-ssd/UltrasoundRAG/CLIP/config/convnext'
    
    for version_name, version_info in versions.items():
        print(f"\n=== 评估 {version_name} ===")
        
        config_path = os.path.join(config_base, version_info['config'])
        checkpoint_path = os.path.join(args.checkpoint_base, version_info['checkpoint'])
        output_dir = os.path.join(args.output_dir, version_name)
        
        # 检查文件是否存在
        if not os.path.exists(checkpoint_path):
            print(f"警告: {checkpoint_path} 不存在，跳过")
            continue
        
        try:
            results = evaluate_model(config_path, checkpoint_path, output_dir)
            all_results[version_name] = results
        except Exception as e:
            print(f"评估 {version_name} 失败: {e}")
            continue
    
    # 创建对比表格
    if all_results:
        df = create_comparison_table(all_results)
        print("\n=== 性能对比表 ===")
        print(df.to_string(index=False))
        
        # 保存表格
        df.to_csv(os.path.join(args.output_dir, 'performance_comparison.csv'), index=False)
        
        # 绘制对比图表
        plot_comparison_charts(all_results, args.output_dir)
        
        # 生成总结报告
        with open(os.path.join(args.output_dir, 'experiment_summary.md'), 'w') as f:
            f.write("# 超声图文对齐实验结果总结\n\n")
            f.write("## 性能对比表\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n## 关键发现\n\n")
            
            # 找出最佳版本
            best_version = max(all_results.items(), 
                             key=lambda x: x[1].get('standard_metrics', {}).get('mean_r1', 0))
            f.write(f"- 最佳版本: {best_version[0]}\n")
            f.write(f"- 最佳Mean R@1: {best_version[1]['standard_metrics']['mean_r1']*100:.2f}%\n")
            
            # 计算相对提升
            baseline_r1 = all_results.get('baseline', {}).get('standard_metrics', {}).get('mean_r1', 0)
            best_r1 = best_version[1]['standard_metrics']['mean_r1']
            if baseline_r1 > 0:
                improvement = (best_r1 - baseline_r1) / baseline_r1 * 100
                f.write(f"- 相对基线提升: {improvement:.2f}%\n")
    
    print(f"\n所有结果已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
