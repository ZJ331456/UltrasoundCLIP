#!/usr/bin/env python3
"""
整合的CLIP模型评估脚本
支持标准检索评估和细粒度分析
"""

import argparse
import torch
import json
import os
from config_manager import ConfigManager
from models.model_factory import ModelFactory
from evaluator import Evaluator
from dataload.factory import DataLoaderFactory


def main():
    parser = argparse.ArgumentParser(description='CLIP模型整合评估')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--model_path', type=str, required=False, default='', help='模型权重路径（可选）')
    parser.add_argument('--dataset', type=str, choices=['valid', 'test'], default='test', help='评估数据集')
    parser.add_argument('--device', type=str, default='auto', help='设备选择')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='结果保存目录')
    parser.add_argument('--eval_type', type=str, choices=['standard', 'finegrained', 'both'], 
                       default='both', help='评估类型')
    parser.add_argument('--max_batches', type=int, default=100, help='细粒度评估的最大批次数')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')
    # CN-CLIP性能优化参数
    parser.add_argument('--eval_batch_size', type=int, default=32768, 
                       help='批处理大小（参考CN-CLIP优化，默认32768）')
    # 保持向后兼容
    parser.add_argument('--output', type=str, help='结果保存路径（向后兼容）')
    
    args = parser.parse_args()
    
    # 向后兼容处理
    if args.output and not args.output_dir:
        args.output_dir = os.path.dirname(args.output) or './eval_results'
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载配置
    config_manager = ConfigManager(args.config)
    config_dict = config_manager.config
    print("配置加载完成")

    # 创建数据集
    print("创建数据集...")
    datasets = DataLoaderFactory.create_datasets(
        config_dict,
        max_samples=None,
        validate_files=True,
        cache_size=500
    )
    dataloaders = DataLoaderFactory.create_dataloaders(datasets, config_dict)

    if args.dataset not in dataloaders:
        raise ValueError(f"数据集 {args.dataset} 未找到")

    dataloader = dataloaders[args.dataset]
    print(f"{args.dataset}集大小: {len(datasets[args.dataset])}")

    # 创建模型
    model_config = config_dict['model']
    model = ModelFactory.create_model(model_config)

    # 处理meta tensor
    try:
        if any(getattr(p, 'is_meta', False) for p in model.parameters()):
            print("检测到meta张量，使用to_empty在设备侧初始化空权重...")
            if hasattr(model, 'to_empty'):
                try:
                    model.to_empty(device=device)
                except TypeError:
                    model.to_empty()
    except Exception as e:
        print(f"meta初始化警告: {e}")

    # 加载权重
    if args.model_path and os.path.isfile(args.model_path):
        try:
            ckpt = torch.load(args.model_path, map_location=device)
            state = ckpt.get('model_state_dict', ckpt)
            try:
                model.load_state_dict(state, strict=False, assign=True)
            except TypeError:
                model.load_state_dict(state, strict=False)
            print("模型权重加载完成")
        except Exception as e:
            print(f"权重加载失败: {e}")
            model = ModelFactory.load_checkpoint(model, args.model_path)
        
        model = model.to(device)
        print(f"模型加载完成: {type(model).__name__}")
    else:
        print("未加载模型权重，使用初始化模型")

    # 创建评估器（添加CN-CLIP性能优化）
    evaluator = Evaluator(model, device, eval_batch_size=args.eval_batch_size)
    
    if args.verbose:
        print(f"使用CN-CLIP优化的批处理大小: {args.eval_batch_size}")

    # 执行评估
    results = {}
    
    if args.eval_type in ['standard', 'both']:
        print(f"\n开始标准检索评估...")
        standard_metrics = evaluator.evaluate(dataloader, verbose=args.verbose)
        evaluator.print_metrics(standard_metrics)
        results['standard_metrics'] = standard_metrics
        
        # 保存标准评估结果
        with open(f"{args.output_dir}/standard_evaluation.json", 'w', encoding='utf-8') as f:
            json.dump(standard_metrics, f, ensure_ascii=False, indent=2)
    
    if args.eval_type in ['finegrained', 'both']:
        print(f"\n开始细粒度评估...")
        finegrained_report = evaluator.evaluate_finegrained(
            dataloader, 
            save_dir=args.output_dir,
            max_batches=args.max_batches
        )
        evaluator.print_finegrained_metrics(finegrained_report)
        results['finegrained_report'] = finegrained_report
    
    # 保存综合结果
    with open(f"{args.output_dir}/comprehensive_evaluation.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n所有评估结果已保存到: {args.output_dir}")
    
    # 生成评估摘要
    generate_summary_report(results, args.output_dir)
    
    # 向后兼容：如果指定了--output参数，也保存到该路径
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            # 保存主要的标准评估结果以保持兼容性
            output_data = results.get('standard_metrics', results)
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"兼容性结果已保存到: {args.output}")


def generate_summary_report(results: dict, output_dir: str):
    """生成评估摘要报告 - 参考CN-CLIP标准"""
    summary_lines = []
    summary_lines.append("# CLIP模型评估摘要报告 (CN-CLIP标准)\n")
    
    # 标准检索指标
    if 'standard_metrics' in results:
        metrics = results['standard_metrics']
        
        # 双向检索结果
        summary_lines.append("## 检索性能指标")
        summary_lines.append(f"### 图像→文本检索 (I2T):")
        summary_lines.append(f"- Recall@1: {metrics.get('i2t_r1', 0):.4f} ({metrics.get('i2t_r1', 0)*100:.2f}%)")
        summary_lines.append(f"- Recall@5: {metrics.get('i2t_r5', 0):.4f} ({metrics.get('i2t_r5', 0)*100:.2f}%)")
        summary_lines.append(f"- Recall@10: {metrics.get('i2t_r10', 0):.4f} ({metrics.get('i2t_r10', 0)*100:.2f}%)")
        
        summary_lines.append(f"### 文本→图像检索 (T2I):")
        summary_lines.append(f"- Recall@1: {metrics.get('t2i_r1', 0):.4f} ({metrics.get('t2i_r1', 0)*100:.2f}%)")
        summary_lines.append(f"- Recall@5: {metrics.get('t2i_r5', 0):.4f} ({metrics.get('t2i_r5', 0)*100:.2f}%)")
        summary_lines.append(f"- Recall@10: {metrics.get('t2i_r10', 0):.4f} ({metrics.get('t2i_r10', 0)*100:.2f}%)")
        
        # 平均性能
        summary_lines.append(f"### 综合性能 (CN-CLIP标准):")
        summary_lines.append(f"- Mean Recall@1: {metrics.get('mean_r1', 0):.4f} ({metrics.get('mean_r1', 0)*100:.2f}%)")
        summary_lines.append(f"- Mean Recall@5: {metrics.get('mean_r5', 0):.4f} ({metrics.get('mean_r5', 0)*100:.2f}%)")
        summary_lines.append(f"- Mean Recall@10: {metrics.get('mean_r10', 0):.4f} ({metrics.get('mean_r10', 0)*100:.2f}%)")
        
        # CN-CLIP综合得分
        cn_clip_score = (metrics.get('mean_r1', 0) + metrics.get('mean_r5', 0) + metrics.get('mean_r10', 0)) / 3.0 * 100
        summary_lines.append(f"- **CN-CLIP综合得分: {cn_clip_score:.2f}**")
        
        summary_lines.append(f"### 排名统计:")
        summary_lines.append(f"- I2T平均排名: {metrics.get('i2t_mean_rank', 0):.2f}")
        summary_lines.append(f"- T2I平均排名: {metrics.get('t2i_mean_rank', 0):.2f}\n")
    
    # 细粒度分析
    if 'finegrained_report' in results:
        report = results['finegrained_report']
        overall = report.get('overall_metrics', {})
        
        summary_lines.append("## 细粒度分析")
        summary_lines.append(f"- 整体准确率: {overall.get('accuracy', 0):.4f}")
        summary_lines.append(f"- Top-5准确率: {overall.get('top5_accuracy', 0):.4f}")
        summary_lines.append(f"- 总样本数: {overall.get('total_samples', 0)}")
        
        # 相似度分析
        sim_stats = report.get('similarity_analysis', {})
        if sim_stats:
            separation = sim_stats.get('separation', 0)
            summary_lines.append(f"- 正负样本分离度: {separation:.4f}")
        
        summary_lines.append("")
    
    # 模型性能评级 (参考CN-CLIP标准)
    if 'standard_metrics' in results:
        mean_r1 = results['standard_metrics'].get('mean_r1', 0)
        cn_clip_score = (results['standard_metrics'].get('mean_r1', 0) + 
                        results['standard_metrics'].get('mean_r5', 0) + 
                        results['standard_metrics'].get('mean_r10', 0)) / 3.0
        
        if cn_clip_score >= 0.8:
            grade = "优秀"
        elif cn_clip_score >= 0.6:
            grade = "良好"
        elif cn_clip_score >= 0.4:
            grade = "一般"
        else:
            grade = "需要改进"
        
        summary_lines.append(f"## 综合评级: {grade}")
        summary_lines.append(f"基于CN-CLIP综合得分: {cn_clip_score*100:.2f}")
        summary_lines.append(f"Mean Recall@1: {mean_r1:.4f}")
    
    # 保存摘要
    with open(f"{output_dir}/evaluation_summary.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))


if __name__ == '__main__':
    main()
