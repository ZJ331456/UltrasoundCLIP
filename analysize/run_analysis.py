#!/usr/bin/env python3
"""
模型分析运行脚本
基于run_eval_finetuned.sh的评估方式，运行全面的模型性能分析
"""

import os
import sys
import argparse
import json
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from analysize.model_analyzer import ModelAnalyzer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型性能分析工具')
    parser.add_argument('--config', 
                       default='config/convnext/convnext_config_2_combined_best_all_data_mask_3_v1.json',
                       help='配置文件路径')
    parser.add_argument('--model_path', 
                       default='/media/ps/data-ssd/UltrasoundRAG/CLIP/output/convnext_config_2_combined_best_all_data_mask_3_v1/best_model.pth',
                       help='模型权重路径')
    parser.add_argument('--device', 
                       default='cuda:0', 
                       help='设备')
    parser.add_argument('--output_dir', 
                       default='analysis_results/convnext_mask_3_v1', 
                       help='输出目录')
    parser.add_argument('--max_batches', 
                       type=int, 
                       default=50, 
                       help='最大分析批次数')
    parser.add_argument('--analysis_type', 
                       default='comprehensive',
                       choices=['comprehensive', 'basic', 'visualization', 'feature', 'error', 'ablation'],
                       help='分析类型')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        print("请检查配置文件路径是否正确")
        sys.exit(1)
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        print("请检查模型路径是否正确")
        print("可选模型文件:")
        # 查找可用的模型文件
        output_dir = Path(args.model_path).parent
        if output_dir.exists():
            for model_file in output_dir.glob("*.pth"):
                print(f"  - {model_file}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 打印配置信息
    print("=" * 60)
    print("启动模型性能分析")
    print("=" * 60)
    print("配置信息:")
    print(f"  - 配置文件: {args.config}")
    print(f"  - 模型权重: {args.model_path}")
    print(f"  - 使用设备: {args.device}")
    print(f"  - 输出目录: {args.output_dir}")
    print(f"  - 分析类型: {args.analysis_type}")
    print(f"  - 最大批次: {args.max_batches}")
    print("")
    
    try:
        # 创建分析器
        print("初始化模型分析器...")
        analyzer = ModelAnalyzer(
            config_path=args.config,
            model_path=args.model_path,
            device=args.device,
            output_dir=args.output_dir
        )
        
        # 根据分析类型运行不同的分析
        if args.analysis_type == 'comprehensive':
            print("运行全面分析...")
            results = analyzer.run_comprehensive_analysis(
                max_batches=args.max_batches,
                save_plots=True
            )
        elif args.analysis_type == 'basic':
            print("运行基础分析...")
            results = analyzer._evaluate_basic_performance(args.max_batches)
        elif args.analysis_type == 'visualization':
            print("运行可视化分析...")
            results = analyzer.viz.visualize_feature_space(
                analyzer.dataloader, args.max_batches, save_plots=True
            )
        elif args.analysis_type == 'feature':
            print("运行特征分析...")
            results = analyzer.feature_analyzer.analyze_features(
                analyzer.dataloader, args.max_batches, save_plots=True
            )
        elif args.analysis_type == 'error':
            print("运行错误分析...")
            results = analyzer.error_analyzer.analyze_errors(
                analyzer.dataloader, args.max_batches, save_plots=True
            )
        elif args.analysis_type == 'ablation':
            print("运行消融实验...")
            results = analyzer.ablation_study.run_ablation_study(
                analyzer.dataloader, args.max_batches, save_plots=True
            )
        
        print("")
        print("=" * 60)
        print("分析完成！")
        print("=" * 60)
        print(f"结果已保存到: {args.output_dir}")
        print("")
        print("生成的文件:")
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                print(f"  - {file_path.relative_to(output_dir)}")
        print("")
        print("可以使用以下命令查看结果:")
        print(f"  - 查看综合报告: cat {output_dir}/comprehensive_analysis_report.md")
        print(f"  - 查看错误分析: cat {output_dir}/error_analysis/error_analysis_report.md")
        print(f"  - 查看消融实验: cat {output_dir}/ablation_study/ablation_study_report.md")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
