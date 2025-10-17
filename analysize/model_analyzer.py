"""
主模型分析器
整合所有分析功能，提供统一的模型性能评估接口
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.convnext_model import ConvNeXtCLIPModel, ConvNeXtCLIPTrainingWrapper
from dataload.factory import create_dataloader
from loss import LossFactory


class ModelAnalyzer:
    """主模型分析器类"""
    
    def __init__(self, 
                 config_path: str,
                 model_path: str,
                 device: str = "cuda:0",
                 output_dir: str = "analysis_results"):
        """
        初始化模型分析器
        
        Args:
            config_path: 配置文件路径
            model_path: 模型权重路径
            device: 设备
            output_dir: 输出目录
        """
        self.config_path = config_path
        self.model_path = model_path
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.config = self._load_config()
        
        # 加载模型
        self.model = self._load_model()
        
        # 加载数据
        self.dataloader = self._load_dataloader()
        
        # 初始化分析工具
        self._init_analyzers()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    
    def _load_model(self) -> ConvNeXtCLIPTrainingWrapper:
        """加载模型"""
        print(f"加载模型: {self.model_path}")
        
        # 创建模型
        model = ConvNeXtCLIPModel(self.config)
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 处理不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 加载权重
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # 包装为训练包装器
        wrapper = ConvNeXtCLIPTrainingWrapper(model)
        wrapper.eval()
        
        print("模型加载完成")
        return wrapper
    
    def _load_dataloader(self):
        """加载数据加载器"""
        print("加载数据...")
        
        # 使用测试集
        data_config = self.config['data'].copy()
        data_config['train_json'] = data_config['valid_json']  # 使用验证集作为测试
        
        dataloader = create_dataloader(
            config=data_config,
            split='valid',
            batch_size=32,  # 分析时使用较小batch size
            num_workers=4
        )
        
        print(f"数据加载完成，共 {len(dataloader)} 个批次")
        return dataloader
    
    def _init_analyzers(self):
        """初始化分析工具"""
        from .visualization import VisualizationTools
        from .feature_analysis import FeatureAnalyzer
        from .error_analysis import ErrorAnalyzer
        from .ablation_study import AblationStudy
        
        self.viz = VisualizationTools(self.model, self.output_dir)
        self.feature_analyzer = FeatureAnalyzer(self.model, self.output_dir)
        self.error_analyzer = ErrorAnalyzer(self.model, self.output_dir)
        self.ablation_study = AblationStudy(self.model, self.output_dir)
    
    def run_comprehensive_analysis(self, 
                                 max_batches: int = 50,
                                 save_plots: bool = True) -> Dict[str, Any]:
        """
        运行全面的模型分析
        
        Args:
            max_batches: 最大分析批次数
            save_plots: 是否保存图表
            
        Returns:
            分析结果字典
        """
        print("=" * 60)
        print("开始全面模型分析")
        print("=" * 60)
        
        results = {}
        
        # 1. 基础性能评估
        print("\n1. 基础性能评估...")
        basic_metrics = self._evaluate_basic_performance(max_batches)
        results['basic_metrics'] = basic_metrics
        
        # 2. 特征质量分析
        print("\n2. 特征质量分析...")
        feature_analysis = self.feature_analyzer.analyze_features(
            self.dataloader, max_batches, save_plots
        )
        results['feature_analysis'] = feature_analysis
        
        # 3. 错误案例分析
        print("\n3. 错误案例分析...")
        error_analysis = self.error_analyzer.analyze_errors(
            self.dataloader, max_batches, save_plots
        )
        results['error_analysis'] = error_analysis
        
        # 4. 注意力可视化
        print("\n4. 注意力可视化...")
        attention_analysis = self.viz.visualize_attention_maps(
            self.dataloader, max_batches=10, save_plots=save_plots
        )
        results['attention_analysis'] = attention_analysis
        
        # 5. 特征空间可视化
        print("\n5. 特征空间可视化...")
        space_analysis = self.viz.visualize_feature_space(
            self.dataloader, max_batches, save_plots
        )
        results['space_analysis'] = space_analysis
        
        # 6. 消融实验
        print("\n6. 消融实验...")
        ablation_results = self.ablation_study.run_ablation_study(
            self.dataloader, max_batches=20, save_plots=save_plots
        )
        results['ablation_study'] = ablation_results
        
        # 7. 生成综合报告
        print("\n7. 生成综合报告...")
        self._generate_comprehensive_report(results)
        
        print("\n" + "=" * 60)
        print("分析完成！结果已保存到:", self.output_dir)
        print("=" * 60)
        
        return results
    
    def _evaluate_basic_performance(self, max_batches: int) -> Dict[str, float]:
        """评估基础性能指标"""
        from .evaluation_utils import calculate_retrieval_metrics
        
        all_image_features = []
        all_text_features = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i >= max_batches:
                    break
                    
                images, texts = batch
                image_features, text_features = self.model(images, texts)
                
                all_image_features.append(image_features.cpu())
                all_text_features.append(text_features.cpu())
        
        # 合并所有特征
        image_features = torch.cat(all_image_features, dim=0)
        text_features = torch.cat(all_text_features, dim=0)
        
        # 计算检索指标
        metrics = calculate_retrieval_metrics(image_features, text_features)
        
        return metrics
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]):
        """生成综合分析报告"""
        report_path = self.output_dir / "comprehensive_analysis_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 模型性能综合分析报告\n\n")
            f.write(f"**模型路径**: {self.model_path}\n")
            f.write(f"**配置文件**: {self.config_path}\n")
            f.write(f"**分析时间**: {torch.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 基础性能指标
            f.write("## 1. 基础性能指标\n\n")
            basic_metrics = results.get('basic_metrics', {})
            for metric, value in basic_metrics.items():
                f.write(f"- **{metric}**: {value:.4f}\n")
            f.write("\n")
            
            # 特征质量分析
            f.write("## 2. 特征质量分析\n\n")
            feature_analysis = results.get('feature_analysis', {})
            if 'similarity_separation' in feature_analysis:
                f.write(f"- **正负样本相似度分离度**: {feature_analysis['similarity_separation']:.4f}\n")
            if 'feature_diversity' in feature_analysis:
                f.write(f"- **特征多样性**: {feature_analysis['feature_diversity']:.4f}\n")
            f.write("\n")
            
            # 错误分析
            f.write("## 3. 错误分析\n\n")
            error_analysis = results.get('error_analysis', {})
            if 'failure_rate' in error_analysis:
                f.write(f"- **失败率**: {error_analysis['failure_rate']:.2%}\n")
            if 'common_failure_patterns' in error_analysis:
                f.write("- **常见失败模式**:\n")
                for pattern in error_analysis['common_failure_patterns']:
                    f.write(f"  - {pattern}\n")
            f.write("\n")
            
            # 消融实验结果
            f.write("## 4. 消融实验结果\n\n")
            ablation_results = results.get('ablation_study', {})
            if ablation_results:
                f.write("| 组件 | R@1 | R@5 | R@10 | MRR |\n")
                f.write("|------|-----|-----|------|-----|\n")
                for component, metrics in ablation_results.items():
                    f.write(f"| {component} | {metrics.get('r1', 0):.3f} | "
                           f"{metrics.get('r5', 0):.3f} | {metrics.get('r10', 0):.3f} | "
                           f"{metrics.get('mrr', 0):.3f} |\n")
            f.write("\n")
            
            # 建议
            f.write("## 5. 改进建议\n\n")
            f.write("基于分析结果，建议关注以下方面：\n\n")
            
            if basic_metrics.get('r1', 0) < 0.1:
                f.write("- **召回率较低**: 考虑调整学习率、增加训练轮数或优化损失函数\n")
            
            if feature_analysis.get('similarity_separation', 0) < 0.2:
                f.write("- **特征分离度不足**: 考虑增强对比学习或调整温度参数\n")
            
            if error_analysis.get('failure_rate', 0) > 0.3:
                f.write("- **失败率较高**: 建议增加数据增强或改进模型架构\n")
            
            f.write("\n详细分析结果请查看各子目录中的具体文件。\n")
        
        print(f"综合报告已保存到: {report_path}")


def main():
    """主函数 - 运行模型分析"""
    import argparse
    
    parser = argparse.ArgumentParser(description='模型性能分析工具')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--model_path', required=True, help='模型权重路径')
    parser.add_argument('--device', default='cuda:0', help='设备')
    parser.add_argument('--output_dir', default='analysis_results', help='输出目录')
    parser.add_argument('--max_batches', type=int, default=50, help='最大分析批次数')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = ModelAnalyzer(
        config_path=args.config,
        model_path=args.model_path,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # 运行分析
    results = analyzer.run_comprehensive_analysis(
        max_batches=args.max_batches,
        save_plots=True
    )
    
    print("分析完成！")


if __name__ == "__main__":
    main()
