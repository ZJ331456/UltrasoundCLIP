"""
消融实验工具
分析各组件对模型性能的贡献
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
from copy import deepcopy


class AblationStudy:
    """消融实验类"""
    
    def __init__(self, model, output_dir: Path):
        self.model = model
        self.output_dir = output_dir
        self.ablation_dir = output_dir / "ablation_study"
        self.ablation_dir.mkdir(exist_ok=True)
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def run_ablation_study(self, 
                          dataloader, 
                          max_batches: int = 20,
                          save_plots: bool = True) -> Dict[str, Any]:
        """运行消融实验"""
        print("运行消融实验...")
        
        results = {}
        
        # 1. 完整模型性能
        print("评估完整模型...")
        full_model_results = self._evaluate_model_performance(dataloader, max_batches)
        results['full_model'] = full_model_results
        
        # 2. 关闭attention head
        print("评估无attention head模型...")
        no_attention_results = self._evaluate_without_component('attention_head', dataloader, max_batches)
        results['no_attention_head'] = no_attention_results
        
        # 3. 关闭gating机制
        print("评估无gating机制模型...")
        no_gating_results = self._evaluate_without_component('gating', dataloader, max_batches)
        results['no_gating'] = no_gating_results
        
        # 4. 关闭多尺度特征
        print("评估无多尺度特征模型...")
        no_multi_scale_results = self._evaluate_without_component('multi_scale', dataloader, max_batches)
        results['no_multi_scale'] = no_multi_scale_results
        
        # 5. 关闭稀疏注意力
        print("评估无稀疏注意力模型...")
        no_sparse_attention_results = self._evaluate_without_component('sparse_attention', dataloader, max_batches)
        results['no_sparse_attention'] = no_sparse_attention_results
        
        # 6. 只使用全局特征
        print("评估仅全局特征模型...")
        global_only_results = self._evaluate_global_only(dataloader, max_batches)
        results['global_only'] = global_only_results
        
        # 7. 只使用局部特征
        print("评估仅局部特征模型...")
        local_only_results = self._evaluate_local_only(dataloader, max_batches)
        results['local_only'] = local_only_results
        
        # 8. 生成消融实验报告
        if save_plots:
            self._generate_ablation_report(results)
            self._plot_ablation_results(results)
            self._plot_component_importance(results)
        
        return results
    
    def _evaluate_model_performance(self, dataloader, max_batches: int) -> Dict[str, float]:
        """评估模型性能"""
        from .evaluation_utils import calculate_retrieval_metrics
        
        all_image_features = []
        all_text_features = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
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
    
    def _evaluate_without_component(self, 
                                  component: str, 
                                  dataloader, 
                                  max_batches: int) -> Dict[str, float]:
        """评估关闭特定组件后的性能"""
        # 保存原始状态
        original_state = self._save_component_state(component)
        
        try:
            # 关闭组件
            self._disable_component(component)
            
            # 评估性能
            results = self._evaluate_model_performance(dataloader, max_batches)
            
            return results
            
        finally:
            # 恢复原始状态
            self._restore_component_state(component, original_state)
    
    def _save_component_state(self, component: str) -> Dict[str, Any]:
        """保存组件状态"""
        state = {}
        
        if component == 'attention_head':
            if hasattr(self.model.core, 'use_attention_head'):
                state['use_attention_head'] = self.model.core.use_attention_head
        elif component == 'gating':
            if hasattr(self.model.core, 'use_gating'):
                state['use_gating'] = self.model.core.use_gating
        elif component == 'multi_scale':
            if hasattr(self.model.core, 'use_multi_scale'):
                state['use_multi_scale'] = self.model.core.use_multi_scale
        elif component == 'sparse_attention':
            if hasattr(self.model.core, 'use_sparse_attention'):
                state['use_sparse_attention'] = self.model.core.use_sparse_attention
        
        return state
    
    def _disable_component(self, component: str):
        """禁用组件"""
        if component == 'attention_head':
            if hasattr(self.model.core, 'use_attention_head'):
                self.model.core.use_attention_head = False
        elif component == 'gating':
            if hasattr(self.model.core, 'use_gating'):
                self.model.core.use_gating = False
        elif component == 'multi_scale':
            if hasattr(self.model.core, 'use_multi_scale'):
                self.model.core.use_multi_scale = False
        elif component == 'sparse_attention':
            if hasattr(self.model.core, 'use_sparse_attention'):
                self.model.core.use_sparse_attention = False
    
    def _restore_component_state(self, component: str, state: Dict[str, Any]):
        """恢复组件状态"""
        if component == 'attention_head':
            if 'use_attention_head' in state:
                self.model.core.use_attention_head = state['use_attention_head']
        elif component == 'gating':
            if 'use_gating' in state:
                self.model.core.use_gating = state['use_gating']
        elif component == 'multi_scale':
            if 'use_multi_scale' in state:
                self.model.core.use_multi_scale = state['use_multi_scale']
        elif component == 'sparse_attention':
            if 'use_sparse_attention' in state:
                self.model.core.use_sparse_attention = state['use_sparse_attention']
    
    def _evaluate_global_only(self, dataloader, max_batches: int) -> Dict[str, float]:
        """评估仅使用全局特征的性能"""
        # 保存原始投影层
        original_image_projection = self.model.core.image_projection
        
        try:
            # 使用全局投影层
            self.model.core.image_projection = self.model.core.image_projection_global
            
            # 评估性能
            results = self._evaluate_model_performance(dataloader, max_batches)
            
            return results
            
        finally:
            # 恢复原始投影层
            self.model.core.image_projection = original_image_projection
    
    def _evaluate_local_only(self, dataloader, max_batches: int) -> Dict[str, float]:
        """评估仅使用局部特征的性能"""
        # 保存原始投影层
        original_image_projection = self.model.core.image_projection
        
        try:
            # 使用局部投影层
            self.model.core.image_projection = self.model.core.image_projection_local
            
            # 评估性能
            results = self._evaluate_model_performance(dataloader, max_batches)
            
            return results
            
        finally:
            # 恢复原始投影层
            self.model.core.image_projection = original_image_projection
    
    def _generate_ablation_report(self, results: Dict[str, Any]):
        """生成消融实验报告"""
        report_path = self.ablation_dir / "ablation_study_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 消融实验报告\n\n")
            
            # 实验概述
            f.write("## 实验概述\n\n")
            f.write("本实验通过逐步移除模型中的各个组件，分析各组件对模型性能的贡献。\n\n")
            
            # 结果表格
            f.write("## 实验结果\n\n")
            f.write("| 组件配置 | R@1 | R@5 | R@10 | MRR | mAP@10 |\n")
            f.write("|----------|-----|-----|------|-----|--------|\n")
            
            for config_name, metrics in results.items():
                f.write(f"| {config_name} | "
                       f"{metrics.get('r1', 0):.3f} | "
                       f"{metrics.get('r5', 0):.3f} | "
                       f"{metrics.get('r10', 0):.3f} | "
                       f"{metrics.get('mrr', 0):.3f} | "
                       f"{metrics.get('mAP@10', 0):.3f} |\n")
            
            f.write("\n")
            
            # 组件重要性分析
            f.write("## 组件重要性分析\n\n")
            
            if 'full_model' in results:
                full_model_r1 = results['full_model'].get('r1', 0)
                
                # 分析各组件的重要性
                component_importance = {}
                
                for config_name, metrics in results.items():
                    if config_name != 'full_model':
                        r1 = metrics.get('r1', 0)
                        importance = full_model_r1 - r1
                        component_importance[config_name] = importance
                
                # 按重要性排序
                sorted_importance = sorted(component_importance.items(), 
                                        key=lambda x: x[1], reverse=True)
                
                f.write("### 组件重要性排序（基于R@1下降）\n\n")
                for i, (component, importance) in enumerate(sorted_importance, 1):
                    f.write(f"{i}. **{component}**: {importance:.4f}\n")
                
                f.write("\n")
                
                # 分析结果
                f.write("### 分析结果\n\n")
                
                if sorted_importance:
                    most_important = sorted_importance[0]
                    least_important = sorted_importance[-1]
                    
                    f.write(f"- **最重要组件**: {most_important[0]} (R@1下降 {most_important[1]:.4f})\n")
                    f.write(f"- **最不重要组件**: {least_important[0]} (R@1下降 {least_important[1]:.4f})\n")
                    
                    # 给出建议
                    f.write("\n### 改进建议\n\n")
                    
                    if most_important[1] > 0.05:
                        f.write(f"- **{most_important[0]}** 对性能影响很大，建议保留并进一步优化\n")
                    
                    if least_important[1] < 0.01:
                        f.write(f"- **{least_important[0]}** 对性能影响较小，可以考虑移除以简化模型\n")
                    
                    # 分析是否有组件组合效果
                    f.write("\n### 组件组合效果\n\n")
                    
                    # 计算组合效果（这里简化处理）
                    f.write("- 建议进一步分析组件间的交互作用\n")
                    f.write("- 考虑使用更精细的消融实验来理解组件组合效果\n")
            
            f.write("\n详细分析结果请查看各子目录中的具体文件。\n")
        
        print(f"消融实验报告已保存到: {report_path}")
    
    def _plot_ablation_results(self, results: Dict[str, Any]):
        """绘制消融实验结果"""
        if not results:
            return
        
        # 提取指标
        configs = list(results.keys())
        metrics = ['r1', 'r5', 'r10', 'mrr']
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results[config].get(metric, 0) for config in configs]
            
            # 绘制柱状图
            bars = axes[i].bar(range(len(configs)), values, alpha=0.7)
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].set_xticks(range(len(configs)))
            axes[i].set_xticklabels(configs, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)
            
            # 添加数值标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.ablation_dir / "ablation_results.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_component_importance(self, results: Dict[str, Any]):
        """绘制组件重要性图"""
        if 'full_model' not in results:
            return
        
        full_model_r1 = results['full_model'].get('r1', 0)
        
        # 计算各组件的重要性
        component_importance = {}
        for config_name, metrics in results.items():
            if config_name != 'full_model':
                r1 = metrics.get('r1', 0)
                importance = full_model_r1 - r1
                component_importance[config_name] = importance
        
        if not component_importance:
            return
        
        # 按重要性排序
        sorted_components = sorted(component_importance.items(), 
                                key=lambda x: x[1], reverse=True)
        
        components, importances = zip(*sorted_components)
        
        # 绘制水平柱状图
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(components)), importances, alpha=0.7)
        plt.yticks(range(len(components)), components)
        plt.xlabel('R@1 Drop')
        plt.title('Component Importance (R@1 Drop from Full Model)')
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                    f'{width:.4f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.ablation_dir / "component_importance.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_fine_grained_ablation(self, 
                                dataloader, 
                                max_batches: int = 10) -> Dict[str, Any]:
        """运行细粒度消融实验"""
        print("运行细粒度消融实验...")
        
        results = {}
        
        # 1. 不同的温度参数
        print("测试不同温度参数...")
        temperatures = [0.01, 0.05, 0.07, 0.1, 0.2]
        temp_results = {}
        
        for temp in temperatures:
            # 修改温度参数
            original_temp = getattr(self.model.core, 'logit_scale', None)
            if hasattr(self.model.core, 'logit_scale'):
                self.model.core.logit_scale.data = torch.log(torch.tensor(1.0 / temp))
            
            # 评估性能
            metrics = self._evaluate_model_performance(dataloader, max_batches)
            temp_results[f'temp_{temp}'] = metrics
            
            # 恢复原始温度
            if original_temp is not None:
                self.model.core.logit_scale.data = original_temp
        
        results['temperature_ablation'] = temp_results
        
        # 2. 不同的注意力头数
        print("测试不同注意力头数...")
        attention_heads = [4, 8, 12, 16]
        head_results = {}
        
        for num_heads in attention_heads:
            # 修改注意力头数
            original_attn = self.model.core.attn_layer
            new_attn = torch.nn.MultiheadAttention(
                embed_dim=original_attn.embed_dim,
                num_heads=num_heads,
                dropout=original_attn.dropout,
                batch_first=True
            )
            self.model.core.attn_layer = new_attn
            
            # 评估性能
            metrics = self._evaluate_model_performance(dataloader, max_batches)
            head_results[f'heads_{num_heads}'] = metrics
            
            # 恢复原始注意力层
            self.model.core.attn_layer = original_attn
        
        results['attention_heads_ablation'] = head_results
        
        # 3. 不同的dropout率
        print("测试不同dropout率...")
        dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
        dropout_results = {}
        
        for dropout_rate in dropout_rates:
            # 修改dropout率
            original_dropout = self.model.core.dropout
            new_dropout = torch.nn.Dropout(dropout_rate)
            self.model.core.dropout = new_dropout
            
            # 评估性能
            metrics = self._evaluate_model_performance(dataloader, max_batches)
            dropout_results[f'dropout_{dropout_rate}'] = metrics
            
            # 恢复原始dropout
            self.model.core.dropout = original_dropout
        
        results['dropout_ablation'] = dropout_results
        
        # 生成细粒度消融报告
        self._generate_fine_grained_report(results)
        
        return results
    
    def _generate_fine_grained_report(self, results: Dict[str, Any]):
        """生成细粒度消融报告"""
        report_path = self.ablation_dir / "fine_grained_ablation_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 细粒度消融实验报告\n\n")
            
            # 温度参数分析
            if 'temperature_ablation' in results:
                f.write("## 温度参数分析\n\n")
                f.write("| 温度 | R@1 | R@5 | R@10 | MRR |\n")
                f.write("|------|-----|-----|------|-----|\n")
                
                temp_results = results['temperature_ablation']
                for temp_name, metrics in temp_results.items():
                    temp_value = temp_name.split('_')[1]
                    f.write(f"| {temp_value} | "
                           f"{metrics.get('r1', 0):.3f} | "
                           f"{metrics.get('r5', 0):.3f} | "
                           f"{metrics.get('r10', 0):.3f} | "
                           f"{metrics.get('mrr', 0):.3f} |\n")
                
                f.write("\n")
            
            # 注意力头数分析
            if 'attention_heads_ablation' in results:
                f.write("## 注意力头数分析\n\n")
                f.write("| 头数 | R@1 | R@5 | R@10 | MRR |\n")
                f.write("|------|-----|-----|------|-----|\n")
                
                head_results = results['attention_heads_ablation']
                for head_name, metrics in head_results.items():
                    head_count = head_name.split('_')[1]
                    f.write(f"| {head_count} | "
                           f"{metrics.get('r1', 0):.3f} | "
                           f"{metrics.get('r5', 0):.3f} | "
                           f"{metrics.get('r10', 0):.3f} | "
                           f"{metrics.get('mrr', 0):.3f} |\n")
                
                f.write("\n")
            
            # Dropout率分析
            if 'dropout_ablation' in results:
                f.write("## Dropout率分析\n\n")
                f.write("| Dropout | R@1 | R@5 | R@10 | MRR |\n")
                f.write("|---------|-----|-----|------|-----|\n")
                
                dropout_results = results['dropout_ablation']
                for dropout_name, metrics in dropout_results.items():
                    dropout_rate = dropout_name.split('_')[1]
                    f.write(f"| {dropout_rate} | "
                           f"{metrics.get('r1', 0):.3f} | "
                           f"{metrics.get('r5', 0):.3f} | "
                           f"{metrics.get('r10', 0):.3f} | "
                           f"{metrics.get('mrr', 0):.3f} |\n")
                
                f.write("\n")
            
            f.write("详细分析结果请查看各子目录中的具体文件。\n")
        
        print(f"细粒度消融报告已保存到: {report_path}")
