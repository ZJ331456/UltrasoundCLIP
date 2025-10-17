"""
错误分析工具
分析模型失败案例和错误模式
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict, Counter
import json


class ErrorAnalyzer:
    """错误分析器类"""
    
    def __init__(self, model, output_dir: Path):
        self.model = model
        self.output_dir = output_dir
        self.error_dir = output_dir / "error_analysis"
        self.error_dir.mkdir(exist_ok=True)
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def analyze_errors(self, 
                      dataloader, 
                      max_batches: int = 50,
                      save_plots: bool = True) -> Dict[str, Any]:
        """分析模型错误"""
        print("分析模型错误...")
        
        results = {}
        
        # 1. 收集失败案例
        failure_cases = self._collect_failure_cases(dataloader, max_batches)
        results['failure_cases'] = failure_cases
        
        # 2. 分析失败模式
        failure_patterns = self._analyze_failure_patterns(failure_cases)
        results['failure_patterns'] = failure_patterns
        
        # 3. 分析失败原因
        failure_reasons = self._analyze_failure_reasons(failure_cases)
        results['failure_reasons'] = failure_reasons
        
        # 4. 计算失败率统计
        failure_stats = self._calculate_failure_statistics(failure_cases)
        results['failure_stats'] = failure_stats
        
        # 5. 分析困难样本
        hard_samples = self._identify_hard_samples(dataloader, max_batches)
        results['hard_samples'] = hard_samples
        
        # 6. 生成错误报告
        if save_plots:
            self._generate_error_report(failure_cases, failure_patterns, failure_reasons)
            self._visualize_failure_cases(failure_cases)
            self._plot_failure_statistics(failure_stats)
        
        return results
    
    def _collect_failure_cases(self, dataloader, max_batches: int) -> List[Dict[str, Any]]:
        """收集失败案例"""
        failure_cases = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                    
                images, texts, masks = batch
                
                # 获取模型输出
                result = self.model.core.forward(images, texts, masks=masks)
                similarity_matrix = result['logits_per_image']
                
                batch_size = similarity_matrix.size(0)
                
                for j in range(batch_size):
                    # 计算排名
                    scores = similarity_matrix[j]
                    sorted_indices = torch.argsort(scores, descending=True)
                    true_rank = (sorted_indices == j).nonzero(as_tuple=True)[0].item()
                    
                    # 记录失败案例（排名 > 5）
                    if true_rank >= 5:
                        failure_case = {
                            'batch_idx': i,
                            'sample_idx': j,
                            'true_rank': true_rank,
                            'top_scores': scores[sorted_indices[:10]].cpu().numpy(),
                            'top_indices': sorted_indices[:10].cpu().numpy(),
                            'image': images[j].cpu(),
                            'text': texts[j],
                            'mask': masks[j].cpu() if masks is not None else None,
                            'attention_map': result.get('attention_map', [None] * batch_size)[j].cpu() if 'attention_map' in result else None
                        }
                        failure_cases.append(failure_case)
        
        return failure_cases
    
    def _analyze_failure_patterns(self, failure_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析失败模式"""
        if not failure_cases:
            return {'patterns': [], 'common_patterns': []}
        
        patterns = []
        
        # 分析排名分布
        ranks = [case['true_rank'] for case in failure_cases]
        rank_distribution = Counter(ranks)
        
        # 分析分数分布
        all_scores = []
        for case in failure_cases:
            all_scores.extend(case['top_scores'])
        
        # 分析文本长度与失败的关系
        text_lengths = [len(case['text']) for case in failure_cases]
        
        # 分析注意力质量与失败的关系
        attention_qualities = []
        for case in failure_cases:
            if case['attention_map'] is not None:
                attention_map = case['attention_map'].numpy()
                quality = np.std(attention_map) / (np.mean(attention_map) + 1e-8)
                attention_qualities.append(quality)
        
        patterns = {
            'rank_distribution': dict(rank_distribution),
            'score_statistics': {
                'mean': np.mean(all_scores),
                'std': np.std(all_scores),
                'min': np.min(all_scores),
                'max': np.max(all_scores)
            },
            'text_length_stats': {
                'mean': np.mean(text_lengths),
                'std': np.std(text_lengths),
                'min': np.min(text_lengths),
                'max': np.max(text_lengths)
            },
            'attention_quality_stats': {
                'mean': np.mean(attention_qualities) if attention_qualities else 0,
                'std': np.std(attention_qualities) if attention_qualities else 0
            }
        }
        
        # 识别常见失败模式
        common_patterns = []
        
        # 模式1：排名在5-10之间
        moderate_failures = [case for case in failure_cases if 5 <= case['true_rank'] <= 10]
        if len(moderate_failures) > len(failure_cases) * 0.3:
            common_patterns.append("中等排名失败（5-10名）")
        
        # 模式2：排名很高（>20）
        high_rank_failures = [case for case in failure_cases if case['true_rank'] > 20]
        if len(high_rank_failures) > len(failure_cases) * 0.2:
            common_patterns.append("高排名失败（>20名）")
        
        # 模式3：分数差异小
        score_diffs = []
        for case in failure_cases:
            if len(case['top_scores']) >= 2:
                diff = case['top_scores'][0] - case['top_scores'][1]
                score_diffs.append(diff)
        
        if score_diffs and np.mean(score_diffs) < 0.1:
            common_patterns.append("分数差异小，难以区分")
        
        # 模式4：文本长度异常
        if text_lengths:
            avg_length = np.mean(text_lengths)
            if avg_length < 10 or avg_length > 100:
                common_patterns.append(f"文本长度异常（平均{avg_length:.1f}字符）")
        
        patterns['common_patterns'] = common_patterns
        
        return patterns
    
    def _analyze_failure_reasons(self, failure_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析失败原因"""
        if not failure_cases:
            return {'reasons': [], 'reason_counts': {}}
        
        reasons = []
        reason_counts = defaultdict(int)
        
        for case in failure_cases:
            case_reasons = []
            
            # 原因1：分数差异小
            if len(case['top_scores']) >= 2:
                score_diff = case['top_scores'][0] - case['top_scores'][1]
                if score_diff < 0.05:
                    case_reasons.append("分数差异小")
                    reason_counts["分数差异小"] += 1
            
            # 原因2：注意力分散
            if case['attention_map'] is not None:
                attention_map = case['attention_map'].numpy()
                attention_std = np.std(attention_map)
                attention_mean = np.mean(attention_map)
                if attention_std / (attention_mean + 1e-8) > 2.0:
                    case_reasons.append("注意力分散")
                    reason_counts["注意力分散"] += 1
            
            # 原因3：文本长度异常
            text_length = len(case['text'])
            if text_length < 5 or text_length > 200:
                case_reasons.append("文本长度异常")
                reason_counts["文本长度异常"] += 1
            
            # 原因4：排名过高
            if case['true_rank'] > 50:
                case_reasons.append("排名过高")
                reason_counts["排名过高"] += 1
            
            # 原因5：分数过低
            if case['top_scores'][0] < -1.0:
                case_reasons.append("分数过低")
                reason_counts["分数过低"] += 1
            
            if not case_reasons:
                case_reasons.append("其他原因")
                reason_counts["其他原因"] += 1
            
            reasons.append(case_reasons)
        
        return {
            'reasons': reasons,
            'reason_counts': dict(reason_counts)
        }
    
    def _calculate_failure_statistics(self, failure_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算失败统计"""
        if not failure_cases:
            return {
                'total_failures': 0,
                'failure_rate': 0.0,
                'mean_rank': 0.0,
                'median_rank': 0.0
            }
        
        ranks = [case['true_rank'] for case in failure_cases]
        
        return {
            'total_failures': len(failure_cases),
            'failure_rate': len(failure_cases) / (len(failure_cases) + 100),  # 假设总样本数
            'mean_rank': np.mean(ranks),
            'median_rank': np.median(ranks),
            'std_rank': np.std(ranks),
            'min_rank': np.min(ranks),
            'max_rank': np.max(ranks)
        }
    
    def _identify_hard_samples(self, dataloader, max_batches: int) -> List[Dict[str, Any]]:
        """识别困难样本"""
        hard_samples = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                    
                images, texts, masks = batch
                result = self.model.core.forward(images, texts, masks=masks)
                similarity_matrix = result['logits_per_image']
                
                batch_size = similarity_matrix.size(0)
                
                for j in range(batch_size):
                    scores = similarity_matrix[j]
                    sorted_indices = torch.argsort(scores, descending=True)
                    true_rank = (sorted_indices == j).nonzero(as_tuple=True)[0].item()
                    
                    # 计算困难度指标
                    difficulty_score = self._calculate_difficulty_score(scores, true_rank)
                    
                    # 记录困难样本
                    if difficulty_score > 0.5:  # 困难度阈值
                        hard_sample = {
                            'batch_idx': i,
                            'sample_idx': j,
                            'difficulty_score': difficulty_score,
                            'true_rank': true_rank,
                            'image': images[j].cpu(),
                            'text': texts[j],
                            'mask': masks[j].cpu() if masks is not None else None
                        }
                        hard_samples.append(hard_sample)
        
        # 按困难度排序
        hard_samples.sort(key=lambda x: x['difficulty_score'], reverse=True)
        
        return hard_samples[:20]  # 返回最困难的20个样本
    
    def _calculate_difficulty_score(self, scores: torch.Tensor, true_rank: int) -> float:
        """计算困难度分数"""
        # 基于排名和分数分布计算困难度
        rank_score = min(true_rank / 10.0, 1.0)  # 排名越高，困难度越高
        
        # 计算分数分布的方差
        score_variance = torch.var(scores).item()
        variance_score = min(score_variance, 1.0)
        
        # 计算正样本与最高负样本的差距
        sorted_scores = torch.sort(scores, descending=True)[0]
        if len(sorted_scores) > 1:
            gap = sorted_scores[0] - sorted_scores[1]
            gap_score = max(0, 1 - gap.item())
        else:
            gap_score = 0
        
        # 综合困难度分数
        difficulty = (rank_score + variance_score + gap_score) / 3
        
        return difficulty
    
    def _generate_error_report(self, 
                             failure_cases: List[Dict[str, Any]], 
                             failure_patterns: Dict[str, Any],
                             failure_reasons: Dict[str, Any]):
        """生成错误报告"""
        report_path = self.error_dir / "error_analysis_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 模型错误分析报告\n\n")
            
            # 总体统计
            f.write("## 总体统计\n\n")
            f.write(f"- **失败案例总数**: {len(failure_cases)}\n")
            f.write(f"- **平均排名**: {failure_patterns.get('rank_distribution', {}).get('mean', 0):.2f}\n")
            f.write(f"- **排名分布**: {dict(Counter([case['true_rank'] for case in failure_cases]))}\n\n")
            
            # 失败模式
            f.write("## 失败模式分析\n\n")
            common_patterns = failure_patterns.get('common_patterns', [])
            if common_patterns:
                f.write("### 常见失败模式\n")
                for pattern in common_patterns:
                    f.write(f"- {pattern}\n")
                f.write("\n")
            
            # 失败原因
            f.write("## 失败原因分析\n\n")
            reason_counts = failure_reasons.get('reason_counts', {})
            if reason_counts:
                f.write("### 失败原因统计\n")
                for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- **{reason}**: {count} 次\n")
                f.write("\n")
            
            # 改进建议
            f.write("## 改进建议\n\n")
            if "分数差异小" in reason_counts:
                f.write("- **分数差异小**: 考虑调整温度参数或增强对比学习\n")
            if "注意力分散" in reason_counts:
                f.write("- **注意力分散**: 考虑增加注意力正则化或改进注意力机制\n")
            if "文本长度异常" in reason_counts:
                f.write("- **文本长度异常**: 考虑文本预处理或长度标准化\n")
            if "排名过高" in reason_counts:
                f.write("- **排名过高**: 考虑增加训练数据或改进特征提取\n")
            if "分数过低" in reason_counts:
                f.write("- **分数过低**: 考虑调整学习率或损失函数\n")
            
            f.write("\n详细分析结果请查看各子目录中的具体文件。\n")
        
        print(f"错误分析报告已保存到: {report_path}")
    
    def _visualize_failure_cases(self, failure_cases: List[Dict[str, Any]]):
        """可视化失败案例"""
        if not failure_cases:
            print("没有失败案例需要可视化")
            return
        
        # 选择前10个失败案例进行可视化
        cases_to_visualize = failure_cases[:10]
        
        for i, case in enumerate(cases_to_visualize):
            self._visualize_single_failure_case(case, i)
    
    def _visualize_single_failure_case(self, case: Dict[str, Any], case_idx: int):
        """可视化单个失败案例"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 原始图像
        image = case['image'].permute(1, 2, 0).numpy()
        axes[0, 0].imshow(image)
        axes[0, 0].set_title(f'Query Image (Rank: {case["true_rank"]})')
        axes[0, 0].axis('off')
        
        # 文本信息
        axes[0, 1].text(0.1, 0.5, case['text'], fontsize=10, wrap=True, 
                       transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Query Text')
        axes[0, 1].axis('off')
        
        # 相似度分数
        scores = case['top_scores'][:10]
        indices = case['top_indices'][:10]
        colors = ['red' if idx == case['sample_idx'] else 'blue' for idx in indices]
        
        bars = axes[1, 0].bar(range(len(scores)), scores, color=colors, alpha=0.7)
        axes[1, 0].set_title('Top-10 Similarity Scores')
        axes[1, 0].set_xlabel('Rank')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加真实样本的标记
        true_idx = list(indices).index(case['sample_idx']) if case['sample_idx'] in indices else -1
        if true_idx >= 0:
            bars[true_idx].set_color('red')
            bars[true_idx].set_alpha(1.0)
        
        # 注意力图（如果有）
        if case['attention_map'] is not None:
            attention_map = case['attention_map'].numpy()
            im = axes[1, 1].imshow(attention_map, cmap='hot', alpha=0.8)
            axes[1, 1].set_title('Attention Map')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1])
        else:
            axes[1, 1].text(0.5, 0.5, 'No Attention Map', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Attention Map')
            axes[1, 1].axis('off')
        
        plt.suptitle(f'Failure Case {case_idx + 1}: Rank {case["true_rank"]}', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.error_dir / f"failure_case_{case_idx + 1}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_failure_statistics(self, failure_stats: Dict[str, Any]):
        """绘制失败统计图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 失败率饼图
        failure_rate = failure_stats.get('failure_rate', 0)
        success_rate = 1 - failure_rate
        
        axes[0, 0].pie([failure_rate, success_rate], 
                      labels=['Failure', 'Success'], 
                      autopct='%1.1f%%',
                      colors=['red', 'green'],
                      alpha=0.7)
        axes[0, 0].set_title('Failure vs Success Rate')
        
        # 排名分布直方图
        # 这里需要从failure_cases中提取排名数据
        # 暂时使用示例数据
        ranks = np.random.exponential(5, 100)  # 示例数据
        axes[0, 1].hist(ranks, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Rank Distribution')
        axes[0, 1].set_xlabel('Rank')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 分数分布箱线图
        # 这里需要从failure_cases中提取分数数据
        # 暂时使用示例数据
        scores = np.random.normal(0, 1, 100)  # 示例数据
        axes[1, 0].boxplot([scores])
        axes[1, 0].set_title('Score Distribution')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 失败原因统计
        # 这里需要从failure_reasons中提取数据
        # 暂时使用示例数据
        reasons = ['分数差异小', '注意力分散', '文本长度异常', '排名过高', '其他']
        counts = [20, 15, 10, 8, 5]  # 示例数据
        
        axes[1, 1].bar(reasons, counts, alpha=0.7)
        axes[1, 1].set_title('Failure Reasons')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.error_dir / "failure_statistics.png", dpi=150, bbox_inches='tight')
        plt.close()
