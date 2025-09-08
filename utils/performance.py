"""
性能监控和优化工具
"""

import time
import torch
import psutil
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional
import numpy as np

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
        self._start_times = {}
        
    @contextmanager
    def timer(self, name: str):
        """计时上下文管理器"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(elapsed)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """获取统计信息"""
        if name not in self.metrics:
            return {}
        
        times = self.metrics[name]
        return {
            'count': len(times),
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'total': np.sum(times)
        }
    
    def print_summary(self):
        """打印性能摘要"""
        print("\n性能摘要:")
        print("-" * 60)
        for name, times in self.metrics.items():
            stats = self.get_stats(name)
            print(f"{name:.<30} {stats['mean']:.3f}s (±{stats['std']:.3f}s) x{stats['count']}")

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_memory_usage = []
        
    def start_monitoring(self, interval: float = 1.0):
        """开始监控"""
        self.monitoring = True
        self.cpu_usage.clear()
        self.memory_usage.clear()
        self.gpu_memory_usage.clear()
        
        def monitor():
            while self.monitoring:
                # CPU使用率
                self.cpu_usage.append(psutil.cpu_percent())
                
                # 内存使用率
                memory = psutil.virtual_memory()
                self.memory_usage.append(memory.percent)
                
                # GPU内存使用率
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                        usage = (allocated / total) * 100
                        if len(self.gpu_memory_usage) <= i:
                            self.gpu_memory_usage.append([])
                        self.gpu_memory_usage[i].append(usage)
                
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """获取资源使用摘要"""
        summary = {}
        
        if self.cpu_usage:
            summary['cpu'] = {
                'mean': np.mean(self.cpu_usage),
                'max': np.max(self.cpu_usage),
                'min': np.min(self.cpu_usage)
            }
        
        if self.memory_usage:
            summary['memory'] = {
                'mean': np.mean(self.memory_usage),
                'max': np.max(self.memory_usage),
                'min': np.min(self.memory_usage)
            }
        
        if self.gpu_memory_usage:
            for i, gpu_usage in enumerate(self.gpu_memory_usage):
                if gpu_usage:
                    summary[f'gpu_{i}'] = {
                        'mean': np.mean(gpu_usage),
                        'max': np.max(gpu_usage),
                        'min': np.min(gpu_usage)
                    }
        
        return summary

def optimize_torch_settings():
    """优化PyTorch设置"""
    # 启用cudnn基准测试模式
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # 设置内存分配策略
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        print("CUDA优化设置已启用")
    
    # 设置线程数
    torch.set_num_threads(min(8, torch.get_num_threads()))
    print(f"PyTorch线程数设置为: {torch.get_num_threads()}")

def get_memory_info() -> Dict[str, float]:
    """获取内存信息"""
    info = {}
    
    # 系统内存
    memory = psutil.virtual_memory()
    info['system_memory_total'] = memory.total / 1024**3
    info['system_memory_used'] = memory.used / 1024**3
    info['system_memory_percent'] = memory.percent
    
    # GPU内存
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            info[f'gpu_{i}_allocated'] = allocated
            info[f'gpu_{i}_cached'] = cached
            info[f'gpu_{i}_total'] = total
            info[f'gpu_{i}_free'] = total - allocated
    
    return info
