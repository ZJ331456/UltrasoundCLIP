# 配置差异分析：性能下降原因

## 关键差异对比

### 1. 学习率相关 ❌ 关键问题
```json
// 原版 (性能好)
"learning_rate": 4e-5,
"weight_decay": 0.02,
"gradient_clip_val": 1.0,
"warmup_steps": 1000,
"max_steps": 8000,
"eta_min": 1e-6,
"warmup_init_lr": 1e-6

// 稳定版 (性能差)
"learning_rate": 2e-5,  // 学习率减半！
"weight_decay": 0.01,
"gradient_clip_val": 0.5,
"warmup_steps": 500,   // warmup步数减半
"max_steps": 6000,
"eta_min": 1e-7,       // 最小学习率过低
"warmup_init_lr": 1e-7
```

### 2. 损失函数数值稳定性过于保守 ❌ 核心问题
```json
// 原版
"temperature": 0.07,
"label_smoothing": 0.1,
"feature_clamp_range": [-10.0, 10.0],
"similarity_clamp_range": [-50.0, 50.0]

// 稳定版 (过于保守)
"temperature": 0.1,     // 温度过高，降低区分度
"label_smoothing": 0.05,
"feature_clamp_range": [-3.0, 3.0],    // 特征范围过小！
"similarity_clamp_range": [-15.0, 15.0] // 相似度范围过小！
```

### 3. 梯度累积和EMA配置
```json
// 原版
"gradient_accumulation_steps": 4,
"ema_decay": 0.9995,
"ema_update_after_step": 200

// 稳定版
"gradient_accumulation_steps": 2,  // 有效batch size减半
"ema_decay": 0.999,               // EMA衰减太快
"ema_update_after_step": 100
```

### 4. Mask监督权重
```json
// 原版
"mask_supervision": {"weight": 0.1}

// 稳定版
"mask_supervision": {"weight": 0.05}  // 监督信号减半
```

## 问题诊断

### 主要问题：数值稳定性修复过度
1. **特征裁剪范围过小** (`[-3.0, 3.0]`) - 限制了特征表达能力
2. **相似度裁剪过保守** (`[-15.0, 15.0]`) - 降低了对比学习效果
3. **学习率过低** - 模型学习能力不足
4. **温度参数过高** (0.1 vs 0.07) - 降低了特征区分度

### 次要问题
- 有效batch size减半（梯度累积从4降到2）
- EMA配置过于激进
- Warmup步数不足

## 修复建议

### 1. 恢复关键超参数
```json
{
  "learning_rate": 3e-5,              // 中等学习率
  "temperature": 0.07,                // 恢复较低温度
  "gradient_accumulation_steps": 4,   // 恢复较大有效batch size
  "warmup_steps": 800,                // 增加warmup
  "eta_min": 5e-7                     // 适中的最小学习率
}
```

### 2. 调整数值稳定性配置（关键）
```json
{
  "numerical_stability": {
    "feature_clamp_range": [-5.0, 5.0],     // 放宽特征范围
    "similarity_clamp_range": [-25.0, 25.0], // 放宽相似度范围
    "safe_normalize_eps": 1e-7,
    "gradient_clip_loss": true
  }
}
```

### 3. 恢复mask监督权重
```json
{
  "mask_supervision": {
    "weight": 0.08  // 中等监督强度
  }
}
```

## 结论

**数值稳定性修复虽然防止了训练崩溃，但过于保守的参数设置严重限制了模型的学习能力。**

建议：
1. 创建一个平衡版本的配置
2. 逐步放宽数值限制
3. 恢复关键的学习率和损失函数参数
4. 保留必要的稳定性保护
