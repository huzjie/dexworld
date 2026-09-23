# 基准评测

## 指标

| 指标 | 含义 | 方向 |
|---|---|---|
| prediction_error | 预测与真实未来的平均 L2 误差 | 越低越好 |
| vision_error | 视觉预测误差 | 越低越好 |
| tactile_error | 触觉重建误差 | 越低越好 |
| skill | 模型能力（0~1） | 越高越好 |

## 运行

```bash
python examples/benchmark.py
```

## 消融实验

```bash
python examples/demo_ablation.py
```

对比完整模型 vs 去视觉 vs 去触觉的预测误差，验证各模态的贡献。
