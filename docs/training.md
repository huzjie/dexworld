# 训练指南

## 训练流程

1. 数据生成（自主式触觉仿真引擎）
2. 联合去噪损失（视觉 + 触觉 + 动作三路 MSE）
3. 训练循环（Trainer）+ 回调（日志 / 早停）

## 示例

```python
from dexworld.data.sim_engine import TactileSimEngine
from dexworld.model.world_model import DexWorldModel
from dexworld.train.trainer import Trainer

eng = TactileSimEngine(seed=42)
trajs = eng.generate_dataset(episodes=64, horizon=9)
model = DexWorldModel(vision_dim=64, tactile_dim=272, action_dim=7)
history = Trainer(model, epochs=5).fit(trajs)
```

## 优化器与调度

- `SGD`：带动量的随机梯度下降
- `Adam`：自适应学习率
- `CosineScheduler` / `StepScheduler`：学习率调度

## 分布式 / 混合精度

- `DistributedTrainer`：数据分片 + 能力平均（单机模拟）
- `MixedPrecisionWrapper`：FP16 量化模拟

## 检查点

```python
from dexworld.train.checkpoint import state_dict, save_checkpoint
from dexworld.utils.io import save_json

save_checkpoint({"skill": model.skill}, "checkpoints/model.json")
```
