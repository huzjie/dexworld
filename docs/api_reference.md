# API 参考

## 核心类

### DexWorldPipeline

端到端流水线。

```python
from dexworld import DexWorldPipeline
from dexworld.config import DexWorldConfig

pipeline = DexWorldPipeline(DexWorldConfig())
result = pipeline.run()
result.metrics      # 训练指标列表
result.benchmark    # 评测指标 dict
result.report       # 文本报告
```

### DexWorldModel

触觉世界模型。

```python
model = DexWorldModel(vision_dim=1024, tactile_dim=272, action_dim=7)
pred = model.predict(history, horizon=1)
pred.predicted_vision
pred.predicted_tactile
pred.confidence
pred.error
```

### TactileSimEngine

仿真数据引擎。

```python
engine = TactileSimEngine(seed=42)
traj = engine.roll_trajectory(horizon=16)          # 单条轨迹
trajs = engine.generate_dataset(episodes=64)       # 批量
```

### MPCPlanner

模型预测控制规划器。

```python
planner = MPCPlanner(model, top_k=8)
best_action, score = planner.plan(history, candidates)
```

## 配置

全部通过 `DexWorldConfig` 或 YAML 配置：

```yaml
model:
  backend: mock
  hidden_dim: 64
  denoise_steps: 20
train:
  epochs: 5
  lr: 0.0003
  seed: 42
data:
  sim_episodes: 64
  seq_len: 8
```

详见 `config.example.yaml`。
