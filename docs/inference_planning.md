# 推理与规划

## 世界模型 rollout

```python
from dexworld.inference.rollout import Rollout

roll = Rollout(model, seed=0)
future = roll.roll(history, horizon=16)
```

## MPC 规划

用世界模型预测每个候选动作的触觉接触质量，选最优：

```python
from dexworld.inference.planner import MPCPlanner

planner = MPCPlanner(model, top_k=8)
best_action, score = planner.plan(history, candidates)
```

## Beam search 规划

```python
from dexworld.inference.beam import BeamPlanner

beam = BeamPlanner(score_fn, beam_width=4, horizon=8)
best, score = beam.search(history, action_pool)
```

## 集成预测

```python
from dexworld.inference.ensemble import EnsemblePredictor

pred = EnsemblePredictor(model, n_seeds=5).predict(history)
```
