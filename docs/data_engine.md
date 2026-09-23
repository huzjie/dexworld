# 自主式触觉数据引擎

## 背景

真实触觉数据采集昂贵且慢。ME-Dex 提出**自主式触觉数据引擎**：用仿真 + 自主策略
闭环扩充触觉数据。

## 本框架实现

`TactileSimEngine` 是一个零依赖的确定性接触-rich 抓取模拟器：

1. **物体模型**：位置、纹理、刚度、可抓取性
2. **接触模型**：末端位置与物体位置的距离 -> 接触强度 / 剪切 / 压力
3. **传感器读数**：按 34 区域生成差异化的触觉特征
4. **视觉渲染**：物体亮斑 + 末端亮斑（接触越强越亮）
5. **自主策略**：向物体方向移动 + 探索噪声，驱动轨迹

## 示例

```python
from dexworld.data.sim_engine import TactileSimEngine

engine = TactileSimEngine(seed=42, action_dim=7, vision_size=16)
traj = engine.roll_trajectory(horizon=16)
trajs = engine.generate_dataset(episodes=64, horizon=16)
```

## 确定性保证

随机性基于 md5 稳定种子，同一 `seed` 跨进程、跨机器产生完全一致的轨迹，
便于复现实验与回归测试。
