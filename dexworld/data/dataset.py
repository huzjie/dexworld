# -*- coding: utf-8 -*-
"""世界轨迹数据集与采样。"""

from __future__ import annotations

from typing import List, Optional

from ..types import WorldTrajectory
from ..utils.seed import stable_seed


class WorldDataset:
    """一组世界轨迹，提供切片采样。"""

    def __init__(self, trajectories: Optional[List[WorldTrajectory]] = None):
        self.trajectories: List[WorldTrajectory] = trajectories or []

    def __len__(self) -> int:
        return len(self.trajectories)

    def add(self, traj: WorldTrajectory) -> None:
        self.trajectories.append(traj)

    def sample_window(self, seq_len: int, seed: int = 0):
        """随机采样一段长度为 seq_len 的状态窗口。"""
        import random
        rng = random.Random(seed)
        traj = rng.choice(self.trajectories)
        n = len(traj)
        if n <= seq_len:
            return traj.states
        start = rng.randint(0, n - seq_len)
        return traj.states[start:start + seq_len]


def sample_trajectory(engine, horizon: int = 16) -> WorldTrajectory:
    """便捷：从仿真引擎采样一条轨迹。"""
    return engine.roll_trajectory(horizon)


__all__ = ["WorldDataset", "sample_trajectory"]
