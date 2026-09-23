# -*- coding: utf-8 -*-
"""数据增强：触觉加噪 / 时间翻转 / 区域 dropout。"""

from __future__ import annotations

from ..types import TactileObservation, WorldTrajectory, WorldState, ActionChunk


def add_tactile_noise(obs: TactileObservation, sigma: float = 0.05, seed: int = 0) -> TactileObservation:
    import random
    rng = random.Random(seed)
    def _noise(rows):
        return [[max(0.0, min(1.0, x + rng.gauss(0, sigma))) for x in row] for row in rows]
    return TactileObservation(left=_noise(obs.left), right=_noise(obs.right), dim=obs.dim)


def temporal_flip(traj: WorldTrajectory) -> WorldTrajectory:
    """时间翻转：反转状态顺序。"""
    return WorldTrajectory(states=list(reversed(traj.states)))


def region_dropout(obs: TactileObservation, p: float = 0.2, seed: int = 0) -> TactileObservation:
    """区域 dropout：随机把部分区域置零。"""
    import random
    rng = random.Random(seed)
    def _drop(rows):
        out = []
        for row in rows:
            if rng.random() < p:
                out.append([0.0] * len(row))
            else:
                out.append(list(row))
        return out
    return TactileObservation(left=_drop(obs.left), right=_drop(obs.right), dim=obs.dim)


__all__ = ["add_tactile_noise", "temporal_flip", "region_dropout"]
