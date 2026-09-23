# -*- coding: utf-8 -*-
"""候选动作生成器：随机 / 网格 / 螺旋搜索。"""

from __future__ import annotations

from typing import List

from ..types import ActionChunk


def random_candidates(n: int, action_dim: int = 7, horizon: int = 1,
                      seed: int = 0) -> List[ActionChunk]:
    import random
    rng = random.Random(seed)
    return [ActionChunk(values=[[rng.uniform(-1, 1) for _ in range(action_dim)]
                                for _ in range(horizon)], dim=action_dim)
            for _ in range(n)]


def grid_candidates(n_per_dim: int = 3, action_dim: int = 2) -> List[ActionChunk]:
    """网格采样（仅前两维，其余置零）。"""
    import itertools
    vals = [i / (n_per_dim - 1) * 2 - 1 for i in range(n_per_dim)]
    out = []
    for a, b in itertools.product(vals, vals):
        row = [a, b] + [0.0] * (action_dim - 2)
        out.append(ActionChunk(values=[row], dim=action_dim))
    return out


def spiral_candidates(n: int, action_dim: int = 2, radius: float = 1.0) -> List[ActionChunk]:
    """螺旋搜索采样。"""
    import math
    out = []
    for i in range(n):
        r = radius * math.sqrt((i + 0.5) / n)
        theta = i * 2.39996323  # 黄金角
        row = [r * math.cos(theta), r * math.sin(theta)] + [0.0] * (action_dim - 2)
        out.append(ActionChunk(values=[row], dim=action_dim))
    return out


class CandidateGenerator:
    """可配置的候选生成器。"""

    def __init__(self, method: str = "random", action_dim: int = 7, horizon: int = 1,
                 n: int = 16, seed: int = 0):
        self.method = method
        self.action_dim = action_dim
        self.horizon = horizon
        self.n = n
        self.seed = seed

    def generate(self) -> List[ActionChunk]:
        if self.method == "random":
            return random_candidates(self.n, self.action_dim, self.horizon, self.seed)
        if self.method == "grid":
            return grid_candidates(max(2, int(self.n ** 0.5)), self.action_dim)
        if self.method == "spiral":
            return spiral_candidates(self.n, self.action_dim)
        raise ValueError(f"未知候选生成方法: {self.method}")


__all__ = [
    "random_candidates", "grid_candidates", "spiral_candidates", "CandidateGenerator",
]
