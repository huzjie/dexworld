# -*- coding: utf-8 -*-
"""K 折交叉验证。"""

from __future__ import annotations

from typing import List

from ..types import WorldTrajectory


def kfold(trajectories: List[WorldTrajectory], k: int = 5):
    """切分 K 折，返回 (train, val) 迭代器。"""
    n = len(trajectories)
    per = max(1, n // k)
    for i in range(k):
        start = i * per
        end = start + per if i < k - 1 else n
        val = trajectories[start:end]
        train = trajectories[:start] + trajectories[end:]
        yield train, val


__all__ = ["kfold"]
