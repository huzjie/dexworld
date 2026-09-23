# -*- coding: utf-8 -*-
"""数据预处理管道。"""

from __future__ import annotations

from typing import Callable, List

from ..types import WorldTrajectory


class PreprocessPipeline:
    """可组合的预处理管道。"""

    def __init__(self, steps: List[Callable] = None):
        self.steps = steps or []

    def add(self, fn: Callable) -> "PreprocessPipeline":
        self.steps.append(fn)
        return self

    def apply(self, traj: WorldTrajectory) -> WorldTrajectory:
        cur = traj
        for fn in self.steps:
            cur = fn(cur)
        return cur

    def map(self, trajectories: List[WorldTrajectory]) -> List[WorldTrajectory]:
        return [self.apply(t) for t in trajectories]


__all__ = ["PreprocessPipeline"]
