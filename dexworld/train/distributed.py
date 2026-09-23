# -*- coding: utf-8 -*-
"""模拟分布式训练：数据分片 + 梯度平均（单机模拟）。"""

from __future__ import annotations

from typing import List


class DistributedTrainer:
    """把数据切成 n 个分片，各自训练后平均「能力」参数。"""

    def __init__(self, model_factory, world_size: int = 4):
        self.model_factory = model_factory
        self.world_size = world_size
        self.workers = [model_factory() for _ in range(world_size)]

    def fit(self, trajectories: List, epochs: int = 2):
        from .trainer import Trainer
        n = len(trajectories)
        chunk = max(1, n // self.world_size)
        for w in range(self.world_size):
            part = trajectories[w * chunk:(w + 1) * chunk]
            Trainer(self.workers[w], epochs=epochs).fit(part)
        # 梯度/能力平均：取 skill 均值作为融合
        avg_skill = sum(m.skill for m in self.workers) / len(self.workers)
        for m in self.workers:
            m._skill = avg_skill
        return self.workers[0]


__all__ = ["DistributedTrainer"]
