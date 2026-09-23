# -*- coding: utf-8 -*-
"""学习率调度器。"""

from __future__ import annotations


class CosineScheduler:
    def __init__(self, optimizer, base_lr: float, total_steps: int, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_steps = max(1, total_steps)
        self.min_lr = min_lr
        self._step = 0

    def step(self) -> float:
        import math
        self._step += 1
        t = min(1.0, self._step / self.total_steps)
        lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * t))
        self.optimizer.lr = lr
        return lr


class StepScheduler:
    def __init__(self, optimizer, base_lr: float, step_size: int, gamma: float = 0.1):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma
        self._step = 0

    def step(self) -> float:
        self._step += 1
        lr = self.base_lr * (self.gamma ** (self._step // self.step_size))
        self.optimizer.lr = lr
        return lr


__all__ = ["CosineScheduler", "StepScheduler"]
