# -*- coding: utf-8 -*-
"""优化器：SGD 与 Adam（零依赖）。"""

from __future__ import annotations

from typing import List


class SGD:
    def __init__(self, lr: float = 3e-4, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self._v: dict = {}

    def step(self, params: List[List[float]], grads: List[List[float]]):
        for i, (p, g) in enumerate(zip(params, grads)):
            v = self._v.setdefault(i, [0.0] * len(p))
            for j in range(len(p)):
                v[j] = self.momentum * v[j] - self.lr * g[j]
                p[j] += v[j]


class Adam:
    def __init__(self, lr: float = 3e-4, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: dict = {}
        self._v: dict = {}
        self._t = 0

    def step(self, params: List[List[float]], grads: List[List[float]]):
        self._t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            m = self._m.setdefault(i, [0.0] * len(p))
            v = self._v.setdefault(i, [0.0] * len(p))
            for j in range(len(p)):
                m[j] = self.beta1 * m[j] + (1 - self.beta1) * g[j]
                v[j] = self.beta2 * v[j] + (1 - self.beta2) * g[j] * g[j]
                mh = m[j] / (1 - self.beta1 ** self._t)
                vh = v[j] / (1 - self.beta2 ** self._t)
                p[j] -= self.lr * mh / (vh ** 0.5 + self.eps)


__all__ = ["SGD", "Adam"]
