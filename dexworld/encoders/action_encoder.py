# -*- coding: utf-8 -*-
"""动作专家编码器。

把动作块（T x dim）编码为动作 embedding。按时间步分组点积后求均值，
保留动作序列的时间结构。
"""

from __future__ import annotations

from typing import List

from ..types import ActionChunk
from ..utils.seed import stable_seed
from ..utils.tensor import dot


class ActionEncoder:
    def __init__(self, action_dim: int, hidden_dim: int = 64, seed: int = 3):
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        import random
        rng = random.Random(stable_seed("action_encoder", seed))
        self._w = [rng.uniform(-0.5, 0.5) for _ in range(action_dim * hidden_dim)]

    def encode(self, chunk: ActionChunk) -> List[float]:
        if not chunk.values:
            return [0.0] * self.hidden_dim
        acc = [0.0] * self.hidden_dim
        n = self.action_dim
        for row in chunk.values:
            for h in range(self.hidden_dim):
                w = self._w[h * n:(h + 1) * n]
                if len(row) < n:
                    pad = row + [0.0] * (n - len(row))
                    acc[h] += dot(w, pad)
                else:
                    acc[h] += dot(w, row[:n])
        t = len(chunk.values)
        acc = [x / t for x in acc]
        import math
        s = sum(x * x for x in acc) ** 0.5 or 1.0
        return [x / s for x in acc]


__all__ = ["ActionEncoder"]
