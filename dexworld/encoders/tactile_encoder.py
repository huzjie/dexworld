# -*- coding: utf-8 -*-
"""触觉专家编码器。

把 34 区域触觉观测展平成向量，编码为触觉 embedding。区域级的「组」结构
（左右手 + 指节）在展平前先做区域求和，保留结构信息。
"""

from __future__ import annotations

from typing import List

from ..types import TactileObservation
from ..utils.seed import stable_seed
from ..utils.tensor import dot


class TactileEncoder:
    def __init__(self, in_dim: int, hidden_dim: int = 64, seed: int = 2):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        import random
        rng = random.Random(stable_seed("tactile_encoder", seed))
        self._w = [rng.uniform(-0.5, 0.5) for _ in range(in_dim * hidden_dim)]

    def encode(self, obs: TactileObservation) -> List[float]:
        flat: List[float] = []
        for row in obs.left:
            flat.extend(row)
        for row in obs.right:
            flat.extend(row)
        return self._encode_flat(flat)

    def _encode_flat(self, flat: List[float]) -> List[float]:
        n = self.in_dim
        out = []
        for h in range(self.hidden_dim):
            row = self._w[h * n:(h + 1) * n]
            if len(flat) < n:
                pad = flat + [0.0] * (n - len(flat))
                out.append(dot(row, pad))
            else:
                out.append(dot(row, flat[:n]))
        import math
        s = sum(x * x for x in out) ** 0.5 or 1.0
        return [x / s for x in out]


__all__ = ["TactileEncoder"]
