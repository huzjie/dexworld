# -*- coding: utf-8 -*-
"""联合去噪扩散模型。

ME-Dex 的核心：单个扩散过程同时对视频、触觉、动作序列去噪。这里实现
- NoiseSchedule：线性 beta 噪声调度
- JointDenoiser：给定带噪的三模态 token + 条件，预测去噪后的 token

零依赖、确定性实现（seed 可控），用于训练「预测未来世界状态」。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..utils.seed import stable_seed
from ..utils.tensor import dot


@dataclass
class NoiseSchedule:
    """线性噪声调度。"""
    steps: int = 20
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def betas(self) -> List[float]:
        return [self.beta_start + (self.beta_end - self.beta_start) * t / max(1, self.steps - 1)
                for t in range(self.steps)]

    def alphas(self) -> List[float]:
        return [1.0 - b for b in self.betas()]

    def alpha_bars(self) -> List[float]:
        import math
        bars = []
        prod = 1.0
        for a in self.alphas():
            prod *= a
            bars.append(prod)
        return bars


class JointDenoiser:
    """联合去噪器：预测三模态去噪 token。"""

    def __init__(self, dim: int = 64, seed: int = 20):
        self.dim = dim
        import random
        rng = random.Random(stable_seed("denoiser", seed))
        self._w: Dict[str, List[float]] = {
            "vision": [rng.uniform(-0.5, 0.5) for _ in range(dim)],
            "tactile": [rng.uniform(-0.5, 0.5) for _ in range(dim)],
            "action": [rng.uniform(-0.5, 0.5) for _ in range(dim)],
        }

    def forward(self, noisy: Dict[str, List[float]], t: int,
                condition: Optional[List[float]] = None) -> Dict[str, List[float]]:
        """一步去噪：noisy - w * (noisy - condition) 的确定性近似。"""
        out = {}
        for mod, vec in noisy.items():
            w = self._w[mod]
            cond = condition if condition is not None else [0.0] * self.dim
            out[mod] = [
                v - w[i] * (v - cond[i]) * (0.5 + 0.5 * t / 20.0)
                for i, v in enumerate(vec)
            ]
        return out

    def sample(self, tokens: Dict[str, List[float]], condition: Optional[List[float]] = None,
               steps: int = 20, seed: int = 0) -> Dict[str, List[float]]:
        """从噪声逆扩散采样（确定性去噪近似）。"""
        import random
        rng = random.Random(seed)
        noisy = {m: [rng.gauss(0, 0.1) for _ in vec] for m, vec in tokens.items()}
        cur = dict(noisy)
        for t in range(steps):
            cur = self.forward(cur, t, condition)
        return cur


__all__ = ["NoiseSchedule", "JointDenoiser"]
