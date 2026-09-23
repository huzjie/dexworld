# -*- coding: utf-8 -*-
"""视觉 / 触觉 / 动作三专家混合（Mixture of Experts）。

三个专家各自独立前向，最后由一个可学习的门控（gating）加权融合。
"""

from __future__ import annotations

from typing import Dict, List

from ..utils.seed import stable_seed
from ..utils.tensor import dot
from ..utils.mathutil import softmax


class _Expert:
    def __init__(self, dim: int, seed: int):
        import random
        rng = random.Random(stable_seed("expert", seed))
        self._w = [rng.uniform(-0.5, 0.5) for _ in range(dim)]

    def forward(self, x: List[float]) -> List[float]:
        return [wi * xi for wi, xi in zip(self._w, x)]


class ExpertMixture:
    """三专家 + 门控融合。"""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.vision_expert = _Expert(dim, 10)
        self.tactile_expert = _Expert(dim, 11)
        self.action_expert = _Expert(dim, 12)
        import random
        rng = random.Random(stable_seed("gating"))
        self._gate = [rng.uniform(-0.5, 0.5) for _ in range(dim)]

    def forward(self, tokens: Dict[str, List[float]]) -> List[float]:
        """输入三模态 token，输出融合表示。"""
        v = self.vision_expert.forward(tokens.get("vision", [0.0] * self.dim))
        t = self.tactile_expert.forward(tokens.get("tactile", [0.0] * self.dim))
        a = self.action_expert.forward(tokens.get("action", [0.0] * self.dim))
        # 门控：对每个 token 计算 gate logit，softmax 后加权
        out = [0.0] * self.dim
        for i in range(self.dim):
            g = softmax([self._gate[i] * v[i], self._gate[i] * t[i], self._gate[i] * a[i]])
            out[i] = g[0] * v[i] + g[1] * t[i] + g[2] * a[i]
        return out


__all__ = ["ExpertMixture"]
