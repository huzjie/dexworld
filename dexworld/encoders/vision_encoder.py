# -*- coding: utf-8 -*-
"""视觉专家编码器。

把视觉帧（展平像素）编码为固定维度的视觉 embedding。零依赖实现用「随机投影 +
固定可学习权重」的简化 MLP（权重用 md5 种子确定性生成，保证跨运行一致）。
"""

from __future__ import annotations

from typing import List, Optional

from ..types import VisionFrame
from ..utils.seed import stable_seed
from ..utils.tensor import dot


class VisionEncoder:
    """视觉帧 -> embedding。"""

    def __init__(self, in_dim: int, hidden_dim: int = 64, seed: int = 1):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        import random
        rng = random.Random(stable_seed("vision_encoder", seed))
        self._w: List[float] = [rng.uniform(-0.5, 0.5) for _ in range(in_dim * hidden_dim)]

    def encode(self, frame: VisionFrame) -> List[float]:
        pixels = frame.pixels
        out = []
        n = self.in_dim
        for h in range(self.hidden_dim):
            # 取对应行切片做点积
            row = self._w[h * n:(h + 1) * n]
            if len(pixels) < n:
                pad = pixels + [0.0] * (n - len(pixels))
                out.append(dot(row, pad))
            else:
                out.append(dot(row, pixels[:n]))
        return self._norm(out)

    @staticmethod
    def _norm(v: List[float]) -> List[float]:
        import math
        s = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / s for x in v]


__all__ = ["VisionEncoder"]
