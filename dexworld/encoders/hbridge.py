# -*- coding: utf-8 -*-
"""H-Bridge 共享注意力。

ME-Dex 用 H-Bridge 让视觉、触觉、动作三模态 token 在同一语义空间对齐。
零依赖实现：把三模态 embedding 拼接后做「跨模态注意力」，即每个模态作为 query
去 attend 另两个模态的 key/value，产出桥接后的统一表示。
"""

from __future__ import annotations

from typing import Dict, List

from ..utils.tensor import dot
from ..utils.mathutil import softmax


class HBridge:
    """跨模态共享注意力桥。"""

    def __init__(self, dim: int = 64, heads: int = 8):
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

    def _attend(self, q: List[float], kv: List[List[float]]) -> List[float]:
        if not kv:
            return [0.0] * self.dim
        scores = [dot(q, k) / (self.dim ** 0.5) for k in kv]
        w = softmax(scores)
        out = [0.0] * self.dim
        for wi, v in zip(w, kv):
            for i in range(self.dim):
                out[i] += wi * v[i]
        return out

    def fuse(self, vision: List[float], tactile: List[float],
             action: List[float]) -> Dict[str, List[float]]:
        """三模态互相 attend 后融合。返回桥接后的各模态表示 + 联合表示。"""
        v = self._attend(vision, [tactile, action])
        t = self._attend(tactile, [vision, action])
        a = self._attend(action, [vision, tactile])
        joint = [(x + y + z) / 3.0 for x, y, z in zip(v, t, a)]
        return {"vision": v, "tactile": t, "action": a, "joint": joint}


__all__ = ["HBridge"]
