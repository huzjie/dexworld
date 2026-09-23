# -*- coding: utf-8 -*-
"""通用多头注意力（零依赖）。"""

from __future__ import annotations

from typing import List

from ..utils.tensor import dot
from ..utils.mathutil import softmax


def scaled_dot_product_attention(q: List[float], keys: List[List[float]],
                                 values: List[List[float]]) -> List[float]:
    """单头注意力：q 对 keys attend，加权 values。"""
    if not keys:
        return [0.0] * (len(values[0]) if values else 1)
    d = len(q)
    scores = [dot(q, k) / (d ** 0.5) for k in keys]
    w = softmax(scores)
    dim = len(values[0])
    out = [0.0] * dim
    for wi, v in zip(w, values):
        for i in range(dim):
            out[i] += wi * v[i]
    return out


class MultiHeadAttention:
    """多头注意力：把 dim 拆成 heads 个子空间分别 attend。"""

    def __init__(self, dim: int = 64, heads: int = 8):
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

    def __call__(self, q: List[float], keys: List[List[float]],
                 values: List[List[float]]) -> List[float]:
        out = [0.0] * self.dim
        for h in range(self.heads):
            qh = q[h * self.head_dim:(h + 1) * self.head_dim]
            kh = [k[h * self.head_dim:(h + 1) * self.head_dim] for k in keys]
            vh = [v[h * self.head_dim:(h + 1) * self.head_dim] for v in values]
            oh = scaled_dot_product_attention(qh, kh, vh)
            for i, x in enumerate(oh):
                out[h * self.head_dim + i] = x
        return out


__all__ = ["scaled_dot_product_attention", "MultiHeadAttention"]
