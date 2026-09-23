# -*- coding: utf-8 -*-
"""残差连接。"""

from __future__ import annotations

from typing import List


def residual_add(x: List[float], fx: List[float], scale: float = 1.0) -> List[float]:
    """x + scale * f(x)。"""
    return [xi + scale * fi for xi, fi in zip(x, fx)]


class ResidualBlock:
    """残差块：输入 + 变换输出。"""

    def __init__(self, transform):
        self.transform = transform

    def forward(self, x: List[float]) -> List[float]:
        return residual_add(x, self.transform(x))


__all__ = ["residual_add", "ResidualBlock"]
