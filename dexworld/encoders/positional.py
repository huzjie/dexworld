# -*- coding: utf-8 -*-
"""位置编码（正弦）。"""

from __future__ import annotations

import math
from typing import List


def sinusoidal_positional_encoding(seq_len: int, dim: int) -> List[List[float]]:
    """生成 [seq_len, dim] 的正弦位置编码。"""
    enc = []
    for pos in range(seq_len):
        row = []
        for i in range(dim):
            angle = pos / (10000 ** (2 * (i // 2) / dim))
            row.append(math.sin(angle) if i % 2 == 0 else math.cos(angle))
        enc.append(row)
    return enc


__all__ = ["sinusoidal_positional_encoding"]
