# -*- coding: utf-8 -*-
"""多尺度编码：把 token 分窗口编码再拼接。"""

from __future__ import annotations

from typing import List


class MultiScaleEncoder:
    """把输入向量按多个窗口尺度分别降采样后拼接。"""

    def __init__(self, scales=(1, 2, 4)):
        self.scales = scales

    def encode(self, vec: List[float]) -> List[float]:
        out: List[float] = []
        for s in self.scales:
            for i in range(0, len(vec), s):
                window = vec[i:i + s]
                out.append(sum(window) / len(window))
        return out


__all__ = ["MultiScaleEncoder"]
