# -*- coding: utf-8 -*-
"""数据变换：归一化、展平、时间对齐。"""

from __future__ import annotations

from typing import List

from ..types import VisionFrame, TactileObservation, ActionChunk


def flatten_vision(frame: VisionFrame) -> List[float]:
    return list(frame.pixels)


def flatten_tactile(obs: TactileObservation) -> List[float]:
    out: List[float] = []
    for row in obs.left:
        out.extend(row)
    for row in obs.right:
        out.extend(row)
    return out


def flatten_action(chunk: ActionChunk) -> List[float]:
    out: List[float] = []
    for row in chunk.values:
        out.extend(row)
    return out


def normalize(values: List[float], lo: float = 0.0, hi: float = 1.0) -> List[float]:
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx - mn < 1e-9:
        return [0.5 * (lo + hi)] * len(values)
    return [lo + (hi - lo) * (v - mn) / (mx - mn) for v in values]


__all__ = [
    "flatten_vision", "flatten_tactile", "flatten_action", "normalize",
]
