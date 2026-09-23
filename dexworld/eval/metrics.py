# -*- coding: utf-8 -*-
"""评测指标。"""

from __future__ import annotations

from typing import List

from ..types import TactileObservation, VisionFrame
from ..utils.tensor import l2


def prediction_error(pred_vision: VisionFrame, true_vision: VisionFrame) -> float:
    return l2(pred_vision.pixels, true_vision.pixels)


def tactile_reconstruction_error(pred: TactileObservation,
                                 true: TactileObservation) -> float:
    errs = []
    for pr, tr in zip(pred.left, true.left):
        errs.append(l2(pr, tr))
    for pr, tr in zip(pred.right, true.right):
        errs.append(l2(pr, tr))
    return sum(errs) / len(errs) if errs else 0.0


def contact_score(tactile: TactileObservation) -> float:
    """触觉「接触强度」评分：所有区域特征求和归一化。"""
    total = 0.0
    n = 0
    for row in tactile.left:
        total += sum(row)
        n += len(row)
    for row in tactile.right:
        total += sum(row)
        n += len(row)
    return total / max(1, n)


__all__ = ["prediction_error", "tactile_reconstruction_error", "contact_score"]
