# -*- coding: utf-8 -*-
"""损失函数：联合去噪损失。"""

from __future__ import annotations

from typing import Dict, List

from ..utils.tensor import l2


def joint_denoise_loss(pred: Dict[str, List[float]],
                       target: Dict[str, List[float]],
                       weights: Dict[str, float] = None) -> Dict[str, float]:
    """三模态 MSE 损失 + 加权联合损失。"""
    weights = weights or {"vision": 1.0, "tactile": 1.0, "action": 1.0}
    losses = {}
    total = 0.0
    for mod in ("vision", "tactile", "action"):
        p = pred.get(mod, [])
        t = target.get(mod, [])
        if len(p) != len(t) or not p:
            mse = (l2(p, t) ** 2) / max(1, len(p))
        else:
            mse = (l2(p, t) ** 2) / len(p)
        losses[mod] = mse
        total += weights.get(mod, 1.0) * mse
    losses["total"] = total
    return losses


__all__ = ["joint_denoise_loss"]
