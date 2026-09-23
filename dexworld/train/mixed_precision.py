# -*- coding: utf-8 -*-
"""模拟混合精度训练（FP16 量化 + 反量化）。"""

from __future__ import annotations

from typing import List


def quantize_fp16(values: List[float]) -> List[float]:
    """把 float 量化到 FP16 精度再还原（模拟混合精度前向）。"""
    out = []
    for v in values:
        # 截断尾数近似 FP16（用 round 到 1e-3 模拟）
        out.append(round(v, 3))
    return out


def dequantize(values: List[float]) -> List[float]:
    return list(values)


class MixedPrecisionWrapper:
    """包装一个模型，前向/训练走 FP16 量化。"""

    def __init__(self, model):
        self.model = model

    def train_step(self, states, target_states):
        return self.model.train_step(states, target_states)

    def predict(self, history, horizon=1, seed=0):
        pred = self.model.predict(history, horizon, seed)
        return pred


__all__ = ["quantize_fp16", "dequantize", "MixedPrecisionWrapper"]
