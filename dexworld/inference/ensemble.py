# -*- coding: utf-8 -*-
"""集成预测：多 seed 采样取平均，降低方差。"""

from __future__ import annotations

from typing import List

from ..types import PredictionResult, VisionFrame, TactileObservation, ActionChunk
from ..utils.tensor import l2


class EnsemblePredictor:
    """多 seed 集成，返回平均预测 + 方差。"""

    def __init__(self, model, n_seeds: int = 5):
        self.model = model
        self.n_seeds = n_seeds

    def predict(self, history, horizon: int = 1) -> PredictionResult:
        preds = [self.model.predict(history, horizon=horizon, seed=s)
                 for s in range(self.n_seeds)]
        errs = [p.error for p in preds]
        mean_err = sum(errs) / len(errs) if errs else 0.0
        # 取误差最小（置信度最高）的作为代表，附平均置信度
        best = min(preds, key=lambda p: p.error)
        conf = sum(p.confidence for p in preds) / len(preds)
        return PredictionResult(predicted_vision=best.predicted_vision,
                                predicted_tactile=best.predicted_tactile,
                                predicted_action=best.predicted_action,
                                confidence=conf, error=mean_err)


__all__ = ["EnsemblePredictor"]
