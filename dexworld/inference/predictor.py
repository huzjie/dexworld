# -*- coding: utf-8 -*-
"""触觉反馈预测器：给定动作，预测未来触觉观测。"""

from __future__ import annotations

from ..types import TactileObservation
from ..utils.logging import get_logger


class TactileFeedbackPredictor:
    """预测「做了某动作后，手会摸到什么」。"""

    def __init__(self, model):
        self.model = model
        self.log = get_logger("dexworld.predict")

    def predict_tactile(self, history) -> TactileObservation:
        pred = self.model.predict(history, horizon=1)
        return pred.predicted_tactile or TactileObservation.zeros()


__all__ = ["TactileFeedbackPredictor"]
