# -*- coding: utf-8 -*-
"""触觉世界模型基准评测。"""

from __future__ import annotations

from typing import Dict, List

from ..utils.logging import get_logger
from .metrics import prediction_error, tactile_reconstruction_error


class DexWorldBenchmark:
    """评测：预测精度 + 触觉重建误差。"""

    def __init__(self, model):
        self.model = model
        self.log = get_logger("dexworld.eval")

    def run(self, trajectories, horizon: int = 1) -> Dict[str, float]:
        vision_errs: List[float] = []
        tactile_errs: List[float] = []
        pred_errs: List[float] = []
        n = 0
        for traj in trajectories:
            if len(traj) < 2:
                continue
            split = len(traj) // 2
            hist = traj.states[:split]
            true = traj.states[split]
            pred = self.model.predict(hist, horizon=horizon)
            pred_errs.append(pred.error)
            if pred.predicted_vision and true.vision:
                vision_errs.append(prediction_error(pred.predicted_vision, true.vision))
            if pred.predicted_tactile and true.tactile:
                tactile_errs.append(tactile_reconstruction_error(pred.predicted_tactile, true.tactile))
            n += 1
        return {
            "n": n,
            "prediction_error": sum(pred_errs) / len(pred_errs) if pred_errs else 0.0,
            "vision_error": sum(vision_errs) / len(vision_errs) if vision_errs else 0.0,
            "tactile_error": sum(tactile_errs) / len(tactile_errs) if tactile_errs else 0.0,
            "skill": self.model.skill,
        }


__all__ = ["DexWorldBenchmark"]
