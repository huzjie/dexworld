# -*- coding: utf-8 -*-
"""世界模型 rollout：用模型逐步预测未来，形成一条虚拟轨迹。"""

from __future__ import annotations

from typing import List

from ..types import WorldState, WorldTrajectory
from ..utils.logging import get_logger


class Rollout:
    def __init__(self, model, seed: int = 0):
        self.model = model
        self.seed = seed
        self.log = get_logger("dexworld.inference")

    def roll(self, history, horizon: int = 16) -> WorldTrajectory:
        """从历史出发，逐步预测 horizon 步未来。"""
        predicted: List[WorldState] = []
        cur = list(history)
        for step in range(horizon):
            pred = self.model.predict(cur, horizon=1, seed=self.seed + step)
            st = WorldState(vision=pred.predicted_vision,
                            tactile=pred.predicted_tactile,
                            action=pred.predicted_action,
                            timestep=(history[-1].timestep if history else 0) + step + 1)
            predicted.append(st)
            # 用预测结果扩展上下文（滑窗，最多保留 8 步）
            cur.append(st)
            cur = cur[-8:]
        return WorldTrajectory(states=predicted)


__all__ = ["Rollout"]
