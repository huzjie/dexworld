# -*- coding: utf-8 -*-
"""MPC 规划器：用世界模型做模型预测控制。

对每个候选动作序列，用世界模型预测未来触觉反馈（如「是否稳稳抓住」），
选择预测接触质量最高的动作序列作为规划结果。
"""

from __future__ import annotations

from typing import List, Tuple

from ..errors import PlanningError
from ..types import ActionChunk
from ..utils.logging import get_logger


class MPCPlanner:
    def __init__(self, model, top_k: int = 8, horizon: int = 16, seed: int = 0):
        self.model = model
        self.top_k = top_k
        self.horizon = horizon
        self.seed = seed
        self.log = get_logger("dexworld.plan")

    def _score(self, history, action_chunk: ActionChunk) -> float:
        """给候选动作打分：预测未来触觉的「接触强度」总和。"""
        # 把动作注入历史末步，让模型预测
        states = list(history)
        if states:
            last = states[-1]
            states[-1] = _clone_with_action(last, action_chunk)
        pred = self.model.predict(states, horizon=1, seed=self.seed)
        if pred.predicted_tactile is None:
            return 0.0
        score = 0.0
        for row in pred.predicted_tactile.left:
            score += sum(row)
        for row in pred.predicted_tactile.right:
            score += sum(row)
        return score

    def plan(self, history, candidates: List[ActionChunk]) -> Tuple[ActionChunk, float]:
        """从候选动作中选最优。"""
        if not candidates:
            raise PlanningError("无候选动作")
        scored = [(self._score(history, c), c) for c in candidates]
        scored.sort(key=lambda x: -x[0])
        best_score, best = scored[0]
        return best, best_score


def _clone_with_action(state, action_chunk: ActionChunk):
    """浅拷贝状态并替换动作。"""
    from ..types import WorldState
    return WorldState(vision=state.vision, tactile=state.tactile,
                      action=action_chunk, timestep=state.timestep)


__all__ = ["MPCPlanner"]
