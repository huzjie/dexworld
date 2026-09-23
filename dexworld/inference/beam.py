# -*- coding: utf-8 -*-
"""Beam search 规划：逐步保留 top-k 动作序列。"""

from __future__ import annotations

from typing import List, Tuple

from ..types import ActionChunk


class BeamPlanner:
    """对动作序列做 beam search，用评分函数保留 top-k。"""

    def __init__(self, score_fn, beam_width: int = 4, horizon: int = 8):
        self.score_fn = score_fn
        self.beam_width = beam_width
        self.horizon = horizon

    def search(self, history, action_pool: List[List[float]]) -> Tuple[ActionChunk, float]:
        """逐步展开动作序列，保留 beam_width 个最优前缀。"""
        beams: List[Tuple[List[List[float]], float]] = [([], 0.0)]
        for _ in range(self.horizon):
            candidates = []
            for prefix, _score in beams:
                for act in action_pool:
                    seq = prefix + [act]
                    chunk = ActionChunk(values=seq, dim=len(act))
                    s = self.score_fn(history, chunk)
                    candidates.append((seq, s))
            candidates.sort(key=lambda x: -x[1])
            beams = candidates[: self.beam_width]
        best_seq, best_score = beams[0]
        return ActionChunk(values=best_seq, dim=len(best_seq[0])), best_score


__all__ = ["BeamPlanner"]
