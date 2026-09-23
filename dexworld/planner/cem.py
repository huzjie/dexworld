# -*- coding: utf-8 -*-
"""交叉熵方法（CEM）规划。"""

from __future__ import annotations

from typing import List, Tuple

from ..types import ActionChunk


class CEMPlanner:
    """用交叉熵方法迭代优化动作分布，选出最优动作序列。"""

    def __init__(self, score_fn, action_dim: int = 7, horizon: int = 8,
                 elite_ratio: float = 0.25, iterations: int = 5,
                 population: int = 32, seed: int = 0):
        self.score_fn = score_fn
        self.action_dim = action_dim
        self.horizon = horizon
        self.elite_ratio = elite_ratio
        self.iterations = iterations
        self.population = population
        self.seed = seed

    def plan(self, history) -> Tuple[ActionChunk, float]:
        import random
        rng = random.Random(self.seed)
        mean = [0.0] * (self.action_dim * self.horizon)
        std = [1.0] * (self.action_dim * self.horizon)
        best_chunk = None
        best_score = float("-inf")
        for _ in range(self.iterations):
            population = []
            scores = []
            for _ in range(self.population):
                vec = [rng.gauss(mean[i], std[i]) for i in range(len(mean))]
                rows = [vec[i * self.action_dim:(i + 1) * self.action_dim]
                        for i in range(self.horizon)]
                chunk = ActionChunk(values=rows, dim=self.action_dim)
                s = self.score_fn(history, chunk)
                population.append(vec)
                scores.append(s)
                if s > best_score:
                    best_score = s
                    best_chunk = chunk
            # 精英更新
            n_elite = max(1, int(self.population * self.elite_ratio))
            elite_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:n_elite]
            elite = [population[i] for i in elite_idx]
            mean = [sum(e[i] for e in elite) / n_elite for i in range(len(mean))]
            std = [(sum((e[i] - mean[i]) ** 2 for e in elite) / n_elite) ** 0.5
                   for i in range(len(mean))]
            std = [max(0.1, s) for s in std]
        return best_chunk, best_score


__all__ = ["CEMPlanner"]
