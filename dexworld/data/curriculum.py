# -*- coding: utf-8 -*-
"""课程学习：按难度分级组织训练样本。"""

from __future__ import annotations

from typing import List

from ..types import WorldTrajectory


def difficulty_of(traj: WorldTrajectory) -> float:
    """用平均接触强度近似难度（接触越弱越难抓住）。"""
    if not traj.states:
        return 0.0
    total = 0.0
    n = 0
    for st in traj.states:
        if st.tactile:
            for row in st.tactile.left:
                total += sum(row)
                n += len(row)
    mean_contact = total / max(1, n)
    return 1.0 - mean_contact  # 接触弱 -> 难度高


class Curriculum:
    """把轨迹按难度排序，支持从易到难逐步投喂。"""

    def __init__(self, trajectories: List[WorldTrajectory]):
        self.sorted = sorted(trajectories, key=difficulty_of)

    def stages(self, n_stages: int = 3) -> List[List[WorldTrajectory]]:
        n = len(self.sorted)
        per = max(1, n // n_stages)
        return [self.sorted[i:i + per] for i in range(0, n, per)]


__all__ = ["difficulty_of", "Curriculum"]
