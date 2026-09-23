# -*- coding: utf-8 -*-
"""消融实验：分别移除某模态，观察预测误差变化。"""

from __future__ import annotations

from typing import Dict, List

from ..types import WorldState, WorldTrajectory, TactileObservation, VisionFrame


def _blank_states(states: List[WorldState], blank: str) -> List[WorldState]:
    out = []
    for st in states:
        vision = st.vision
        tactile = st.tactile
        if blank == "vision":
            vision = VisionFrame.zeros(st.vision.height, st.vision.width) if st.vision else None
        elif blank == "tactile":
            tactile = TactileObservation.zeros() if st.tactile else None
        out.append(WorldState(vision=vision, tactile=tactile,
                              action=st.action, timestep=st.timestep))
    return out


def run_ablation(model, trajectories: List[WorldTrajectory]) -> Dict[str, float]:
    """完整 vs 去视觉 vs 去触觉 的预测误差对比。"""
    from .benchmark import DexWorldBenchmark
    bench = DexWorldBenchmark(model)
    results = {"full": bench.run(trajectories)["prediction_error"]}

    # 去视觉
    blank_vision = [WorldTrajectory(states=_blank_states(t.states, "vision"))
                    for t in trajectories]
    results["no_vision"] = bench.run(blank_vision)["prediction_error"]

    # 去触觉
    blank_tactile = [WorldTrajectory(states=_blank_states(t.states, "tactile"))
                     for t in trajectories]
    results["no_tactile"] = bench.run(blank_tactile)["prediction_error"]
    return results


__all__ = ["run_ablation"]
