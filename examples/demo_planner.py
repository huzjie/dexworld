# -*- coding: utf-8 -*-
"""引导：让 examples/ 目录脚本可直接 python 运行。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：MPC 规划器选择最优抓取动作。"""

from dexworld.data.sim_engine import TactileSimEngine
from dexworld.model.world_model import DexWorldModel
from dexworld.inference.planner import MPCPlanner
from dexworld.types import ActionChunk


def main():
    engine = TactileSimEngine(seed=1)
    hist = engine.roll_trajectory(horizon=3).states
    model = DexWorldModel(vision_dim=256, tactile_dim=272, action_dim=7)
    planner = MPCPlanner(model, top_k=4)
    import random
    rng = random.Random(0)
    candidates = [ActionChunk(values=[[rng.uniform(-1, 1) for _ in range(7)]
                                      for _ in range(2)]) for _ in range(6)]
    best, score = planner.plan(hist, candidates)
    print(f"候选动作数: {len(candidates)}")
    print(f"最优动作评分（预测触觉接触强度）: {score:.4f}")
    print(f"最优动作首步: {[round(x, 3) for x in best.values[0]]}")


if __name__ == "__main__":
    main()
