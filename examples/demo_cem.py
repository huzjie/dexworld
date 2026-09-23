# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：CEM 规划。"""
from dexworld.data.sim_engine import TactileSimEngine
from dexworld.model.world_model import DexWorldModel
from dexworld.planner.cem import CEMPlanner


def main():
    eng = TactileSimEngine(seed=2, vision_size=8)
    hist = eng.roll_trajectory(horizon=3).states
    model = DexWorldModel(vision_dim=64, tactile_dim=272, action_dim=7)

    def score(h, chunk):
        p = model.predict(h + [type(h[-1])(vision=h[-1].vision, tactile=h[-1].tactile,
                                           action=chunk, timestep=0)])
        c = 0.0
        if p.predicted_tactile:
            for row in p.predicted_tactile.left:
                c += sum(row)
        return c

    planner = CEMPlanner(score, action_dim=7, horizon=4, iterations=3, population=16)
    best, s = planner.plan(hist)
    print(f"CEM 最优动作评分: {s:.4f}")
    print(f"最优动作首步: {[round(x, 3) for x in best.values[0]]}")


if __name__ == "__main__":
    main()
