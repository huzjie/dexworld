# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：可视化触觉热图 + 视觉帧 + 时间线。"""
from dexworld.data.sim_engine import TactileSimEngine
from dexworld.visualization import render_tactile_heatmap, render_vision_ascii, render_timeline


def main():
    eng = TactileSimEngine(seed=5, vision_size=8)
    traj = eng.roll_trajectory(horizon=5)
    print(render_tactile_heatmap(traj.states[-1].tactile, "left"))
    print()
    print(render_vision_ascii(traj.states[-1].vision))
    print()
    print(render_timeline(traj))


if __name__ == "__main__":
    main()
