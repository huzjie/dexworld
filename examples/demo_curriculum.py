# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：课程学习。"""
from dexworld.data.sim_engine import TactileSimEngine
from dexworld.data.curriculum import Curriculum, difficulty_of


def main():
    eng = TactileSimEngine(seed=4, vision_size=8)
    trajs = eng.generate_dataset(episodes=12, horizon=5)
    cur = Curriculum(trajs)
    stages = cur.stages(n_stages=3)
    print(f"轨迹总数: {len(trajs)}")
    for i, s in enumerate(stages):
        print(f"  阶段 {i + 1}: {len(s)} 条（难度递增）")
    print(f"首条难度: {difficulty_of(cur.sorted[0]):.3f}")


if __name__ == "__main__":
    main()
