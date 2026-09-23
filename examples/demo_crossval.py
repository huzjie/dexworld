# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：K 折交叉验证。"""
from dexworld.data.sim_engine import TactileSimEngine
from dexworld.eval.cross_validation import kfold


def main():
    eng = TactileSimEngine(seed=6, vision_size=8)
    trajs = eng.generate_dataset(episodes=10, horizon=5)
    for i, (train, val) in enumerate(kfold(trajs, k=5)):
        print(f"折 {i + 1}: train={len(train)} val={len(val)}")


if __name__ == "__main__":
    main()
