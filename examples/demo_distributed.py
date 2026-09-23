# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：模拟分布式训练。"""
from dexworld.data.sim_engine import TactileSimEngine
from dexworld.model.world_model import DexWorldModel
from dexworld.train.distributed import DistributedTrainer


def main():
    eng = TactileSimEngine(seed=1, vision_size=8)
    trajs = eng.generate_dataset(episodes=16, horizon=6)
    factory = lambda: DexWorldModel(vision_dim=64, tactile_dim=272, action_dim=7)
    merged = DistributedTrainer(factory, world_size=4).fit(trajs, epochs=2)
    print(f"分布式训练后融合模型 skill: {merged.skill:.3f}")


if __name__ == "__main__":
    main()
