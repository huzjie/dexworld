# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：消融实验（完整 vs 去视觉 vs 去触觉）。"""
from dexworld.data.sim_engine import TactileSimEngine
from dexworld.model.world_model import DexWorldModel
from dexworld.eval.ablation import run_ablation


def main():
    eng = TactileSimEngine(seed=8, vision_size=8)
    trajs = eng.generate_dataset(episodes=8, horizon=6)
    model = DexWorldModel(vision_dim=64, tactile_dim=272, action_dim=7)
    res = run_ablation(model, trajs)
    for k, v in res.items():
        print(f"  {k:>12}: 预测误差={v:.4f}")


if __name__ == "__main__":
    main()
