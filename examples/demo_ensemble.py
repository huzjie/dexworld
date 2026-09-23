# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：集成预测。"""
from dexworld.data.sim_engine import TactileSimEngine
from dexworld.model.world_model import DexWorldModel
from dexworld.inference.ensemble import EnsemblePredictor


def main():
    eng = TactileSimEngine(seed=3, vision_size=8)
    hist = eng.roll_trajectory(horizon=3).states
    model = DexWorldModel(vision_dim=64, tactile_dim=272, action_dim=7)
    ens = EnsemblePredictor(model, n_seeds=5)
    pred = ens.predict(hist)
    print(f"集成预测置信度: {pred.confidence:.4f}  误差: {pred.error:.4f}")


if __name__ == "__main__":
    main()
