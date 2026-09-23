# -*- coding: utf-8 -*-
"""引导：让 examples/ 目录脚本可直接 python 运行。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：完整 benchmark（训练前后对比）。"""

from dexworld.data.sim_engine import TactileSimEngine
from dexworld.data.dataset import WorldDataset
from dexworld.model.world_model import DexWorldModel
from dexworld.eval.benchmark import DexWorldBenchmark
from dexworld.train.trainer import Trainer


def main():
    engine = TactileSimEngine(seed=42)
    trajs = engine.generate_dataset(episodes=12, horizon=7)
    model = DexWorldModel(vision_dim=256, tactile_dim=272, action_dim=7)
    bench = DexWorldBenchmark(model)
    before = bench.run(trajs)
    print(f"训练前: 预测误差={before['prediction_error']:.4f} 触觉误差={before['tactile_error']:.4f}")
    Trainer(model, epochs=3).fit(trajs)
    after = bench.run(trajs)
    print(f"训练后: 预测误差={after['prediction_error']:.4f} 触觉误差={after['tactile_error']:.4f}")
    print(f"skill: {before['skill']:.2f} -> {after['skill']:.2f}")


if __name__ == "__main__":
    main()
