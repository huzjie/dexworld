# -*- coding: utf-8 -*-
"""引导：让 examples/ 目录脚本可直接 python 运行。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：零依赖 Mock 后端完整训练，展示「预测误差随训练下降」。"""

from dexworld import DexWorldPipeline, TrainConfig
from dexworld.config import DexWorldConfig


def main():
    cfg = DexWorldConfig()
    cfg.model.backend = "mock"
    cfg.train.epochs = 4
    cfg.data.sim_episodes = 16
    cfg.data.seq_len = 6
    result = DexWorldPipeline(cfg).run()
    print(result.summary())
    print()
    print("训练曲线（prediction_error）：")
    for m in result.metrics:
        bar = "#" * int(m.get("prediction_error", 0) * 60)
        print(f"  epoch {m['epoch']:>2}: {m.get('prediction_error', 0):.4f} {bar}")
    print()
    print(result.report)


if __name__ == "__main__":
    main()
