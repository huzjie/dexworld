# -*- coding: utf-8 -*-
"""引导：让 examples/ 目录脚本可直接 python 运行。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：世界模型预测未来视觉 + 触觉。"""

from dexworld.data.sim_engine import TactileSimEngine
from dexworld.model.world_model import DexWorldModel


def main():
    engine = TactileSimEngine(seed=7, vision_size=16)
    hist = engine.roll_trajectory(horizon=4).states
    model = DexWorldModel(vision_dim=256, tactile_dim=272, action_dim=7)
    pred = model.predict(hist, horizon=1)
    print(f"历史步数: {len(hist)}")
    print(f"预测置信度: {pred.confidence:.4f}")
    print(f"预测误差: {pred.error:.4f}")
    print(f"预测触觉区域数: 左 {len(pred.predicted_tactile.left)} / 右 {len(pred.predicted_tactile.right)}")
    print(f"预测动作维度: {pred.predicted_action.dim}")


if __name__ == "__main__":
    main()
