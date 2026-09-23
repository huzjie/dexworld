# -*- coding: utf-8 -*-
"""引导：让 examples/ 目录脚本可直接 python 运行。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：自主式触觉数据引擎生成仿真轨迹。"""

from dexworld.data.sim_engine import TactileSimEngine


def main():
    engine = TactileSimEngine(seed=42)
    traj = engine.roll_trajectory(horizon=8)
    print(f"生成轨迹步数: {len(traj)}")
    s0 = traj.states[0]
    print(f"视觉帧尺寸: {s0.vision.width}x{s0.vision.height}")
    print(f"触觉区域: 左 {len(s0.tactile.left)} / 右 {len(s0.tactile.right)}")
    # 批量
    dataset = engine.generate_dataset(episodes=8, horizon=5)
    print(f"批量生成轨迹数: {len(dataset)}")


if __name__ == "__main__":
    main()
