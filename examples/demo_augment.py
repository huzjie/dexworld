# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：数据增强。"""
from dexworld.data.sim_engine import TactileSimEngine
from dexworld.data.augment import add_tactile_noise, temporal_flip, region_dropout


def main():
    eng = TactileSimEngine(seed=2)
    traj = eng.roll_trajectory(horizon=4)
    obs = traj.states[0].tactile
    noisy = add_tactile_noise(obs, sigma=0.1)
    dropped = region_dropout(obs, p=0.3)
    flipped = temporal_flip(traj)
    print(f"原触觉首区均值: {sum(obs.left[0])/len(obs.left[0]):.3f}")
    print(f"加噪后首区均值: {sum(noisy.left[0])/len(noisy.left[0]):.3f}")
    print(f"dropout 后首区均值: {sum(dropped.left[0])/len(dropped.left[0]):.3f}")
    print(f"时间翻转步数: {len(flipped)}")


if __name__ == "__main__":
    main()
