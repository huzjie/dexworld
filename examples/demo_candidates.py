# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：候选动作生成器。"""
from dexworld.planner.candidates import random_candidates, grid_candidates, spiral_candidates


def main():
    print(f"随机候选: {len(random_candidates(8, action_dim=7))} 个")
    print(f"网格候选: {len(grid_candidates(3, action_dim=2))} 个")
    print(f"螺旋候选: {len(spiral_candidates(8, action_dim=2))} 个")


if __name__ == "__main__":
    main()
