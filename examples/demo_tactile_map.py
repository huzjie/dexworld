# -*- coding: utf-8 -*-
"""引导：让 examples/ 目录脚本可直接 python 运行。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：34 区域触觉模板。"""

from dexworld.data.tactile_map import TactileMap


def main():
    tm = TactileMap()
    print(f"总区域数: {len(tm.regions)}")
    print(f"左手区域数: {len(tm.left_indices())}")
    print(f"右手区域数: {len(tm.right_indices())}")
    print("区域示例：")
    for r in tm.regions[:6]:
        print(f"  {r} -> index {tm.index(r)}")
    print("  ...")


if __name__ == "__main__":
    main()
