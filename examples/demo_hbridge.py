# -*- coding: utf-8 -*-
"""引导：让 examples/ 目录脚本可直接 python 运行。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：H-Bridge 三模态共享注意力融合。"""

from dexworld.encoders import HBridge


def main():
    bridge = HBridge(dim=64, heads=8)
    v = [0.1] * 64
    t = [0.5] * 64
    a = [-0.3] * 64
    out = bridge.fuse(v, t, a)
    print("H-Bridge 融合输出：")
    for k, vec in out.items():
        print(f"  {k:>8}: 均值={sum(vec)/len(vec):.4f}  维度={len(vec)}")


if __name__ == "__main__":
    main()
