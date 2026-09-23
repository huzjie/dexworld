# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：经验回放缓冲。"""
from dexworld.data.replay import ReplayBuffer


def main():
    buf = ReplayBuffer(capacity=10)
    for i in range(15):
        buf.push(f"sample-{i}")
    print(f"缓冲容量: {len(buf)}")
    print(f"随机采样 3 条: {buf.sample(3, seed=0)}")


if __name__ == "__main__":
    main()
