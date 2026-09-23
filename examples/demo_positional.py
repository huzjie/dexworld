# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：位置编码。"""
from dexworld.encoders.positional import sinusoidal_positional_encoding


def main():
    enc = sinusoidal_positional_encoding(seq_len=4, dim=8)
    for i, row in enumerate(enc):
        print(f"pos {i}: {[round(x, 3) for x in row[:4]]}...")


if __name__ == "__main__":
    main()
