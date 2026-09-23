# -*- coding: utf-8 -*-
"""确定性随机种子管理。"""

import random
import hashlib


def set_seed(seed: int) -> None:
    """设置全局随机种子（random 模块）。"""
    random.seed(seed)


class SeedContext:
    """上下文种子管理（可嵌套，退出后恢复）。"""

    def __init__(self, seed: int):
        self.seed = seed
        self._state = None

    def __enter__(self):
        self._state = random.getstate()
        random.seed(self.seed)
        return self

    def __exit__(self, *exc):
        random.setstate(self._state)
        return False


def stable_seed(*materials) -> int:
    """由任意材料生成稳定（跨进程）的整数种子（基于 md5，非 hash()）。"""
    h = hashlib.md5()
    for m in materials:
        h.update(str(m).encode("utf-8"))
    return int(h.hexdigest()[:8], 16)


__all__ = ["set_seed", "SeedContext", "stable_seed"]
