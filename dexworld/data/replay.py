# -*- coding: utf-8 -*-
"""经验回放缓冲。"""

from __future__ import annotations

import collections
from typing import Any, List


class ReplayBuffer:
    """固定容量环形缓冲，支持随机采样。"""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._buf: collections.deque = collections.deque(maxlen=capacity)

    def push(self, item: Any) -> None:
        self._buf.append(item)

    def sample(self, batch_size: int, seed: int = 0) -> List[Any]:
        import random
        rng = random.Random(seed)
        n = len(self._buf)
        if n == 0:
            return []
        k = min(batch_size, n)
        return [self._buf[i] for i in rng.sample(range(n), k)]

    def __len__(self) -> int:
        return len(self._buf)


__all__ = ["ReplayBuffer"]
