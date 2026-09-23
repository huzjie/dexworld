# -*- coding: utf-8 -*-
"""缓存后端：LRU 缓存预测结果。"""

from __future__ import annotations

import collections
from typing import Any, Dict


class LRUCache:
    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self._data: Dict[Any, Any] = {}
        self._order: collections.deque = collections.deque()

    def get(self, key: Any):
        if key not in self._data:
            return None
        self._order.remove(key)
        self._order.append(key)
        return self._data[key]

    def put(self, key: Any, value: Any) -> None:
        if key in self._data:
            self._order.remove(key)
        elif len(self._data) >= self.capacity:
            old = self._order.popleft()
            del self._data[old]
        self._data[key] = value
        self._order.append(key)

    def __len__(self) -> int:
        return len(self._data)


__all__ = ["LRUCache"]
