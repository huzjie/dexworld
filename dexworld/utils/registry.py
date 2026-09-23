# -*- coding: utf-8 -*-
"""通用注册表。"""

from typing import Any, Callable, Dict, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """按名称注册构造器 / 类，支持别名。"""

    def __init__(self, name: str = "registry"):
        self.name = name
        self._items: Dict[str, T] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, key: str, value: T, aliases=None) -> T:
        self._items[key] = value
        for a in (aliases or []):
            self._aliases[a] = key
        return value

    def get(self, key: str) -> T:
        key = self._aliases.get(key, key)
        if key not in self._items:
            raise KeyError(f"{self.name}: 未注册 '{key}'，可用: {sorted(self._items)}")
        return self._items[key]

    def __contains__(self, key: str) -> bool:
        return key in self._items or key in self._aliases

    def keys(self):
        return list(self._items.keys())


__all__ = ["Registry"]
