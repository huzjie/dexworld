# -*- coding: utf-8 -*-
"""极简张量工具：优先 numpy，缺失则用纯 Python 回退。

仅提供训练/推理用到的最小运算子集（点积、范数、加减缩放）。
"""

from __future__ import annotations

from typing import List

try:
    import numpy as _np  # type: ignore
    _HAS_NUMPY = True
except ImportError:
    _np = None
    _HAS_NUMPY = False


def has_numpy() -> bool:
    return _HAS_NUMPY


def dot(a, b) -> float:
    if _HAS_NUMPY:
        return float(_np.dot(a, b))
    return sum(x * y for x, y in zip(a, b))


def norm(a) -> float:
    return dot(a, a) ** 0.5


def l2(a, b) -> float:
    if _HAS_NUMPY:
        import numpy as np
        return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
    return norm([x - y for x, y in zip(a, b)])


def add(a, b):
    if _HAS_NUMPY:
        import numpy as np
        return (np.asarray(a, dtype=float) + np.asarray(b, dtype=float)).tolist()
    return [x + y for x, y in zip(a, b)]


def scale(a, c):
    if _HAS_NUMPY:
        import numpy as np
        return (np.asarray(a, dtype=float) * c).tolist()
    return [x * c for x in a]


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


__all__ = ["has_numpy", "dot", "norm", "l2", "add", "scale", "mean"]
