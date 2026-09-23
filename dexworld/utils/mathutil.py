# -*- coding: utf-8 -*-
"""数值工具。"""

from .tensor import dot, norm


def cosine_similarity(a, b) -> float:
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


def softmax(scores):
    import math
    if not scores:
        return []
    mx = max(scores)
    exps = [math.exp(s - mx) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def sigmoid(x) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-x))


__all__ = ["cosine_similarity", "softmax", "sigmoid"]
