# -*- coding: utf-8 -*-
"""可观测：指标累积 + 快照。"""

from __future__ import annotations

import time
from typing import Dict, List


class Observability:
    def __init__(self):
        self._metrics: Dict[str, List[float]] = {}
        self._counters: Dict[str, int] = {}
        self._started = time.time()

    def record(self, name: str, value: float) -> None:
        self._metrics.setdefault(name, []).append(value)

    def incr(self, name: str, amount: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + amount

    def snapshot(self) -> Dict:
        snap = {"uptime_sec": round(time.time() - self._started, 2),
                "counters": dict(self._counters)}
        for name, vals in self._metrics.items():
            if vals:
                snap[f"{name}_mean"] = round(sum(vals) / len(vals), 6)
                snap[f"{name}_last"] = round(vals[-1], 6)
        return snap


__all__ = ["Observability"]
