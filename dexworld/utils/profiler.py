# -*- coding: utf-8 -*-
"""性能剖析：函数计时统计。"""

from __future__ import annotations

import time
from typing import Dict, List


class Profiler:
    def __init__(self):
        self._times: Dict[str, List[float]] = {}
        self._t0: Dict[str, float] = {}

    def start(self, name: str) -> None:
        self._t0[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        dt = time.perf_counter() - self._t0.pop(name, time.perf_counter())
        self._times.setdefault(name, []).append(dt)
        return dt

    def report(self) -> str:
        lines = ["性能剖析:"]
        for name, times in sorted(self._times.items()):
            total = sum(times)
            avg = total / len(times)
            lines.append(f"  {name:>20}: n={len(times):>3} 总={total*1000:8.2f}ms 均={avg*1000:.3f}ms")
        return "\n".join(lines)


__all__ = ["Profiler"]
