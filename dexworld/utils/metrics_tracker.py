# -*- coding: utf-8 -*-
"""指标追踪器。"""

from __future__ import annotations

from typing import Dict, List


class MetricsTracker:
    def __init__(self):
        self._history: Dict[str, List[float]] = {}

    def record(self, name: str, value: float) -> None:
        self._history.setdefault(name, []).append(value)

    def latest(self, name: str) -> float:
        vals = self._history.get(name, [])
        return vals[-1] if vals else 0.0

    def series(self, name: str) -> List[float]:
        return list(self._history.get(name, []))

    def summary(self) -> Dict[str, float]:
        out = {}
        for name, vals in self._history.items():
            if vals:
                out[f"{name}_min"] = min(vals)
                out[f"{name}_max"] = max(vals)
                out[f"{name}_mean"] = sum(vals) / len(vals)
        return out


__all__ = ["MetricsTracker"]
