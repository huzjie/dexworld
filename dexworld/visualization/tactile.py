# -*- coding: utf-8 -*-
"""ASCII 触觉热图渲染。"""

from __future__ import annotations

from ..types import TactileObservation

_CHARS = " .:-=+*#%@"


def _char(v: float) -> str:
    v = max(0.0, min(1.0, v))
    return _CHARS[int(v * (len(_CHARS) - 1))]


def render_tactile_heatmap(obs: TactileObservation, hand: str = "left") -> str:
    """把单手的 17 区触觉观测渲染成 ASCII 热图。"""
    rows = obs.left if hand == "left" else obs.right
    if not rows:
        return "(empty)"
    lines = [f"触觉热图 ({hand} 手):"]
    for i, row in enumerate(rows):
        # 每区显示平均强度 + 字符条
        mean = sum(row) / len(row) if row else 0.0
        bar = "".join(_char(x) for x in row)
        lines.append(f"  r{i:02d} {bar}  {mean:.2f}")
    return "\n".join(lines)


__all__ = ["render_tactile_heatmap"]
