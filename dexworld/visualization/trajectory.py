# -*- coding: utf-8 -*-
"""轨迹时间线渲染。"""

from __future__ import annotations

from ..types import WorldTrajectory


def render_timeline(traj: WorldTrajectory) -> str:
    """把轨迹渲染成时间线（每步的接触强度 + 视觉亮度）。"""
    lines = [f"轨迹时间线（{len(traj)} 步）:"]
    for st in traj.states:
        contact = 0.0
        if st.tactile:
            for row in st.tactile.left:
                contact += sum(row)
        brightness = 0.0
        if st.vision and st.vision.pixels:
            brightness = sum(st.vision.pixels) / len(st.vision.pixels)
        bar = "#" * int(min(20, contact * 4))
        lines.append(f"  t={st.timestep:>2} 接触={contact:5.2f} 亮度={brightness:.2f} {bar}")
    return "\n".join(lines)


__all__ = ["render_timeline"]
