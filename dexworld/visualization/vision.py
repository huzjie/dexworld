# -*- coding: utf-8 -*-
"""ASCII 视觉帧渲染。"""

from __future__ import annotations

from ..types import VisionFrame

_CHARS = " .:-=+*#%@"


def render_vision_ascii(frame: VisionFrame) -> str:
    """把视觉帧渲染成 ASCII 灰度图。"""
    h, w = frame.height, frame.width
    if not frame.pixels or h * w == 0:
        return "(empty)"
    lines = [f"视觉帧 ({w}x{h}):"]
    for i in range(h):
        row = frame.pixels[i * w:(i + 1) * w]
        lines.append("  " + "".join(_CHARS[int(max(0, min(1, p)) * 9)] for p in row))
    return "\n".join(lines)


__all__ = ["render_vision_ascii"]
