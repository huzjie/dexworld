# -*- coding: utf-8 -*-
"""评测报告生成。"""

from __future__ import annotations

from typing import Dict


def build_report(metrics: Dict[str, float]) -> str:
    """把指标渲染为文本报告。"""
    lines = [
        "=" * 60,
        "DexWorld 触觉世界模型评测报告",
        "=" * 60,
        f"样本数      : {metrics.get('n', 0)}",
        f"视觉预测误差: {metrics.get('vision_error', 0):.4f}",
        f"触觉重建误差: {metrics.get('tactile_error', 0):.4f}",
        f"模型能力 skill: {metrics.get('skill', 0):.4f}",
        "-" * 60,
        "注：误差越低、skill 越高表示模型越能准确预测未来触觉与视觉状态。",
    ]
    return "\n".join(lines)


__all__ = ["build_report"]
