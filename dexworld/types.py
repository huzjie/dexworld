# -*- coding: utf-8 -*-
"""核心数据类型（轻量 dataclass）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TactileObservation:
    """单帧触觉观测：左右手各 17 区，每区一个特征向量。"""
    left: List[List[float]] = field(default_factory=list)   # [17][dim]
    right: List[List[float]] = field(default_factory=list)  # [17][dim]
    dim: int = 8

    def as_dict(self) -> Dict[str, Any]:
        return {"left": self.left, "right": self.right, "dim": self.dim}

    @classmethod
    def zeros(cls, dim: int = 8) -> "TactileObservation":
        from .constants import REGIONS_PER_HAND
        z = [[0.0] * dim for _ in range(REGIONS_PER_HAND)]
        return cls(left=[r[:] for r in z], right=[r[:] for r in z], dim=dim)


@dataclass
class VisionFrame:
    """单帧视觉观测（灰度或 RGB，扁平化一维数组 + 形状）。"""
    pixels: List[float] = field(default_factory=list)
    height: int = 0
    width: int = 0
    channels: int = 1

    @classmethod
    def zeros(cls, height: int, width: int, channels: int = 1) -> "VisionFrame":
        return cls(pixels=[0.0] * (height * width * channels),
                   height=height, width=width, channels=channels)


@dataclass
class ActionChunk:
    """动作块：一段时长的末端位姿 / 关节角度序列。"""
    values: List[List[float]] = field(default_factory=list)  # [T][dim]
    dim: int = 7

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class WorldState:
    """一个时间步的世界状态 = 视频 + 触觉 + 动作。"""
    vision: Optional[VisionFrame] = None
    tactile: Optional[TactileObservation] = None
    action: Optional[ActionChunk] = None
    timestep: int = 0


@dataclass
class WorldTrajectory:
    """一段世界轨迹：连续的世界状态序列。"""
    states: List[WorldState] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.states)


@dataclass
class TrainMetrics:
    """训练指标。"""
    epoch: int = 0
    loss: float = 0.0
    vision_loss: float = 0.0
    tactile_loss: float = 0.0
    action_loss: float = 0.0
    prediction_error: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = dict(epoch=self.epoch, loss=self.loss,
                 vision_loss=self.vision_loss, tactile_loss=self.tactile_loss,
                 action_loss=self.action_loss, prediction_error=self.prediction_error)
        d.update(self.extra)
        return d


@dataclass
class PredictionResult:
    """世界模型预测结果。"""
    predicted_vision: Optional[VisionFrame] = None
    predicted_tactile: Optional[TactileObservation] = None
    predicted_action: Optional[ActionChunk] = None
    confidence: float = 1.0
    error: float = 0.0


__all__ = [
    "TactileObservation", "VisionFrame", "ActionChunk",
    "WorldState", "WorldTrajectory", "TrainMetrics", "PredictionResult",
]
