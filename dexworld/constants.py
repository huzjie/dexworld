# -*- coding: utf-8 -*-
"""全局常量与枚举。"""

from enum import Enum


class Hand(Enum):
    """左右手。"""
    LEFT = "left"
    RIGHT = "right"


class Modality(Enum):
    """三模态。"""
    VISION = "vision"
    TACTILE = "tactile"
    ACTION = "action"


class DenoiseTarget(Enum):
    """联合去噪的目标。"""
    ALL = "all"          # 视频 + 触觉 + 动作
    VISION = "vision"
    TACTILE = "tactile"
    ACTION = "action"


class BackendType(Enum):
    """模型后端。"""
    MOCK = "mock"
    OPENAI_COMPAT = "openai_compat"


# 双手 34 区域触觉模板
# 每只手 17 区 = 5 指 x 3 指节（15）+ 掌心近端 / 远端（2）
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
PHALANGES = ["distal", "middle", "proximal"]
PALM_REGIONS = ["palm_distal", "palm_proximal"]
REGIONS_PER_HAND = 5 * 3 + 2          # 17
TOTAL_REGIONS = REGIONS_PER_HAND * 2   # 34

# 默认触觉特征维度（每区域）
DEFAULT_TACTILE_DIM = 8
# 默认动作维度（7 维末端位姿 = 位置3 + 四元数4）
DEFAULT_ACTION_DIM = 7
# 默认视觉 patch 尺寸
DEFAULT_PATCH_SIZE = 16

# 扩散默认
DEFAULT_DENOISE_STEPS = 20
DEFAULT_BETA_START = 1e-4
DEFAULT_BETA_END = 0.02

__all__ = [
    "Hand", "Modality", "DenoiseTarget", "BackendType",
    "FINGERS", "PHALANGES", "PALM_REGIONS",
    "REGIONS_PER_HAND", "TOTAL_REGIONS",
    "DEFAULT_TACTILE_DIM", "DEFAULT_ACTION_DIM", "DEFAULT_PATCH_SIZE",
    "DEFAULT_DENOISE_STEPS", "DEFAULT_BETA_START", "DEFAULT_BETA_END",
]
