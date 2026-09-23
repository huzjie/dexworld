# -*- coding: utf-8 -*-
"""dexworld 统一异常体系。"""


class DexWorldError(Exception):
    """所有 dexworld 异常的基类。"""


class ConfigError(DexWorldError):
    """配置缺失 / 非法。"""


class DataError(DexWorldError):
    """数据加载 / 预处理错误。"""


class SensorError(DataError):
    """传感器抽象错误。"""


class TactileMapError(DataError):
    """34 区域触觉模板映射错误。"""


class EncodingError(DexWorldError):
    """编码器错误。"""


class ModelError(DexWorldError):
    """模型前向 / 状态错误。"""


class TrainingError(DexWorldError):
    """训练错误。"""


class InferenceError(DexWorldError):
    """推理 / 采样错误。"""


class PlanningError(DexWorldError):
    """规划错误。"""


class BackendError(DexWorldError):
    """后端错误。"""


class CheckpointError(DexWorldError):
    """检查点读写错误。"""


__all__ = [
    "DexWorldError", "ConfigError", "DataError", "SensorError",
    "TactileMapError", "EncodingError", "ModelError", "TrainingError",
    "InferenceError", "PlanningError", "BackendError", "CheckpointError",
]
