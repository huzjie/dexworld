# -*- coding: utf-8 -*-
"""传感器抽象层。

ME-Dex 要处理的是异构触觉数据，因此这里定义统一传感器接口，让不同传感器
（GelSight 视触觉、触觉阵列、力/扭矩）都能输出到统一的「区域特征」表示。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from ..errors import SensorError
from ..utils.registry import Registry


class Sensor(ABC):
    """传感器基类。"""

    name: str = "base"

    @abstractmethod
    def sense(self, physics_state: Dict[str, float], rng) -> List[float]:
        """输入物理状态，输出归一化触觉特征向量（长度 = dim）。"""

    @property
    def dim(self) -> int:
        return 8


class GelSightSensor(Sensor):
    """视触觉传感器：模拟从接触几何反推的纹理/几何特征。"""
    name = "gelsight"

    @property
    def dim(self) -> int:
        return 8

    def sense(self, physics_state, rng) -> List[float]:
        contact = float(physics_state.get("contact", 0.0))
        shear = float(physics_state.get("shear", 0.0))
        texture = float(physics_state.get("texture", 0.5))
        base = [contact, shear, texture,
                contact * shear, abs(shear), texture * contact,
                min(1.0, contact + shear), max(0.0, texture - shear)]
        noise = [rng.uniform(-0.02, 0.02) for _ in range(self.dim)]
        return [max(0.0, min(1.0, b + n)) for b, n in zip(base, noise)]


class TactileArraySensor(Sensor):
    """触觉阵列传感器：按接触压力分箱输出。"""
    name = "tactile_array"

    @property
    def dim(self) -> int:
        return 8

    def sense(self, physics_state, rng) -> List[float]:
        pressure = float(physics_state.get("pressure", 0.0))
        out = []
        for i in range(self.dim):
            thresh = (i + 1) / self.dim
            out.append(1.0 if pressure >= thresh else pressure * (i + 1) / self.dim)
        noise = [rng.uniform(-0.01, 0.01) for _ in range(self.dim)]
        return [max(0.0, min(1.0, o + n)) for o, n in zip(out, noise)]


class ForceTorqueSensor(Sensor):
    """力/扭矩传感器：6 轴 F/T + 2 冗余特征。"""
    name = "force_torque"

    @property
    def dim(self) -> int:
        return 8

    def sense(self, physics_state, rng) -> List[float]:
        fx = float(physics_state.get("fx", 0.0))
        fy = float(physics_state.get("fy", 0.0))
        fz = float(physics_state.get("fz", 0.0))
        tx = float(physics_state.get("tx", 0.0))
        ty = float(physics_state.get("ty", 0.0))
        tz = float(physics_state.get("tz", 0.0))
        mag_f = (fx * fx + fy * fy + fz * fz) ** 0.5
        mag_t = (tx * tx + ty * ty + tz * tz) ** 0.5
        vals = [fx, fy, fz, tx, ty, tz, mag_f, mag_t]
        # 归一化到 [0,1]（用 tanh 近似）
        import math
        return [0.5 + 0.5 * math.tanh(v) for v in vals]


class SensorRegistry(Registry):
    def __init__(self):
        super().__init__("sensor")
        self.register("gelsight", GelSightSensor)
        self.register("tactile_array", TactileArraySensor)
        self.register("force_torque", ForceTorqueSensor)

    def create(self, name: str) -> Sensor:
        cls = self.get(name)
        return cls()


_sensor_registry = SensorRegistry()


__all__ = [
    "Sensor", "GelSightSensor", "TactileArraySensor", "ForceTorqueSensor",
    "SensorRegistry", "_sensor_registry",
]
