# -*- coding: utf-8 -*-
"""异构触觉数据映射器。

把任意传感器 + 任意机器人的触觉读数，映射到标准 34 区域模板。
核心：每个机器人末端配置一组传感器，HeteroMapper 根据「传感器 -> 区域」的
分配表（region_assignment），把传感器读数投影到对应区域。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..constants import REGIONS_PER_HAND, DEFAULT_TACTILE_DIM
from ..errors import SensorError
from ..types import TactileObservation
from .sensors import Sensor


class HeteroMapper:
    """传感器 -> 34 区域模板映射器。"""

    def __init__(self, dim: int = DEFAULT_TACTILE_DIM):
        self.dim = dim
        # 默认每只手用一个视触觉传感器（简化），可扩展为逐区域传感器
        self._sensors: Dict[str, Sensor] = {}

    def assign_sensor(self, region: str, sensor: Sensor) -> None:
        """给某个区域绑定传感器。"""
        self._sensors[region] = sensor

    def map_to_observation(self, physics_batches: Dict[str, Dict[str, float]],
                           rng) -> TactileObservation:
        """输入一批物理状态（按区域），输出 TactileObservation。"""
        left = []
        right = []
        from ..data.tactile_map import TactileMap
        tm = TactileMap()
        for is_left, hand_regions in ((True, tm.left_indices()),
                                      (False, tm.right_indices())):
            rows = []
            for idx in hand_regions:
                region = tm.regions[idx]
                sensor = self._sensors.get(region)
                if sensor is None:
                    rows.append([0.0] * self.dim)
                else:
                    physics = physics_batches.get(region, {})
                    rows.append(sensor.sense(physics, rng))
            if is_left:
                left = rows
            else:
                right = rows
        return TactileObservation(left=left, right=right, dim=self.dim)


__all__ = ["HeteroMapper"]
