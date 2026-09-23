# -*- coding: utf-8 -*-
"""双手 34 区域触觉模板。

把异构触觉数据统一映射到标准模板：
- 每只手 17 区 = 5 指 x 3 指节（15）+ 掌心近端 / 远端（2）
- 双手合计 34 区

这是 ME-Dex 的关键设计之一：不同传感器、不同机器人采集的触觉信号形态各异，
统一投影到 34 区域模板后，模型才能在同一语义空间里学习。
"""

from __future__ import annotations

from typing import List, Tuple

from ..constants import FINGERS, PHALANGES, PALM_REGIONS, REGIONS_PER_HAND, TOTAL_REGIONS
from ..errors import TactileMapError


class TactileMap:
    """34 区域触觉模板。"""

    def __init__(self):
        self.regions: List[str] = self._build_regions()

    @staticmethod
    def _build_regions() -> List[str]:
        regions: List[str] = []
        for hand in ("left", "right"):
            for f in FINGERS:
                for p in PHALANGES:
                    regions.append(f"{hand}_{f}_{p}")
            for pr in PALM_REGIONS:
                regions.append(f"{hand}_{pr}")
        assert len(regions) == TOTAL_REGIONS, f"期望 {TOTAL_REGIONS} 区，实际 {len(regions)}"
        return regions

    def index(self, region: str) -> int:
        if region not in self.regions:
            raise TactileMapError(f"未知触觉区域: {region}")
        return self.regions.index(region)

    def hand_of(self, region: str) -> str:
        return region.split("_", 1)[0]

    def left_indices(self) -> List[int]:
        return [i for i, r in enumerate(self.regions) if r.startswith("left_")]

    def right_indices(self) -> List[int]:
        return [i for i, r in enumerate(self.regions) if r.startswith("right_")]

    def all_regions(self) -> List[str]:
        return list(self.regions)


_global_map = TactileMap()


def region_index(region: str) -> int:
    """便捷函数：区域名 -> 索引。"""
    return _global_map.index(region)


__all__ = ["TactileMap", "region_index"]
