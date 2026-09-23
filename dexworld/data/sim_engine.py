# -*- coding: utf-8 -*-
"""自主式触觉数据引擎。

ME-Dex 提出「自主式触觉数据引擎」来扩充仿真数据：模拟器生成物理接触状态，
自主策略（随机 / 探索）驱动动作，传感器读数 + 视觉渲染构成一条世界轨迹。
本引擎零依赖实现一个简化的接触-rich 抓取模拟器。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..constants import TOTAL_REGIONS
from ..types import (
    ActionChunk, TactileObservation, VisionFrame, WorldState, WorldTrajectory,
)
from ..utils.seed import stable_seed
from .sensors import SensorRegistry
from .hetero import HeteroMapper


@dataclass
class SimObject:
    """被操作物体：位姿 + 纹理 + 刚度。"""
    x: float = 0.0
    y: float = 0.0
    stiffness: float = 0.5
    texture: float = 0.5
    graspable: bool = True


class TactileSimEngine:
    """简化接触-rich 抓取模拟器（确定性可复现）。"""

    def __init__(self, seed: int = 42, action_dim: int = 7,
                 vision_size: int = 16, tactile_dim: int = 8):
        import random
        self.rng = random.Random(seed)
        self.action_dim = action_dim
        self.vision_size = vision_size
        self.tactile_dim = tactile_dim
        self._sensor_reg = SensorRegistry()
        self.mapper = HeteroMapper(dim=tactile_dim)
        from .tactile_map import TactileMap
        tm = TactileMap()
        for region in tm.regions:
            self.mapper.assign_sensor(region, self._sensor_reg.create("gelsight"))
        self.object = SimObject()

    def reset(self) -> WorldState:
        self.object = SimObject(
            x=self.rng.uniform(-0.5, 0.5), y=self.rng.uniform(-0.5, 0.5),
            stiffness=self.rng.uniform(0.2, 0.9), texture=self.rng.uniform(0.2, 0.9),
        )
        return self._observe(action=[0.0] * self.action_dim, timestep=0)

    def _observe(self, action: List[float], timestep: int) -> WorldState:
        # 接触强度由末端位置与物体位置的距离决定
        ex = float(action[0]) if action else 0.0
        ey = float(action[1]) if action else 0.0
        dist = ((ex - self.object.x) ** 2 + (ey - self.object.y) ** 2) ** 0.5
        contact = max(0.0, min(1.0, 1.0 - dist))
        shear = abs(ex - self.object.x) * self.object.texture
        physics = {}
        for i in range(TOTAL_REGIONS):
            # 每区域物理状态略作区分，模拟不同接触区域
            region_scale = 0.8 + 0.2 * (i % 5) / 4.0
            physics[str(i)] = {
                "contact": contact * region_scale,
                "shear": shear * region_scale,
                "texture": self.object.texture,
                "pressure": contact * self.object.stiffness,
            }
        # mapper 按区域名索引，这里用区域名 key 重建
        from .tactile_map import TactileMap
        tm = TactileMap()
        physics_by_region = {}
        for i, region in enumerate(tm.regions):
            physics_by_region[region] = physics[str(i)]
        tactile = self.mapper.map_to_observation(physics_by_region, self.rng)
        # 视觉帧：接触强则亮斑靠近物体
        vision = self._render_vision(contact, ex, ey)
        return WorldState(
            vision=vision, tactile=tactile,
            action=ActionChunk(values=[list(action)], dim=self.action_dim),
            timestep=timestep,
        )

    def _render_vision(self, contact: float, ex: float, ey: float) -> VisionFrame:
        s = self.vision_size
        pixels = []
        for i in range(s):
            for j in range(s):
                cx, cy = (j / s - 0.5) * 2.0, (i / s - 0.5) * 2.0
                # 物体亮斑 + 末端亮斑
                obj = max(0.0, 1.0 - ((cx - self.object.x) ** 2 + (cy - self.object.y) ** 2) * 8)
                eff = max(0.0, 1.0 - ((cx - ex) ** 2 + (cy - ey) ** 2) * 8)
                val = max(obj * 0.7, eff * (0.3 + 0.7 * contact))
                pixels.append(max(0.0, min(1.0, val)))
        return VisionFrame(pixels=pixels, height=s, width=s, channels=1)

    def step(self, action: List[float], timestep: int) -> WorldState:
        return self._observe(action, timestep)

    def roll_trajectory(self, horizon: int = 16) -> WorldTrajectory:
        """随机策略 rollout 一条轨迹。"""
        states = [self.reset()]
        for t in range(1, horizon):
            # 简单探索策略：向物体方向移动 + 噪声
            dx = self.object.x - float(states[-1].action.values[-1][0])
            dy = self.object.y - float(states[-1].action.values[-1][1])
            a = [0.0] * self.action_dim
            a[0] = states[-1].action.values[-1][0] + 0.3 * dx + self.rng.uniform(-0.1, 0.1)
            a[1] = states[-1].action.values[-1][1] + 0.3 * dy + self.rng.uniform(-0.1, 0.1)
            for k in range(2, self.action_dim):
                a[k] = self.rng.uniform(-1.0, 1.0)
            states.append(self.step(a, t))
        return WorldTrajectory(states=states)

    def generate_dataset(self, episodes: int = 64, horizon: int = 16) -> List[WorldTrajectory]:
        """自主式数据引擎：批量生成仿真轨迹。"""
        out = []
        for ep in range(episodes):
            self.rng = __import__("random").Random(stable_seed("sim", ep, self.object.__dict__))
            out.append(self.roll_trajectory(horizon))
        return out


__all__ = ["SimObject", "TactileSimEngine"]
