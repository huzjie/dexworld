# -*- coding: utf-8 -*-
"""Mock 后端：零依赖、可训练、确定性。

内置一个可训练的 DexWorldModel，用于在无真实模型/无网络环境下完整跑通
「仿真数据 -> 训练 -> 预测 -> 规划」闭环，并演示「预测误差随训练下降」。
"""

from __future__ import annotations

from ..model.world_model import DexWorldModel
from ..utils.logging import get_logger


class MockBackend:
    def __init__(self, **kwargs):
        self.model = DexWorldModel(
            vision_dim=kwargs.get("vision_dim", 1024),
            tactile_dim=kwargs.get("tactile_dim", 272),
            action_dim=kwargs.get("action_dim", 7),
            hidden_dim=kwargs.get("hidden_dim", 64),
            denoise_steps=kwargs.get("denoise_steps", 20),
        )
        self.log = get_logger("dexworld.mock")

    def get_model(self):
        return self.model


__all__ = ["MockBackend"]
