# -*- coding: utf-8 -*-
"""H-Bridge 融合模型：封装 H-Bridge + 专家混合，对外提供单一前向接口。"""

from __future__ import annotations

from typing import Dict, List

from ..encoders import HBridge
from .experts import ExpertMixture


class HBridgeModel:
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.bridge = HBridge(dim)
        self.experts = ExpertMixture(dim)

    def forward(self, tokens: Dict[str, List[float]]) -> List[float]:
        fused = self.bridge.fuse(tokens.get("vision", [0.0] * self.dim),
                                 tokens.get("tactile", [0.0] * self.dim),
                                 tokens.get("action", [0.0] * self.dim))
        return self.experts.forward(fused)


__all__ = ["HBridgeModel"]
