# -*- coding: utf-8 -*-
"""联合 tokenizer：把世界状态编码为联合 token 序列。"""

from __future__ import annotations

from typing import Dict, List

from ..types import VisionFrame, TactileObservation, ActionChunk
from .vision_encoder import VisionEncoder
from .tactile_encoder import TactileEncoder
from .action_encoder import ActionEncoder


class JointTokenizer:
    """世界状态 -> 三模态 token（供扩散模型使用）。"""

    def __init__(self, vision_dim: int, tactile_dim: int, action_dim: int,
                 hidden_dim: int = 64):
        self.hidden_dim = hidden_dim
        self.vision = VisionEncoder(vision_dim, hidden_dim)
        self.tactile = TactileEncoder(tactile_dim, hidden_dim)
        self.action = ActionEncoder(action_dim, hidden_dim)

    def tokenize(self, vision: VisionFrame, tactile: TactileObservation,
                 action: ActionChunk) -> Dict[str, List[float]]:
        return {
            "vision": self.vision.encode(vision),
            "tactile": self.tactile.encode(tactile),
            "action": self.action.encode(action),
        }


__all__ = ["JointTokenizer"]
