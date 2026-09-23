# -*- coding: utf-8 -*-
"""扩散采样器：从模型采样未来 token。"""

from __future__ import annotations

from typing import Dict, List, Optional


class DiffusionSampler:
    def __init__(self, model):
        self.model = model

    def sample_future(self, history, steps: int = 20,
                      seed: int = 0) -> Dict[str, List[float]]:
        """采样未来三模态 token（历史编码 + 融合 + 去噪）。"""
        tokens = self.model._encode_history(history)
        fused = self.model.bridge.fuse(tokens["vision"], tokens["tactile"],
                                       tokens["action"])
        cond = self.model.experts.forward(fused)
        return self.model.denoiser.sample(tokens, condition=cond, steps=steps, seed=seed)


__all__ = ["DiffusionSampler"]
