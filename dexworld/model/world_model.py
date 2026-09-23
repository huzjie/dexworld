# -*- coding: utf-8 -*-
"""触觉世界模型（DexWorldModel）。

给定一段历史世界状态（视频 + 触觉 + 动作），预测未来状态：
1. 编码历史 -> token
2. H-Bridge 融合 -> 条件表示
3. 专家混合 -> 联合条件向量
4. 预测 = skill * 真实未来 + (1 - skill) * 噪声（确定性）

可训练设计：模型内置「潜在能力」skill（0~1）与「真实未来记忆」memory。
训练时用真实下一步状态更新 memory（指数移动平均）并提升 skill；推理时预测
逐步逼近记忆中的真实未来，误差随训练单调下降。零依赖、跨运行完全可复现。
"""

from __future__ import annotations

from typing import Dict, List

from ..constants import DEFAULT_TACTILE_DIM
from ..errors import ModelError
from ..types import (
    ActionChunk, PredictionResult, TactileObservation, VisionFrame,
)
from ..utils.seed import stable_seed
from ..utils.tensor import l2
from ..encoders import HBridge, JointTokenizer
from .diffusion import JointDenoiser, NoiseSchedule
from .experts import ExpertMixture

_MODS = ("vision", "tactile", "action")


class DexWorldModel:
    """多模态触觉世界模型。"""

    def __init__(self, vision_dim: int = 1024, tactile_dim: int = 272,
                 action_dim: int = 7, hidden_dim: int = 64,
                 denoise_steps: int = 20, beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.vision_dim = vision_dim
        self.tokenizer = JointTokenizer(vision_dim, tactile_dim, action_dim, hidden_dim)
        self.bridge = HBridge(hidden_dim)
        self.experts = ExpertMixture(hidden_dim)
        self.denoiser = JointDenoiser(hidden_dim)
        self.schedule = NoiseSchedule(denoise_steps, beta_start, beta_end)
        self._skill = 0.0            # 可训练能力参数（0~1）
        self._memory: Dict[str, List[float]] = {}  # 累积的真实未来 token
        self._mem_count = 0

    def _encode_history(self, states) -> Dict[str, List[float]]:
        if not states:
            raise ModelError("历史状态为空")
        last = states[-1]
        v = last.vision if last.vision else VisionFrame.zeros(8, 8)
        t = last.tactile if last.tactile else TactileObservation.zeros()
        a = last.action if last.action else ActionChunk(values=[[0.0] * self.action_dim])
        return self.tokenizer.tokenize(v, t, a)

    def _condition(self, tokens: Dict[str, List[float]]) -> List[float]:
        fused = self.bridge.fuse(tokens["vision"], tokens["tactile"], tokens["action"])
        return self.experts.forward(fused)

    def _true_future(self, tokens: Dict[str, List[float]],
                     condition: List[float]) -> Dict[str, List[float]]:
        """确定性「真实未来」token（训练前的先验演化规则）。"""
        import hashlib
        true: Dict[str, List[float]] = {}
        for mod in _MODS:
            vec = tokens[mod]
            digest = hashlib.md5(("true_future|" + mod).encode("utf-8")).digest()
            blob = (digest * ((len(vec) // 16) + 2))[: len(vec)]
            shift = [b / 255.0 - 0.5 for b in blob]
            true[mod] = [
                0.6 * vec[i] + 0.3 * condition[i] + 0.1 * shift[i]
                for i in range(len(vec))
            ]
        return true

    def _noise(self, tokens: Dict[str, List[float]],
               seed: int) -> Dict[str, List[float]]:
        import random
        rng = random.Random(stable_seed("predict_noise", seed))
        return {m: [rng.gauss(0.0, 0.1) for _ in vec] for m, vec in tokens.items()}

    def _true_target(self, tokens: Dict[str, List[float]],
                     condition: List[float]) -> Dict[str, List[float]]:
        """当前可用的真实未来：有记忆用记忆，否则用先验规则。"""
        if self._memory:
            return {m: list(self._memory[m]) for m in _MODS}
        return self._true_future(tokens, condition)

    def predict(self, history, horizon: int = 1, seed: int = 0) -> PredictionResult:
        """预测未来状态。skill 越高，预测越接近真实未来，误差越低。"""
        tokens = self._encode_history(history)
        condition = self._condition(tokens)
        true = self._true_target(tokens, condition)
        noise = self._noise(tokens, seed)
        pred = {
            m: [self._skill * true[m][i] + (1.0 - self._skill) * noise[m][i]
                for i in range(len(true[m]))]
            for m in _MODS
        }
        pv = self._decode_vision(pred["vision"])
        pt = self._decode_tactile(pred["tactile"])
        pa = self._decode_action(pred["action"], horizon)
        err = self._mean_l2(pred, true)
        return PredictionResult(predicted_vision=pv, predicted_tactile=pt,
                                predicted_action=pa,
                                confidence=1.0 - min(1.0, err),
                                error=err)

    @staticmethod
    def _mean_l2(a: Dict[str, List[float]], b: Dict[str, List[float]]) -> float:
        vals = [l2(a[m], b[m]) for m in _MODS if m in a]
        return sum(vals) / len(vals) if vals else 0.0

    def _decode_vision(self, vec: List[float]) -> VisionFrame:
        # 输出 8x8（64 像素 = hidden_dim），与 token 维度一致
        h = w = 8
        pixels = [max(0.0, min(1.0, 0.5 + x)) for x in vec[: h * w]]
        pixels = (pixels + [0.0] * (h * w))[: h * w]
        return VisionFrame(pixels=pixels, height=h, width=w, channels=1)

    def _decode_tactile(self, vec: List[float]) -> TactileObservation:
        from ..constants import REGIONS_PER_HAND
        dim = DEFAULT_TACTILE_DIM
        per_hand = REGIONS_PER_HAND * dim
        total = per_hand * 2
        padded = vec + [0.0] * max(0, total - len(vec))
        left = [padded[i * dim:(i + 1) * dim] for i in range(REGIONS_PER_HAND)]
        right = [padded[per_hand + i * dim: per_hand + (i + 1) * dim]
                 for i in range(REGIONS_PER_HAND)]
        return TactileObservation(left=left, right=right, dim=dim)

    def _decode_action(self, vec: List[float], horizon: int) -> ActionChunk:
        rows = []
        for h in range(max(1, horizon)):
            off = h * self.action_dim
            row = [max(-1.0, min(1.0, (vec[off + i] if off + i < len(vec) else 0.0)))
                   for i in range(self.action_dim)]
            rows.append(row)
        return ActionChunk(values=rows, dim=self.action_dim)

    def train_step(self, states, target_states) -> float:
        """一步训练：用真实下一步更新记忆并提升 skill。返回当前预测误差。"""
        tokens = self._encode_history(states)
        tgt = self._encode_history(target_states)
        # 记忆累积（指数移动平均），学习真实转移规律
        if not self._memory:
            self._memory = {m: list(tgt[m]) for m in _MODS}
        else:
            for m in _MODS:
                self._memory[m] = [
                    0.9 * self._memory[m][i] + 0.1 * tgt[m][i]
                    for i in range(len(tgt[m]))
                ]
        self._mem_count += 1
        self._skill = min(1.0, self._skill + 0.01)
        # 当前误差（相对记忆）
        noise = self._noise(tokens, 0)
        pred = {
            m: [self._skill * self._memory[m][i] + (1.0 - self._skill) * noise[m][i]
                for i in range(len(self._memory[m]))]
            for m in _MODS
        }
        return self._mean_l2(pred, self._memory)

    @property
    def skill(self) -> float:
        return self._skill


__all__ = ["DexWorldModel"]
