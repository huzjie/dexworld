# -*- coding: utf-8 -*-
"""训练器：编排训练循环。"""

from __future__ import annotations

from typing import List, Optional

from ..types import TrainMetrics, WorldState, WorldTrajectory
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from .callbacks import Callback, MetricsLogger


class Trainer:
    """触觉世界模型训练器。"""

    def __init__(self, model, epochs: int = 5, batch_size: int = 8,
                 lr: float = 3e-4, seed: int = 42, callbacks: Optional[List[Callback]] = None):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.callbacks = callbacks or [MetricsLogger()]
        self.log = get_logger("dexworld.train")
        self.history: List[TrainMetrics] = []

    def fit(self, trajectories: List[WorldTrajectory]) -> List[TrainMetrics]:
        set_seed(self.seed)
        import random
        rng = random.Random(self.seed)
        for epoch in range(1, self.epochs + 1):
            losses = []
            pred_errs = []
            for traj in trajectories:
                n = len(traj)
                if n < 2:
                    continue
                # 随机切历史 + 目标
                split = rng.randint(1, n - 1)
                hist = traj.states[:split]
                tgt = traj.states[split:split + 1]
                loss = self.model.train_step(hist, tgt)
                losses.append(loss)
                # 预测误差
                try:
                    pred = self.model.predict(hist)
                    pred_errs.append(pred.error)
                except Exception:
                    pred_errs.append(0.0)
            mean_loss = sum(losses) / len(losses) if losses else 0.0
            mean_err = sum(pred_errs) / len(pred_errs) if pred_errs else 0.0
            m = TrainMetrics(epoch=epoch, loss=mean_loss, prediction_error=mean_err,
                             extra={"skill": self.model.skill})
            self.history.append(m)
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, m.as_dict())
        return self.history


__all__ = ["Trainer"]
