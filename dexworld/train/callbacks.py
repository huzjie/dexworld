# -*- coding: utf-8 -*-
"""训练回调。"""

from __future__ import annotations

from typing import List

from ..utils.logging import get_logger


class Callback:
    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        pass


class EarlyStopping(Callback):
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.wait = 0
        self.stopped = False

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        loss = metrics.get("loss", float("inf"))
        if loss < self.best - self.min_delta:
            self.best = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped = True


class MetricsLogger(Callback):
    def __init__(self, log_interval: int = 1):
        self.log_interval = log_interval
        self.log = get_logger("dexworld.train")

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        if epoch % self.log_interval == 0:
            self.log.info(f"epoch {epoch}: loss={metrics.get('loss', 0):.4f} "
                          f"pred_err={metrics.get('prediction_error', 0):.4f}")


__all__ = ["Callback", "EarlyStopping", "MetricsLogger"]
