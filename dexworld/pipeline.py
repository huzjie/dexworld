# -*- coding: utf-8 -*-
"""端到端编排 Pipeline：数据 -> 训练 -> 评测 -> 报告。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .backends import get_backend
from .config import DexWorldConfig
from .data.sim_engine import TactileSimEngine
from .data.dataset import WorldDataset
from .eval.benchmark import DexWorldBenchmark
from .train.trainer import Trainer
from .utils.logging import get_logger


@dataclass
class PipelineResult:
    metrics: List[dict]
    benchmark: Dict[str, float]
    report: str

    def summary(self) -> str:
        lines = ["DexWorld 触觉世界模型训练完成",
                 f"训练轮次: {len(self.metrics)}",
                 f"最终预测误差: {self.metrics[-1].get('prediction_error', 0):.4f}"
                 if self.metrics else "无训练",
                 f"触觉重建误差: {self.benchmark.get('tactile_error', 0):.4f}",
                 f"模型能力 skill: {self.benchmark.get('skill', 0):.4f}"]
        return "\n".join(lines)


class DexWorldPipeline:
    """一站式触觉世界模型流水线。"""

    def __init__(self, config: Optional[DexWorldConfig] = None):
        self.config = config or DexWorldConfig()
        self.log = get_logger("dexworld.pipeline")

    def run(self) -> PipelineResult:
        cfg = self.config
        # 1. 后端 / 模型
        backend = get_backend(cfg.model.backend,
                              vision_dim=cfg.vision.height * cfg.vision.width,
                              tactile_dim=cfg.tactile.regions_per_hand * 2 * cfg.tactile.dim,
                              action_dim=cfg.action.dim,
                              hidden_dim=cfg.model.hidden_dim,
                              denoise_steps=cfg.model.denoise_steps,
                              api_base=cfg.model.api_base,
                              api_key=cfg.model.api_key,
                              model_name=cfg.model.model_name)
        model = backend.get_model()

        # 2. 仿真数据
        self.log.info("生成仿真触觉数据 ...")
        engine = TactileSimEngine(seed=cfg.train.seed, action_dim=cfg.action.dim,
                                  tactile_dim=cfg.tactile.dim)
        trajs = engine.generate_dataset(episodes=cfg.data.sim_episodes,
                                        horizon=cfg.data.seq_len + 1)
        dataset = WorldDataset(trajs)

        # 3. 训练
        self.log.info(f"开始训练 {cfg.train.epochs} 轮 ...")
        trainer = Trainer(model, epochs=cfg.train.epochs,
                          batch_size=cfg.train.batch_size, lr=cfg.train.lr,
                          seed=cfg.train.seed)
        history = trainer.fit(dataset.trajectories)

        # 4. 评测
        self.log.info("评测 ...")
        bench = DexWorldBenchmark(model).run(dataset.trajectories,
                                             horizon=cfg.eval.rollout_horizon)

        from .eval.report import build_report
        report = build_report(bench)
        metrics = [m.as_dict() for m in history]
        return PipelineResult(metrics=metrics, benchmark=bench, report=report)


__all__ = ["DexWorldPipeline", "PipelineResult"]
