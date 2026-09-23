# -*- coding: utf-8 -*-
"""命令行入口。

用法：
    python -m dexworld.cli train --config config.yaml
    python -m dexworld.cli eval
    python -m dexworld.cli predict
    python -m dexworld.cli serve --port 8765
    python -m dexworld.cli doctor
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .pipeline import DexWorldPipeline
from .utils.logging import setup_logging


def cmd_train(args):
    cfg = load_config(args.config)
    result = DexWorldPipeline(cfg).run()
    print(result.summary())
    print(result.report)


def cmd_eval(args):
    cfg = load_config(args.config)
    # 轻量：直接跑一轮 pipeline 取评测
    result = DexWorldPipeline(cfg).run()
    print(result.report)


def cmd_predict(args):
    cfg = load_config(args.config)
    from .backends import get_backend
    backend = get_backend(cfg.model.backend,
                          vision_dim=cfg.vision.height * cfg.vision.width,
                          tactile_dim=cfg.tactile.regions_per_hand * 2 * cfg.tactile.dim,
                          action_dim=cfg.action.dim, hidden_dim=cfg.model.hidden_dim,
                          denoise_steps=cfg.model.denoise_steps)
    model = backend.get_model()
    from .data.sim_engine import TactileSimEngine
    engine = TactileSimEngine(seed=cfg.train.seed, action_dim=cfg.action.dim,
                              tactile_dim=cfg.tactile.dim)
    hist = engine.roll_trajectory(horizon=4).states
    pred = model.predict(hist)
    print(f"预测置信度: {pred.confidence:.4f}  误差: {pred.error:.4f}")


def cmd_serve(args):
    cfg = load_config(args.config)
    from .backends import get_backend
    backend = get_backend(cfg.model.backend,
                          vision_dim=cfg.vision.height * cfg.vision.width,
                          tactile_dim=cfg.tactile.regions_per_hand * 2 * cfg.tactile.dim,
                          action_dim=cfg.action.dim, hidden_dim=cfg.model.hidden_dim,
                          denoise_steps=cfg.model.denoise_steps)
    model = backend.get_model()
    from .serving import DexWorldServer
    server = DexWorldServer(model, port=args.port)
    server.serve_forever()


def cmd_doctor(args):
    """健康检查：验证环境与核心链路可用。"""
    print("[doctor] 检查核心模块导入 ...")
    import dexworld
    from dexworld.data.sim_engine import TactileSimEngine
    from dexworld.model.world_model import DexWorldModel
    print(f"[doctor] dexworld {dexworld.__version__} 导入 OK")
    print("[doctor] 检查仿真数据引擎 ...")
    engine = TactileSimEngine(seed=42)
    traj = engine.roll_trajectory(horizon=4)
    assert len(traj) == 4, "轨迹长度异常"
    print("[doctor] 仿真数据引擎 OK（4 步轨迹）")
    print("[doctor] 检查世界模型预测 ...")
    model = DexWorldModel(vision_dim=256, tactile_dim=272, action_dim=7)
    pred = model.predict(traj.states)
    assert pred.predicted_tactile is not None, "触觉预测为空"
    print(f"[doctor] 世界模型预测 OK（confidence={pred.confidence:.3f}）")
    print("[doctor] 全部通过 ✔")


def main(argv=None):
    argv = argv or sys.argv[1:]
    p = argparse.ArgumentParser(prog="dexworld", description="触觉世界模型框架")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("train")
    sub.add_parser("eval")
    sub.add_parser("predict")
    sp = sub.add_parser("serve")
    sp.add_argument("--port", type=int, default=8765)
    sub.add_parser("doctor")

    args = p.parse_args(argv)
    setup_logging()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    elif args.cmd == "serve":
        cmd_serve(args)
    elif args.cmd == "doctor":
        cmd_doctor(args)
    else:
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
