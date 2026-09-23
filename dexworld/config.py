# -*- coding: utf-8 -*-
"""配置加载：支持 YAML（可选 pyyaml，缺失则回退极简解析器）。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from .errors import ConfigError


def _deep_merge(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class TactileConfig:
    dim: int = 8
    regions_per_hand: int = 17


@dataclass
class VisionConfig:
    height: int = 64
    width: int = 64
    channels: int = 3


@dataclass
class ActionConfig:
    dim: int = 7
    horizon: int = 16


@dataclass
class ModelConfig:
    backend: str = "mock"
    hidden_dim: int = 64
    num_layers: int = 4
    hbridge_heads: int = 8
    denoise_steps: int = 20
    beta_start: float = 1e-4
    beta_end: float = 0.02
    # openai_compat 后端参数
    api_base: str = ""
    api_key: str = ""
    model_name: str = ""


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 8
    lr: float = 3e-4
    seed: int = 42
    log_interval: int = 1
    checkpoint_dir: str = "checkpoints"


@dataclass
class DataConfig:
    trajectories: int = 32
    seq_len: int = 8
    sim_episodes: int = 64


@dataclass
class EvalConfig:
    rollout_horizon: int = 16
    top_k_plans: int = 8


@dataclass
class DexWorldConfig:
    tactile: TactileConfig = field(default_factory=TactileConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "DexWorldConfig":
        d = d or {}
        cfg = cls()
        if "tactile" in d:
            cfg.tactile = TactileConfig(**{k: v for k, v in d["tactile"].items()
                                           if k in TactileConfig.__dataclass_fields__})
        if "vision" in d:
            cfg.vision = VisionConfig(**{k: v for k, v in d["vision"].items()
                                         if k in VisionConfig.__dataclass_fields__})
        if "action" in d:
            cfg.action = ActionConfig(**{k: v for k, v in d["action"].items()
                                         if k in ActionConfig.__dataclass_fields__})
        if "model" in d:
            cfg.model = ModelConfig(**{k: v for k, v in d["model"].items()
                                       if k in ModelConfig.__dataclass_fields__})
        if "train" in d:
            cfg.train = TrainConfig(**{k: v for k, v in d["train"].items()
                                       if k in TrainConfig.__dataclass_fields__})
        if "data" in d:
            cfg.data = DataConfig(**{k: v for k, v in d["data"].items()
                                     if k in DataConfig.__dataclass_fields__})
        if "eval" in d:
            cfg.eval = EvalConfig(**{k: v for k, v in d["eval"].items()
                                     if k in EvalConfig.__dataclass_fields__})
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def merged(self, overrides: Optional[Dict[str, Any]]) -> "DexWorldConfig":
        if not overrides:
            return self
        return DexWorldConfig.from_dict(_deep_merge(self.to_dict(), overrides))


def load_yaml(path: str) -> Dict[str, Any]:
    """读取 YAML：优先 pyyaml，缺失回退内置极简解析器。"""
    import os
    if not os.path.exists(path):
        raise ConfigError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except ImportError:
        from .utils.yamlish import parse_yaml
        return parse_yaml(text)
    except Exception as e:  # pyyaml 解析失败也回退
        from .utils.yamlish import parse_yaml
        try:
            return parse_yaml(text)
        except Exception:
            raise ConfigError(f"YAML 解析失败: {e}")


def load_config(path: Optional[str] = None) -> DexWorldConfig:
    """从文件或默认值加载配置。"""
    if not path:
        return DexWorldConfig()
    return DexWorldConfig.from_dict(load_yaml(path))


__all__ = [
    "TactileConfig", "VisionConfig", "ActionConfig", "ModelConfig",
    "TrainConfig", "DataConfig", "EvalConfig", "DexWorldConfig",
    "load_yaml", "load_config",
]
