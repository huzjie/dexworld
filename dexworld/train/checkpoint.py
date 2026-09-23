# -*- coding: utf-8 -*-
"""检查点读写（JSON）。"""

from __future__ import annotations

import os
from typing import Any, Dict

from ..errors import CheckpointError
from ..utils.io import ensure_dir, save_json, load_json


def save_checkpoint(obj: Any, path: str) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    save_json(obj, path)
    return path


def load_checkpoint(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise CheckpointError(f"检查点不存在: {path}")
    return load_json(path)


def state_dict(denoiser, skill: float) -> Dict[str, Any]:
    return {"denoiser_w": denoiser._w, "skill": skill}


def load_state(denoiser, state: Dict[str, Any], model=None) -> None:
    if "denoiser_w" in state:
        denoiser._w = state["denoiser_w"]
    if model is not None and "skill" in state:
        model._skill = state["skill"]


__all__ = ["save_checkpoint", "load_checkpoint", "state_dict", "load_state"]
