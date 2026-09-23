# -*- coding: utf-8 -*-
"""模型状态序列化（JSON）。"""

from __future__ import annotations

from typing import Any, Dict


def serialize_model(model) -> Dict[str, Any]:
    return {"skill": model.skill,
            "memory": model._memory,
            "mem_count": model._mem_count}


def deserialize_model(model, state: Dict[str, Any]) -> None:
    model._skill = state.get("skill", 0.0)
    model._memory = state.get("memory", {})
    model._mem_count = state.get("mem_count", 0)


__all__ = ["serialize_model", "deserialize_model"]
