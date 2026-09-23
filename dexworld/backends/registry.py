# -*- coding: utf-8 -*-
"""后端注册与工厂。"""

from __future__ import annotations

from ..errors import BackendError
from ..utils.registry import Registry

_backends = Registry("backend")


def _build_mock(**kw):
    from .mock import MockBackend
    return MockBackend(**kw)


def _build_openai(**kw):
    from .openai_compat import OpenAICompatBackend
    return OpenAICompatBackend(api_base=kw.get("api_base", ""),
                               api_key=kw.get("api_key", ""),
                               model_name=kw.get("model_name", ""))


_backends.register("mock", _build_mock)
_backends.register("openai_compat", _build_openai)


def get_backend(name: str, **kwargs):
    try:
        return _backends.get(name)(**kwargs)
    except KeyError:
        raise BackendError(f"未知后端: {name}，可用: {_backends.keys()}")


__all__ = ["get_backend"]
