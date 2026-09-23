# -*- coding: utf-8 -*-
"""后端子包：Mock（可训练确定性）+ OpenAI 兼容。"""

from .mock import MockBackend
from .openai_compat import OpenAICompatBackend
from .registry import get_backend

__all__ = ["MockBackend", "OpenAICompatBackend", "get_backend"]
