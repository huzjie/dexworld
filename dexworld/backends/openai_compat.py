# -*- coding: utf-8 -*-
"""OpenAI 兼容后端：可接任意 OpenAI 兼容 API（含 MiMo/GLM 等）作为语义编码器。"""

from __future__ import annotations

import json
import urllib.request

from ..errors import BackendError
from ..utils.logging import get_logger


class OpenAICompatBackend:
    """通过 OpenAI 兼容 /chat/completions 接口做高层语义编码（可选）。"""

    def __init__(self, api_base: str, api_key: str, model_name: str):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.log = get_logger("dexworld.openai")

    def embed(self, text: str) -> list:
        """调用 chat 接口，返回一个简短语义向量（用 token 数近似，简化）。"""
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 16,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            # 简化：把返回文本哈希成固定长度向量
            import hashlib
            h = hashlib.md5(content.encode("utf-8")).digest()
            return [b / 255.0 for b in h]
        except Exception as e:
            raise BackendError(f"OpenAI 兼容调用失败: {e}")


__all__ = ["OpenAICompatBackend"]
