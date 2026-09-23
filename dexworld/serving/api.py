# -*- coding: utf-8 -*-
"""零依赖 HTTP 服务（基于 http.server）。

提供 /health、/predict、/metrics 三个端点，可独立部署为推理服务。
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..utils.logging import get_logger
from .observability import Observability


class DexWorldServer:
    def __init__(self, model, host: str = "127.0.0.1", port: int = 8765):
        self.model = model
        self.host = host
        self.port = port
        self.obs = Observability()
        self.log = get_logger("dexworld.serve")
        self._httpd = None

    def _handler(self):
        model = self.model
        obs = self.obs

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self._reply(200, {"status": "ok", "skill": model.skill})
                elif self.path == "/metrics":
                    self._reply(200, obs.snapshot())
                else:
                    self._reply(404, {"error": "not found"})

            def do_POST(self):
                if self.path == "/predict":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length) or b"{}")
                    try:
                        # 简化：接受一个历史窗口描述（可扩展）
                        pred = model.predict([])
                        self._reply(200, {
                            "error": pred.error,
                            "confidence": pred.confidence,
                            "skill": model.skill,
                        })
                    except Exception as e:
                        self._reply(500, {"error": str(e)})
                else:
                    self._reply(404, {"error": "not found"})

            def _reply(self, code, obj):
                data = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        return Handler

    def start(self):
        self._httpd = HTTPServer((self.host, self.port), self._handler())
        self.log.info(f"服务启动: http://{self.host}:{self.port}")
        return self

    def serve_forever(self):
        if self._httpd is None:
            self.start()
        self._httpd.serve_forever()


__all__ = ["DexWorldServer"]
