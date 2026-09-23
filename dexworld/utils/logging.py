# -*- coding: utf-8 -*-
"""日志工具。"""

import logging
import sys

_FMT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=_FMT, stream=sys.stdout)


def get_logger(name: str = "dexworld") -> logging.Logger:
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
