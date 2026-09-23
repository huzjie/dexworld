# -*- coding: utf-8 -*-
"""计时工具。"""

import time
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self._t0 = None
        self.elapsed = 0.0

    def start(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._t0 is None:
            return 0.0
        self.elapsed = time.perf_counter() - self._t0
        self._t0 = None
        return self.elapsed

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


@contextmanager
def timeit(label: str = "elapsed"):
    t = Timer().start()
    try:
        yield t
    finally:
        t.stop()


__all__ = ["Timer", "timeit"]
