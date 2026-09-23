# -*- coding: utf-8 -*-
import os
import tempfile
import unittest
from dexworld.config import DexWorldConfig, load_yaml


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = DexWorldConfig()
        self.assertEqual(cfg.model.backend, "mock")
        self.assertEqual(cfg.train.epochs, 5)

    def test_yamlish_fallback(self):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        tmp.write("model:\n  backend: mock\n  hidden_dim: 32\ntrain:\n  epochs: 3\n")
        tmp.close()
        try:
            d = load_yaml(tmp.name)
            self.assertEqual(d["model"]["backend"], "mock")
            self.assertEqual(d["model"]["hidden_dim"], 32)
        finally:
            os.unlink(tmp.name)


if __name__ == "__main__":
    unittest.main()
