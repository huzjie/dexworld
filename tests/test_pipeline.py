# -*- coding: utf-8 -*-
import unittest
from dexworld.config import DexWorldConfig
from dexworld.pipeline import DexWorldPipeline


class TestPipeline(unittest.TestCase):
    def test_run(self):
        cfg = DexWorldConfig()
        cfg.model.backend = "mock"
        cfg.train.epochs = 2
        cfg.data.sim_episodes = 8
        cfg.data.seq_len = 5
        result = DexWorldPipeline(cfg).run()
        self.assertGreater(len(result.metrics), 0)
        self.assertIn("tactile_error", result.benchmark)
        self.assertIn("训练", result.summary())


if __name__ == "__main__":
    unittest.main()
