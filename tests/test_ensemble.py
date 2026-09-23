# -*- coding: utf-8 -*-
import unittest
from dexworld.data.sim_engine import TactileSimEngine
from dexworld.model.world_model import DexWorldModel
from dexworld.inference.ensemble import EnsemblePredictor


class TestEnsemble(unittest.TestCase):
    def test_predict(self):
        eng = TactileSimEngine(seed=0, vision_size=8)
        hist = eng.roll_trajectory(3).states
        model = DexWorldModel(vision_dim=64, tactile_dim=272, action_dim=7)
        pred = EnsemblePredictor(model, n_seeds=3).predict(hist)
        self.assertGreaterEqual(pred.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
