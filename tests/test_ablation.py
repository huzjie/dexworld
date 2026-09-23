# -*- coding: utf-8 -*-
import unittest
from dexworld.data.sim_engine import TactileSimEngine
from dexworld.model.world_model import DexWorldModel
from dexworld.eval.ablation import run_ablation


class TestAblation(unittest.TestCase):
    def test_run(self):
        eng = TactileSimEngine(seed=1, vision_size=8)
        trajs = eng.generate_dataset(episodes=4, horizon=5)
        model = DexWorldModel(vision_dim=64, tactile_dim=272, action_dim=7)
        res = run_ablation(model, trajs)
        for k in ("full", "no_vision", "no_tactile"):
            self.assertIn(k, res)


if __name__ == "__main__":
    unittest.main()
