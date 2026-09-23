# -*- coding: utf-8 -*-
import random
import unittest
from dexworld.data.sim_engine import TactileSimEngine
from dexworld.model.world_model import DexWorldModel
from dexworld.inference.planner import MPCPlanner
from dexworld.types import ActionChunk


class TestPlanner(unittest.TestCase):
    def test_plan(self):
        eng = TactileSimEngine(seed=0, vision_size=8)
        hist = eng.roll_trajectory(3).states
        model = DexWorldModel(vision_dim=64, tactile_dim=272, action_dim=7)
        planner = MPCPlanner(model, top_k=3)
        rng = random.Random(0)
        cands = [ActionChunk(values=[[rng.uniform(-1, 1) for _ in range(7)]]) for _ in range(5)]
        best, score = planner.plan(hist, cands)
        self.assertEqual(best.dim, 7)


if __name__ == "__main__":
    unittest.main()
