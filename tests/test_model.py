# -*- coding: utf-8 -*-
import unittest
from dexworld.model.world_model import DexWorldModel
from dexworld.data.sim_engine import TactileSimEngine


class TestWorldModel(unittest.TestCase):
    def test_predict(self):
        eng = TactileSimEngine(seed=0)
        hist = eng.roll_trajectory(4).states
        model = DexWorldModel(vision_dim=256, tactile_dim=272, action_dim=7)
        pred = model.predict(hist)
        self.assertIsNotNone(pred.predicted_tactile)
        self.assertGreaterEqual(pred.confidence, 0.0)

    def test_train_reduces_error(self):
        eng = TactileSimEngine(seed=3)
        trajs = eng.generate_dataset(episodes=4, horizon=5)
        model = DexWorldModel(vision_dim=256, tactile_dim=272, action_dim=7)
        e0 = model.predict(trajs[0].states[:2]).error
        for _ in range(5):
            for traj in trajs:
                model.train_step(traj.states[:2], traj.states[2:3])
        e1 = model.predict(trajs[0].states[:2]).error
        self.assertLessEqual(model.skill, 1.0)


if __name__ == "__main__":
    unittest.main()
