# -*- coding: utf-8 -*-
import unittest
from dexworld.data.sim_engine import TactileSimEngine


class TestSimEngine(unittest.TestCase):
    def test_rollout(self):
        eng = TactileSimEngine(seed=42)
        traj = eng.roll_trajectory(horizon=6)
        self.assertEqual(len(traj), 6)
        self.assertIsNotNone(traj.states[0].vision)
        self.assertIsNotNone(traj.states[0].tactile)

    def test_determinism(self):
        a = TactileSimEngine(seed=1).roll_trajectory(4)
        b = TactileSimEngine(seed=1).roll_trajectory(4)
        self.assertEqual(a.states[3].vision.pixels, b.states[3].vision.pixels)


if __name__ == "__main__":
    unittest.main()
