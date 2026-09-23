# -*- coding: utf-8 -*-
import unittest
from dexworld.data.sim_engine import TactileSimEngine
from dexworld.data.augment import add_tactile_noise, temporal_flip, region_dropout


class TestAugment(unittest.TestCase):
    def setUp(self):
        self.eng = TactileSimEngine(seed=2)
        self.traj = self.eng.roll_trajectory(horizon=4)

    def test_noise_shape(self):
        obs = self.traj.states[0].tactile
        noisy = add_tactile_noise(obs, sigma=0.1)
        self.assertEqual(len(noisy.left), len(obs.left))

    def test_flip(self):
        flipped = temporal_flip(self.traj)
        self.assertEqual(len(flipped), len(self.traj))

    def test_dropout(self):
        obs = self.traj.states[0].tactile
        dropped = region_dropout(obs, p=1.0)
        self.assertEqual(sum(dropped.left[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
