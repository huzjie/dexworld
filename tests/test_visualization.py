# -*- coding: utf-8 -*-
import unittest
from dexworld.data.sim_engine import TactileSimEngine
from dexworld.visualization import render_tactile_heatmap, render_vision_ascii, render_timeline


class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.eng = TactileSimEngine(seed=1, vision_size=8)
        self.traj = self.eng.roll_trajectory(horizon=3)

    def test_heatmap(self):
        s = render_tactile_heatmap(self.traj.states[0].tactile, "left")
        self.assertIn("触觉热图", s)

    def test_vision(self):
        s = render_vision_ascii(self.traj.states[0].vision)
        self.assertIn("视觉帧", s)

    def test_timeline(self):
        s = render_timeline(self.traj)
        self.assertIn("时间线", s)


if __name__ == "__main__":
    unittest.main()
