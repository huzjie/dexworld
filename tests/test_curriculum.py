# -*- coding: utf-8 -*-
import unittest
from dexworld.data.sim_engine import TactileSimEngine
from dexworld.data.curriculum import Curriculum, difficulty_of


class TestCurriculum(unittest.TestCase):
    def test_stages(self):
        eng = TactileSimEngine(seed=0, vision_size=8)
        trajs = eng.generate_dataset(episodes=9, horizon=4)
        cur = Curriculum(trajs)
        stages = cur.stages(n_stages=3)
        self.assertEqual(len(stages), 3)
        total = sum(len(s) for s in stages)
        self.assertEqual(total, len(trajs))


if __name__ == "__main__":
    unittest.main()
