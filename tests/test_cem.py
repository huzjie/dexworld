# -*- coding: utf-8 -*-
import unittest
from dexworld.planner.cem import CEMPlanner


class TestCEM(unittest.TestCase):
    def test_plan(self):
        def score(h, chunk):
            return sum(chunk.values[0])
        planner = CEMPlanner(score, action_dim=2, horizon=2, iterations=2, population=8)
        best, s = planner.plan([])
        self.assertEqual(best.dim, 2)
        self.assertEqual(len(best.values), 2)


if __name__ == "__main__":
    unittest.main()
