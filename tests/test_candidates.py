# -*- coding: utf-8 -*-
import unittest
from dexworld.planner.candidates import (
    random_candidates, grid_candidates, spiral_candidates, CandidateGenerator,
)


class TestCandidates(unittest.TestCase):
    def test_random(self):
        cands = random_candidates(8, action_dim=7)
        self.assertEqual(len(cands), 8)
        self.assertEqual(cands[0].dim, 7)

    def test_grid(self):
        cands = grid_candidates(3, action_dim=2)
        self.assertEqual(len(cands), 9)

    def test_spiral(self):
        cands = spiral_candidates(8, action_dim=2)
        self.assertEqual(len(cands), 8)

    def test_generator(self):
        gen = CandidateGenerator(method="random", n=4, action_dim=7)
        self.assertEqual(len(gen.generate()), 4)


if __name__ == "__main__":
    unittest.main()
