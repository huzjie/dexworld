# -*- coding: utf-8 -*-
import unittest
from dexworld.data.replay import ReplayBuffer


class TestReplay(unittest.TestCase):
    def test_capacity(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(i)
        self.assertEqual(len(buf), 5)

    def test_sample(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(10):
            buf.push(i)
        s = buf.sample(3, seed=0)
        self.assertEqual(len(s), 3)


if __name__ == "__main__":
    unittest.main()
