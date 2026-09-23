# -*- coding: utf-8 -*-
import unittest
from dexworld.train.optimizer import SGD
from dexworld.train.scheduler import CosineScheduler, StepScheduler


class TestScheduler(unittest.TestCase):
    def test_cosine(self):
        opt = SGD(lr=0.1)
        sched = CosineScheduler(opt, base_lr=0.1, total_steps=10)
        lr0 = sched.step()
        self.assertLessEqual(lr0, 0.1)

    def test_step(self):
        opt = SGD(lr=0.1)
        sched = StepScheduler(opt, base_lr=0.1, step_size=2, gamma=0.5)
        for _ in range(3):
            sched.step()
        self.assertLess(opt.lr, 0.1)


if __name__ == "__main__":
    unittest.main()
