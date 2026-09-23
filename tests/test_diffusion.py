# -*- coding: utf-8 -*-
import unittest
from dexworld.model.diffusion import JointDenoiser, NoiseSchedule


class TestDiffusion(unittest.TestCase):
    def test_schedule(self):
        sched = NoiseSchedule(steps=20)
        self.assertEqual(len(sched.betas()), 20)
        self.assertLess(sched.betas()[0], sched.betas()[-1])

    def test_denoiser(self):
        d = JointDenoiser(dim=64)
        tokens = {"vision": [0.1] * 64, "tactile": [0.2] * 64, "action": [0.3] * 64}
        out = d.sample(tokens, steps=10, seed=0)
        self.assertEqual(len(out["vision"]), 64)


if __name__ == "__main__":
    unittest.main()
