# -*- coding: utf-8 -*-
import unittest
from dexworld.encoders import VisionEncoder, TactileEncoder, ActionEncoder, HBridge
from dexworld.types import VisionFrame, TactileObservation, ActionChunk


class TestEncoders(unittest.TestCase):
    def test_vision_encoder(self):
        enc = VisionEncoder(in_dim=256, hidden_dim=64)
        f = VisionFrame(pixels=[0.5] * 256, height=16, width=16)
        out = enc.encode(f)
        self.assertEqual(len(out), 64)

    def test_tactile_encoder(self):
        enc = TactileEncoder(in_dim=272, hidden_dim=64)
        obs = TactileObservation.zeros(dim=8)
        out = enc.encode(obs)
        self.assertEqual(len(out), 64)

    def test_action_encoder(self):
        enc = ActionEncoder(action_dim=7, hidden_dim=64)
        a = ActionChunk(values=[[0.1] * 7 for _ in range(4)], dim=7)
        out = enc.encode(a)
        self.assertEqual(len(out), 64)

    def test_hbridge(self):
        bridge = HBridge(64, 8)
        out = bridge.fuse([0.1] * 64, [0.2] * 64, [0.3] * 64)
        self.assertIn("joint", out)
        self.assertEqual(len(out["joint"]), 64)


if __name__ == "__main__":
    unittest.main()
