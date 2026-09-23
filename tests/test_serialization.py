# -*- coding: utf-8 -*-
import unittest
from dexworld.data.sim_engine import TactileSimEngine
from dexworld.model.world_model import DexWorldModel
from dexworld.utils.serialization import serialize_model, deserialize_model


class TestSerialization(unittest.TestCase):
    def test_roundtrip(self):
        eng = TactileSimEngine(seed=0, vision_size=8)
        trajs = eng.generate_dataset(episodes=2, horizon=4)
        model = DexWorldModel(vision_dim=64, tactile_dim=272, action_dim=7)
        model.train_step(trajs[0].states[:2], trajs[0].states[2:3])
        state = serialize_model(model)
        model2 = DexWorldModel(vision_dim=64, tactile_dim=272, action_dim=7)
        deserialize_model(model2, state)
        self.assertEqual(model2.skill, model.skill)


if __name__ == "__main__":
    unittest.main()
