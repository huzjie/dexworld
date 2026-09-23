# -*- coding: utf-8 -*-
import random
import unittest
from dexworld.data.sensors import (
    GelSightSensor, TactileArraySensor, ForceTorqueSensor, SensorRegistry,
)


class TestSensors(unittest.TestCase):
    def test_gelsight_shape(self):
        s = GelSightSensor()
        rng = random.Random(0)
        out = s.sense({"contact": 0.8, "shear": 0.2, "texture": 0.5}, rng)
        self.assertEqual(len(out), s.dim)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in out))

    def test_registry(self):
        reg = SensorRegistry()
        self.assertIsInstance(reg.create("gelsight"), GelSightSensor)
        self.assertIsInstance(reg.create("force_torque"), ForceTorqueSensor)


if __name__ == "__main__":
    unittest.main()
