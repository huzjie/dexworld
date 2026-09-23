# -*- coding: utf-8 -*-
import unittest
from dexworld.utils.metrics_tracker import MetricsTracker


class TestMetricsTracker(unittest.TestCase):
    def test_record(self):
        mt = MetricsTracker()
        mt.record("loss", 1.0)
        mt.record("loss", 0.5)
        self.assertEqual(mt.latest("loss"), 0.5)
        self.assertEqual(len(mt.series("loss")), 2)
        self.assertIn("loss_min", mt.summary())


if __name__ == "__main__":
    unittest.main()
