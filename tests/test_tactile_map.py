# -*- coding: utf-8 -*-
import unittest
from dexworld.constants import TOTAL_REGIONS, REGIONS_PER_HAND
from dexworld.data.tactile_map import TactileMap, region_index


class TestTactileMap(unittest.TestCase):
    def setUp(self):
        self.tm = TactileMap()

    def test_total_regions(self):
        self.assertEqual(len(self.tm.regions), TOTAL_REGIONS)
        self.assertEqual(TOTAL_REGIONS, REGIONS_PER_HAND * 2)

    def test_left_right_split(self):
        self.assertEqual(len(self.tm.left_indices()), REGIONS_PER_HAND)
        self.assertEqual(len(self.tm.right_indices()), REGIONS_PER_HAND)

    def test_region_index(self):
        self.assertEqual(region_index("left_thumb_distal"), 0)
        self.assertEqual(self.tm.index("right_palm_proximal"), TOTAL_REGIONS - 1)


if __name__ == "__main__":
    unittest.main()
