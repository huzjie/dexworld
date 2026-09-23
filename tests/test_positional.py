# -*- coding: utf-8 -*-
import unittest
from dexworld.encoders.positional import sinusoidal_positional_encoding


class TestPositional(unittest.TestCase):
    def test_shape(self):
        enc = sinusoidal_positional_encoding(4, 8)
        self.assertEqual(len(enc), 4)
        self.assertEqual(len(enc[0]), 8)


if __name__ == "__main__":
    unittest.main()
