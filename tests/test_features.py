"""Unit tests: derived measurements against docs.md rules."""

import unittest

from neev.adapters.base import BoundingBox
from neev.measurements.features import (
    cluster_rows,
    count_storeys,
    ground_openness,
    overhang_ratio,
    pounding_risk,
)


class TestStoreys(unittest.TestCase):
    def test_three_window_rows(self):
        boxes = [
            BoundingBox(0, y, 10, y + 10)
            for y in (0, 5, 100, 105, 200, 205)
        ]
        result = count_storeys(boxes, image_height_px=300)
        self.assertEqual(result.rows, 3)
        self.assertEqual(result.windows_used, 6)

    def test_no_windows_is_zero_rows(self):
        result = count_storeys([], image_height_px=300)
        self.assertEqual(result.rows, 0)

    def test_cluster_rows_boundary(self):
        self.assertEqual(cluster_rows([0.0, 1.0, 5.0], 1.0), [[0.0, 1.0], [5.0]])


class TestOpenness(unittest.TestCase):
    def test_ratio_against_upper_floors(self):
        result = ground_openness(80, 100, 20, 100)
        self.assertAlmostEqual(result.ground_openness, 0.8)
        self.assertAlmostEqual(result.ratio_vs_upper, 4.0)

    def test_zero_bands_are_safe(self):
        result = ground_openness(0, 0, 0, 0)
        self.assertEqual(result.ground_openness, 0.0)
        self.assertIsNone(result.ratio_vs_upper)


class TestOverhang(unittest.TestCase):
    def test_flags_above_1_1(self):
        self.assertTrue(overhang_ratio(115, 100).flag)
        self.assertFalse(overhang_ratio(105, 100).flag)


class TestPounding(unittest.TestCase):
    def test_flags_touching_misaligned_slabs(self):
        self.assertTrue(pounding_risk(5, 1000, slabs_aligned=False).flag)

    def test_no_flag_when_aligned(self):
        self.assertFalse(pounding_risk(5, 1000, slabs_aligned=True).flag)

    def test_no_flag_when_gap_is_wide(self):
        self.assertFalse(pounding_risk(50, 1000, slabs_aligned=False).flag)


if __name__ == "__main__":
    unittest.main()
