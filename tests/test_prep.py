"""Unit tests: frame sampling plans and IoU tracking. No video needed."""

import unittest

from neev.adapters.base import BoundingBox
from neev.prep.frames import ScoredFrame, frame_name, sample_plan, select_frames
from neev.prep.tracking import Tracker, iou


class TestFrames(unittest.TestCase):
    def test_sample_plan_two_fps(self):
        self.assertEqual(sample_plan(2.0, 2.0), [0.0, 0.5, 1.0, 1.5])

    def test_sample_plan_rejects_nonsense(self):
        self.assertEqual(sample_plan(0, 2.0), [])
        self.assertEqual(sample_plan(2.0, 0), [])

    def test_sample_plan_caps_evenly(self):
        self.assertEqual(sample_plan(10.0, 1.0, max_frames=2), [0.0, 5.0])

    def test_frame_name_matches_docs_style(self):
        self.assertEqual(frame_name(132), "f_0132.jpg")

    def test_select_frames_filters_and_caps(self):
        cands = [ScoredFrame(frame_name(i), float(i), s) for i, s in enumerate([5, 50, 60, 1, 70])]
        picked = select_frames(cands, min_sharpness=10, max_frames=2)
        self.assertEqual([f.frame_id for f in picked], ["f_0001.jpg", "f_0002.jpg"])

    def test_all_blurry_is_empty_not_safe(self):
        cands = [ScoredFrame("f_0000.jpg", 0.0, 1.0)]
        self.assertEqual(select_frames(cands, min_sharpness=10, max_frames=5), [])


class TestIou(unittest.TestCase):
    def test_identical_is_one(self):
        box = BoundingBox(0, 0, 10, 10)
        self.assertAlmostEqual(iou(box, box), 1.0)

    def test_disjoint_is_zero(self):
        self.assertEqual(
            iou(BoundingBox(0, 0, 5, 5), BoundingBox(9, 9, 10, 10)), 0.0
        )

    def test_half_overlap(self):
        self.assertAlmostEqual(
            iou(BoundingBox(0, 0, 10, 10), BoundingBox(5, 0, 15, 10)), 1 / 3
        )


class TestTracker(unittest.TestCase):
    def test_stable_id_for_same_building(self):
        tracker = Tracker(iou_threshold=0.3)
        first = tracker.update([BoundingBox(0, 0, 100, 100)])
        second = tracker.update([BoundingBox(5, 5, 105, 105)])
        self.assertEqual(first[0], "B1")
        self.assertEqual(second[0], "B1")

    def test_far_box_gets_new_id(self):
        tracker = Tracker(iou_threshold=0.3)
        tracker.update([BoundingBox(0, 0, 100, 100)])
        second = tracker.update([BoundingBox(500, 500, 600, 600)])
        self.assertEqual(second[0], "B2")

    def test_track_survives_brief_gap_then_expires(self):
        tracker = Tracker(iou_threshold=0.3, max_missing=1)
        tracker.update([BoundingBox(0, 0, 100, 100)])
        tracker.update([])
        self.assertIn("B1", tracker.tracks)
        tracker.update([])
        self.assertNotIn("B1", tracker.tracks)


if __name__ == "__main__":
    unittest.main()
