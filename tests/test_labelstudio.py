"""Tests: native Label Studio parsing, rotated IoU, LS evaluation.

Synthetic geometry carries the exact assertions (hand-computed corners
and IoUs); the real export is read (never written) for preservation
checks: 29 tasks, 657 results, 14 classes, 2 empties, 186 rotated.
"""

import importlib.util
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from neev.eval.compare import run_rotated_evaluation
from neev.eval.labelstudio import (
    load_export,
    ls_rotation_to_corners,
    parse_task,
    rect_polygon,
    resolve_image_path,
    validate_export,
)
from neev.eval.matching import (
    clip_polygon_to_image,
    match_rotated,
    rotated_iou,
)

REPO = Path(__file__).resolve().parent.parent
EXPORT = REPO / "eval" / "annotations_label_studio.json"


def task(task_id, upload, results, width=200, height=100):
    return {
        "id": task_id,
        "data": {"image": f"/data/upload/1/{upload}"},
        "file_upload": upload,
        "annotations": [{
            "result": [
                {"id": f"r{i}", "type": "rectanglelabels",
                 "from_name": "label", "to_name": "image",
                 "image_rotation": 0,
                 "original_width": width, "original_height": height,
                 "value": dict(v)}
                for i, v in enumerate(results)
            ],
        }],
    }


def rect(labels, x, y, w, h, rotation=0):
    return {"rectanglelabels": labels, "x": x, "y": y,
            "width": w, "height": h, "rotation": rotation}


def tree_with(*names):
    tmp = TemporaryDirectory()
    root = Path(tmp.name)
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-image-bytes-" + name.encode())
    return tmp, root


class TestParsing(unittest.TestCase):
    def test_parse_task_and_negatives(self):
        tmp, root = tree_with("real_world/a.png")
        try:
            image, errors = parse_task(
                task(2, "abcd1234-a.png",
                     [rect(["crack"], 10, 20, 30, 40, rotation=15)]), root)
            self.assertEqual(errors, [])
            self.assertEqual(image.task_id, 2)
            self.assertEqual(image.source, "real_world")
            self.assertFalse(image.is_negative)
            (parsed,) = image.rects
            self.assertEqual(parsed.label, "crack")
            self.assertEqual(parsed.rotation_deg, 15)
            self.assertEqual((parsed.x_pct, parsed.w_pct), (10, 30))

            empty, errors = parse_task(task(3, "abcd1234-a.png", []), root)
            self.assertEqual(errors, [])
            self.assertTrue(empty.is_negative)
            self.assertEqual(empty.rects, [])
        finally:
            tmp.cleanup()

    def test_multilabel_expands_without_loss(self):
        tmp, root = tree_with("real_world/a.png")
        try:
            image, errors = parse_task(
                task(2, "u-a.png", [rect(["crack", "spalling"], 0, 0, 10, 10)]),
                root)
            self.assertEqual(errors, [])
            self.assertEqual([r.label for r in image.rects], ["crack", "spalling"])
            self.assertEqual(image.rects[0].result_id, image.rects[1].result_id)
        finally:
            tmp.cleanup()

    def test_task_id_is_not_image_id(self):
        tmp, root = tree_with("real_world/zz.png")
        try:
            image, errors = parse_task(task(30, "u-zz.png", []), root)
            self.assertEqual(errors, [])
            self.assertEqual(image.task_id, 30)
            self.assertEqual(image.image_id, "zz")
        finally:
            tmp.cleanup()


class TestMapping(unittest.TestCase):
    def test_missing_file_held_out(self):
        tmp, root = tree_with("real_world/other.png")
        try:
            image, errors = parse_task(task(2, "u-nope.png", []), root)
            self.assertIsNone(image)
            self.assertTrue(errors)
        finally:
            tmp.cleanup()

    def test_ambiguous_match_held_out(self):
        tmp, root = tree_with("real_world/a/dup.png", "real_world/b/dup.png")
        try:
            image, errors = parse_task(task(2, "u-dup.png", []), root)
            self.assertIsNone(image)
            self.assertIn("DIFFERENT", errors[0])
        finally:
            tmp.cleanup()

    def test_identical_duplicates_resolve_and_report(self):
        tmp = TemporaryDirectory()
        root = Path(tmp.name)
        try:
            for sub in ("a", "backup"):
                path = root / "real_world" / sub / "same.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"identical-bytes")
            image, errors = parse_task(task(2, "u-same.png", []), root)
            self.assertEqual(errors, [])
            self.assertEqual(len(image.duplicate_paths), 1)
            self.assertTrue(image.relpath.replace("\\", "/").endswith(
                "real_world/a/same.png"))
        finally:
            tmp.cleanup()

    def test_upload_reference_mismatch_held_out(self):
        tmp, root = tree_with("real_world/a.png")
        try:
            broken = task(2, "u-a.png", [])
            broken["data"]["image"] = "/data/upload/1/u-other.png"
            image, errors = parse_task(broken, root)
            self.assertIsNone(image)
        finally:
            tmp.cleanup()

    def test_unknown_source_not_guessed(self):
        tmp, root = tree_with("mystery/a.png")
        try:
            image, errors = parse_task(task(2, "u-a.png", []), root)
            self.assertEqual(errors, [])
            self.assertEqual(image.source, "unknown")
        finally:
            tmp.cleanup()


class TestGeometry(unittest.TestCase):
    def test_percent_to_pixels_identity(self):
        corners = ls_rotation_to_corners(50, 25, 100, 50, 0)
        self.assertEqual(corners, [(50, 25), (150, 25), (150, 75), (50, 75)])

    def test_rotation_is_clockwise_about_top_left(self):
        corners = ls_rotation_to_corners(0, 0, 10, 10, 90)
        for got, want in zip(corners, [(0, 0), (0, 10), (-10, 10), (-10, 0)]):
            self.assertAlmostEqual(got[0], want[0])
            self.assertAlmostEqual(got[1], want[1])

    def test_rect_polygon_uses_original_dims(self):
        tmp, root = tree_with("real_world/a.png")
        try:
            image, _ = parse_task(
                task(2, "u-a.png", [rect(["window"], 25, 25, 50, 50)]),
                root, )
            (parsed,) = image.rects
            self.assertEqual(
                rect_polygon(parsed, 200, 100),
                [(50, 25), (150, 25), (150, 75), (50, 75)])
        finally:
            tmp.cleanup()

    def test_rotated_iou_known_values(self):
        sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertEqual(rotated_iou(sq, sq), 1.0)
        far = [(20, 20), (30, 20), (30, 30), (20, 30)]
        self.assertEqual(rotated_iou(sq, far), 0.0)
        c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
        rot = [(5 + (x - 5) * c - (y - 5) * s, 5 + (x - 5) * s + (y - 5) * c)
               for x, y in sq]
        self.assertAlmostEqual(rotated_iou(sq, rot), math.sqrt(2) / 2, places=6)

    def test_axis_aligned_pred_vs_rotated_truth_respects_angle(self):
        # GT square rotated 90 deg about its top-left leaves the frame x>=0
        # entirely: an axis-aligned box on the unrotated frame must score 0,
        # where naive AABB-of-GT scoring would wrongly score 1.
        gt = ls_rotation_to_corners(0, 0, 10, 10, 90)
        pred = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertAlmostEqual(rotated_iou(pred, gt), 0.0)

    def test_clip_to_image(self):
        clipped = clip_polygon_to_image([(-5, 5), (5, 5), (5, 15), (-5, 15)], 10, 10)
        for x, y in clipped:
            self.assertTrue(0 - 1e-9 <= x <= 10 + 1e-9)
            self.assertTrue(0 - 1e-9 <= y <= 10 + 1e-9)
        self.assertEqual(clip_polygon_to_image(
            [(50, 50), (60, 50), (60, 60), (50, 60)], 10, 10), [])


class TestMatching(unittest.TestCase):
    def test_class_constraint(self):
        box = [(0, 0), (10, 0), (10, 10), (0, 10)]
        same = match_rotated([{"polygon": box, "category": "crack"}],
                             [{"polygon": box, "category": "crack"}])
        self.assertEqual((same.tp, same.fp, same.fn), (1, 0, 0))
        cross = match_rotated([{"polygon": box, "category": "crack"}],
                              [{"polygon": box, "category": "spalling"}])
        self.assertEqual((cross.tp, cross.fp, cross.fn), (0, 1, 1))

    def test_duplicates_and_malformed(self):
        box = [(0, 0), (10, 0), (10, 10), (0, 10)]
        gt = [{"polygon": box, "category": "crack"}]
        dup = match_rotated([{"polygon": box, "category": "crack"}] * 2, gt)
        self.assertEqual((dup.tp, dup.fp, dup.fn), (1, 1, 0))
        bad = match_rotated([{"polygon": [(0, 0)], "category": "crack"}], gt)
        self.assertEqual((bad.tp, bad.fp, bad.fn), (0, 1, 1))
        self.assertEqual(bad.malformed_pred_indices, [0])
        with self.assertRaises(ValueError):
            match_rotated([{"polygon": box, "category": "crack"}],
                          [{"polygon": [(0, 0)], "category": "crack"}])


class TestBoundariesAndMalformed(unittest.TestCase):
    def test_fp_noise_snapped_and_recorded(self):
        tmp, root = tree_with("real_world/a.png")
        try:
            image, errors = parse_task(
                task(2, "u-a.png", [rect(["crack"], -2e-16, 0, 10, 10)]), root)
            self.assertEqual(errors, [])
            self.assertTrue(image.rects[0].snapped)
            self.assertEqual(image.rects[0].x_pct, 0.0)
        finally:
            tmp.cleanup()

    def test_genuine_overflow_kept_and_flagged(self):
        tmp, root = tree_with("real_world/a.png")
        try:
            image, errors = parse_task(
                task(2, "u-a.png", [rect(["crack"], 30, 15, 76.5, 6.8, rotation=24)]),
                root)
            self.assertEqual(errors, [])
            (parsed,) = image.rects
            self.assertTrue(parsed.clipped)
            self.assertEqual(parsed.w_pct, 76.5)  # verbatim, not shrunk
            self.assertEqual(parsed.rotation_deg, 24)
        finally:
            tmp.cleanup()

    def test_malformed_results_hold_image_out(self):
        tmp, root = tree_with("real_world/a.png")
        try:
            bad_values = [
                rect([], 0, 0, 10, 10),                       # no label
                rect(["crack"], 0, 0, 0, 10),                 # zero width
                rect(["crack"], 0, 0, 10, 10, rotation=float("nan")),
            ]
            for i, value in enumerate(bad_values):
                image, errors = parse_task(task(2, "u-a.png", [value]), root)
                self.assertIsNone(image, f"case {i} parsed but should not")
                self.assertTrue(errors)
            skewed = task(2, "u-a.png", [rect(["crack"], 0, 0, 10, 10)])
            skewed["annotations"][0]["result"][0]["image_rotation"] = 90
            image, errors = parse_task(skewed, root)
            self.assertIsNone(image)
            skewed = task(2, "u-a.png", [rect(["crack"], 0, 0, 10, 10)])
            skewed["annotations"][0]["result"][0]["original_width"] = 0
            image, errors = parse_task(skewed, root)
            self.assertIsNone(image)
        finally:
            tmp.cleanup()


class TestRotatedEvaluation(unittest.TestCase):
    def predictions(self, sha, phrase, boxes):
        return {(sha, phrase): {"boxes": boxes, "rasters": [None] * len(boxes)}}

    def test_negative_images_count_false_alarms(self):
        tmp, root = tree_with("real_world/neg.png")
        try:
            neg, _ = parse_task(task(17, "u-neg.png", []), root)
            self.assertTrue(neg.is_negative)
            preds = self.predictions(neg.sha256, "crack in wall",
                                     [{"x0": 1, "y0": 1, "x1": 5, "y1": 5}])
            doc = run_rotated_evaluation([neg], preds)
            self.assertEqual(doc["negative_fp"]["total"], 1)
            self.assertEqual(doc["negative_fp"]["by_category"], {"crack": 1})
            (cell,) = doc["negative_fp"]["per_image"]
            self.assertEqual((cell["task_id"], cell["fp"]), (17, 1))
        finally:
            tmp.cleanup()

    def test_rotated_gt_matches_axis_aligned_pred(self):
        # GT rect (20,0)-(40,10)px rotated 90 deg clockwise about its
        # top-left lands exactly on (10,0)-(20,20): a pred there is a
        # true positive, while a pred on the unrotated frame misses.
        tmp, root = tree_with("real_world/a.png")
        try:
            image, _ = parse_task(
                task(2, "u-a.png", [rect(["crack"], 10, 0, 10, 10, rotation=90)]),
                root)
            hit = self.predictions(
                image.sha256, "crack in wall",
                [{"x0": 10, "y0": 0, "x1": 20, "y1": 20}])
            doc = run_rotated_evaluation([image], hit)
            crack = doc["results"]["crack"]["crack in wall"]
            self.assertEqual((crack["tp"], crack["fp"], crack["fn"]), (1, 0, 0))
            miss = self.predictions(
                image.sha256, "crack in wall",
                [{"x0": 20, "y0": 0, "x1": 40, "y1": 10}])
            doc = run_rotated_evaluation([image], miss)
            crack = doc["results"]["crack"]["crack in wall"]
            self.assertEqual((crack["tp"], crack["fp"], crack["fn"]), (0, 1, 1))
        finally:
            tmp.cleanup()

    def test_unmapped_phrase_is_loud(self):
        tmp, root = tree_with("real_world/a.png")
        try:
            image, _ = parse_task(
                task(2, "u-a.png", [rect(["crack"], 0, 0, 10, 10)]), root)
            preds = self.predictions(
                image.sha256, "mystery phrase", [dict(x0=0, y0=0, x1=5, y1=5)])
            doc = run_rotated_evaluation([image], preds)
            self.assertEqual(doc["unevaluated_phrases"], {"mystery phrase": 1})
        finally:
            tmp.cleanup()

    def test_unmapped_gt_label_warns_loudly(self):
        tmp, root = tree_with("real_world/a.png")
        try:
            image, _ = parse_task(
                task(2, "u-a.png", [rect(["spalling"], 0, 0, 10, 10)]), root)
            doc = run_rotated_evaluation([image], {})
            self.assertTrue(any("spalling" in w and "no prompts mapped" in w
                                for w in doc["warnings"]))
            self.assertNotIn("spalling", doc["results"])
        finally:
            tmp.cleanup()

    def test_empty_class_reports_undefined_not_zero(self):
        tmp, root = tree_with("real_world/a.png")
        try:
            image, _ = parse_task(
                task(2, "u-a.png", [rect(["crack"], 0, 0, 10, 10)]), root)
            doc = run_rotated_evaluation([image], {})
            rebar = doc["results"]["rebar"]
            self.assertTrue(all(v["precision"] is None for v in rebar.values()))
            self.assertTrue(any(w.startswith("rebar:") for w in doc["warnings"]))
        finally:
            tmp.cleanup()


class TestRealExport(unittest.TestCase):
    def test_export_loads_with_expected_shape(self):
        tasks = load_export(EXPORT)
        self.assertEqual(len(tasks), 29)
        self.assertEqual(sorted(t["id"] for t in tasks), list(range(2, 31)))

    def test_full_validation_recounts(self):
        from neev.eval.labelstudio import validate_export

        report = validate_export(load_export(EXPORT), REPO / "dataset",
                                 check_dims=False)
        self.assertEqual(report["tasks"], 29)
        self.assertEqual(report["images_parsed"], 29)
        self.assertEqual(report["total_annotations"], 657)
        self.assertEqual(len(report["per_class"]), 14)
        self.assertEqual(report["empty_negative_images"], [17, 19])
        self.assertEqual(report["rotated_boxes"], 186)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["duplicate_image_ids"], [])

    def test_nothing_lost_in_conversion(self):
        tasks = load_export(EXPORT)
        results = sum(len(t["annotations"][0]["result"] or []) for t in tasks)
        parsed = sum(len(i.rects) for i in
                     [parse_task(t, REPO / "dataset")[0] for t in tasks])
        self.assertEqual(results, 657)
        self.assertGreaterEqual(parsed, results)

    def test_pipeline_leaves_protected_files_untouched(self):
        export_before = EXPORT.read_bytes()
        bench_before = (REPO / "eval" / "benchmark_v1.json").read_bytes()
        runs_before = sorted(p.name for p in (REPO / "runs").iterdir())
        report = validate_export(load_export(EXPORT), REPO / "dataset",
                                 check_dims=False)
        doc = run_rotated_evaluation(
            [parse_task(t, REPO / "dataset")[0] for t in load_export(EXPORT)], {})
        self.assertEqual(report["total_annotations"], 657)
        self.assertIn("crack", doc["results"])
        self.assertEqual(EXPORT.read_bytes(), export_before)
        self.assertEqual((REPO / "eval" / "benchmark_v1.json").read_bytes(),
                         bench_before)
        self.assertEqual(sorted(p.name for p in (REPO / "runs").iterdir()),
                         runs_before)


if __name__ == "__main__":
    unittest.main()
