"""Unit tests: benchmark format, matching, variant compare, crops.

No network, no API, no real labels: fixtures are synthetic boxes with
known IoUs, plus the repo's honestly-empty benchmark file. Anything
claiming measured quality must come from labelled data, never these.
"""

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from neev.eval.benchmark import (
    label_status,
    load_benchmark,
    validate_benchmark,
)
from neev.eval.compare import collect_predictions, run_evaluation
from neev.eval.crops import (
    load_crops_config,
    materialize,
    validate_crop_box,
    validate_crops_config,
)
from neev.eval.matching import mask_iou, match_predictions, prf, rle_decode

BOX_A = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
BOX_A_SHIFT = {"x0": 1, "y0": 1, "x1": 11, "y1": 11}  # IoU ~0.68 with A
BOX_FAR = {"x0": 50, "y0": 50, "x1": 60, "y1": 60}


def bench_doc(**overrides):
    doc = {
        "version": 1,
        "name": "fixture",
        "images": [
            {"image_id": "a", "path": "a.jpg", "sha256": "0" * 64,
             "width": 100, "height": 100,
             "labels": {
                 "window": {"present": True, "boxes": [dict(BOX_A)], "masks": []},
                 "crack": {"present": False, "boxes": [], "masks": []},
             }},
            {"image_id": "b", "path": "b.jpg", "sha256": "1" * 64,
             "width": 100, "height": 100, "labels": {}},
        ],
    }
    doc.update(overrides)
    return doc


class TestBenchmarkFormat(unittest.TestCase):
    def test_valid_fixture_passes(self):
        self.assertEqual(validate_benchmark(bench_doc()), [])

    def test_repo_benchmark_loads_and_is_fully_unlabelled(self):
        path = Path(__file__).resolve().parent.parent / "eval" / "benchmark_v1.json"
        doc = load_benchmark(path)
        self.assertEqual(len(doc["images"]), 8)
        for entry in doc["images"]:
            for category in ("building", "window", "crack", "rebar"):
                self.assertEqual(label_status(entry["labels"], category), "unlabelled")

    def test_rejects_bad_version_duplicates_and_sha(self):
        self.assertTrue(validate_benchmark(bench_doc(version=99)))
        dup = bench_doc()
        dup["images"].append(dict(dup["images"][0]))
        self.assertTrue(validate_benchmark(dup))
        bad_sha = bench_doc()
        bad_sha["images"][0]["sha256"] = "abc"
        self.assertTrue(validate_benchmark(bad_sha))

    def test_rejects_bad_gt_boxes(self):
        for bad in [
            {"x0": 5, "y0": 5, "x1": 5, "y1": 9},      # zero area
            {"x0": 9, "y0": 0, "x1": 2, "y1": 9},      # inverted
            {"x0": 0, "y0": 0, "x1": 101, "y1": 9},    # outside dims
            {"x0": True, "y0": 0, "x1": 9, "y1": 9},   # bool is not an int
            {"x0": 0, "y0": 0, "x1": 9},               # missing key
        ]:
            doc = bench_doc()
            doc["images"][0]["labels"]["window"]["boxes"] = [bad]
            self.assertTrue(validate_benchmark(doc), f"accepted {bad!r}")

    def test_rejects_boxes_on_negative_or_unknown(self):
        doc = bench_doc()
        doc["images"][0]["labels"]["crack"]["boxes"] = [dict(BOX_A)]
        self.assertTrue(validate_benchmark(doc))
        doc = bench_doc()
        doc["images"][0]["labels"]["parapet"] = {
            "present": None, "boxes": [dict(BOX_A)], "masks": []}
        self.assertTrue(validate_benchmark(doc))

    def test_rejects_unknown_category_and_bad_rle(self):
        doc = bench_doc()
        doc["images"][0]["labels"]["bogus"] = {"present": False, "boxes": [], "masks": []}
        self.assertTrue(validate_benchmark(doc))
        doc = bench_doc()
        doc["images"][0]["labels"]["window"]["masks"] = [
            {"width": 2, "height": 2, "counts": [1, 1]}]  # sums to 2, want 4
        self.assertTrue(validate_benchmark(doc))

    def test_label_status_four_states(self):
        labels = {
            "a": {"present": True, "boxes": [dict(BOX_A)], "masks": []},
            "b": {"present": True, "boxes": [], "masks": []},
            "c": {"present": False, "boxes": [], "masks": []},
            "d": {"present": None, "boxes": [], "masks": []},
        }
        self.assertEqual(label_status(labels, "a"), "labelled")
        self.assertEqual(label_status(labels, "b"), "presence_only")
        self.assertEqual(label_status(labels, "c"), "negative")
        self.assertEqual(label_status(labels, "d"), "unlabelled")
        self.assertEqual(label_status(labels, "missing"), "unlabelled")


class TestMatching(unittest.TestCase):
    def test_perfect_match(self):
        result = match_predictions([dict(BOX_A)], [dict(BOX_A)])
        self.assertEqual((result.tp, result.fp, result.fn), (1, 0, 0))

    def test_partial_match_scores(self):
        result = match_predictions(
            [dict(BOX_A_SHIFT), dict(BOX_FAR)], [dict(BOX_A), dict(BOX_FAR)])
        self.assertEqual((result.tp, result.fp, result.fn), (2, 0, 0))
        result = match_predictions([dict(BOX_A_SHIFT)], [dict(BOX_A), dict(BOX_FAR)])
        self.assertEqual((result.tp, result.fp, result.fn), (1, 0, 1))
        self.assertEqual(prf(1, 1, 1),
                         {"precision": 0.5, "recall": 0.5, "f1": 0.5})

    def test_duplicates_are_false_positives(self):
        result = match_predictions([dict(BOX_A), dict(BOX_A)], [dict(BOX_A)])
        self.assertEqual((result.tp, result.fp, result.fn), (1, 1, 0))

    def test_malformed_predictions_are_listed_fps(self):
        preds = [{"x0": 5, "y0": 5, "x1": 5, "y1": 9}, {"x0": 0}]
        result = match_predictions(preds, [dict(BOX_A)])
        self.assertEqual((result.tp, result.fp, result.fn), (0, 2, 1))
        self.assertEqual(result.malformed_pred_indices, [0, 1])

    def test_malformed_ground_truth_raises(self):
        with self.assertRaises(ValueError):
            match_predictions([dict(BOX_A)], [{"x0": 0}])

    def test_empty_sides(self):
        result = match_predictions([], [dict(BOX_A)])
        self.assertEqual((result.tp, result.fp, result.fn), (0, 0, 1))
        self.assertEqual(prf(0, 0, 1)["recall"], 0.0)
        self.assertIsNone(prf(0, 0, 1)["precision"])
        result = match_predictions([dict(BOX_A)], [])
        self.assertEqual(prf(0, 1, 0)["precision"], 0.0)
        self.assertIsNone(prf(0, 1, 0)["recall"])
        self.assertEqual(prf(0, 0, 0),
                         {"precision": None, "recall": None, "f1": None})

    def test_matching_is_order_stable(self):
        gt = [dict(BOX_A), dict(BOX_FAR)]
        first = match_predictions([dict(BOX_FAR), dict(BOX_A_SHIFT)], gt)
        second = match_predictions([dict(BOX_A_SHIFT), dict(BOX_FAR)], gt)
        self.assertEqual((first.tp, first.fp, first.fn),
                         (second.tp, second.fp, second.fn))

    def test_rle_and_mask_iou(self):
        raster = rle_decode([2, 2], 2, 2)
        self.assertEqual(raster, b"\x00\x00\x01\x01")
        with self.assertRaises(ValueError):
            rle_decode([1], 2, 2)
        self.assertEqual(mask_iou(b"\x01\x01", b"\x01\x01"), 1.0)
        self.assertEqual(mask_iou(b"\x01\x00", b"\x00\x01"), 0.0)
        with self.assertRaises(ValueError):
            mask_iou(b"\x01", b"\x01\x01")


class TestEvaluation(unittest.TestCase):
    def predictions(self):
        return {
            ("0" * 64, "window"): {"boxes": [dict(BOX_A_SHIFT)], "rasters": [None]},
            ("1" * 64, "window"): {"boxes": [dict(BOX_FAR)], "rasters": [None]},
        }

    def test_labelled_counted_unlabelled_excluded(self):
        doc = run_evaluation(bench_doc(), self.predictions())
        window = doc["results"]["window"]["window"]
        self.assertEqual((window["tp"], window["fp"], window["fn"]), (1, 0, 0))
        self.assertEqual(window["images_labelled"], 1)
        self.assertEqual(window["images_excluded"], 1)
        # The prediction on unlabelled image b is NOT a false positive.
        img_b = [c for c in window["per_image"] if c["image_id"] == "b"][0]
        self.assertEqual(img_b["status"], "unlabelled")

    def test_negative_images_count_false_alarms(self):
        preds = {("0" * 64, "crack in wall"): {
            "boxes": [dict(BOX_A)], "rasters": [None]}}
        doc = run_evaluation(bench_doc(), preds)
        crack = doc["results"]["crack"]["crack in wall"]
        self.assertEqual((crack["tp"], crack["fp"], crack["fn"]), (0, 1, 0))

    def test_negative_uses_same_malformed_rule_as_matching(self):
        preds = {("0" * 64, "crack in wall"): {"boxes": [
            {"x0": True, "y0": 0, "x1": 9, "y1": 9}], "rasters": [None]}}
        doc = run_evaluation(bench_doc(), preds)
        cell = doc["results"]["crack"]["crack in wall"]["per_image"][0]
        self.assertEqual(cell["status"], "negative")
        self.assertEqual(cell["fp"], 1)
        self.assertEqual(cell["malformed"], 1)

    def test_spalling_category_accepted_without_mapped_prompts(self):
        doc = bench_doc()
        doc["images"][0]["labels"]["spalling"] = {
            "present": True, "boxes": [dict(BOX_A)], "masks": []}
        self.assertEqual(validate_benchmark(doc), [])
        out = run_evaluation(doc, {})
        self.assertTrue(any("spalling" in w and "no prompts mapped" in w
                            for w in out["warnings"]))

    def test_labelled_without_predictions_warns_distinctly(self):
        out = run_evaluation(bench_doc(), {})
        self.assertTrue(any("window" in w and "no cached predictions" in w
                            for w in out["warnings"]))

    def test_presence_only_and_missing_predictions(self):
        doc = bench_doc()
        doc["images"][0]["labels"]["window"] = {
            "present": True, "boxes": [], "masks": []}
        out = run_evaluation(doc, self.predictions())
        window = out["results"]["window"]["window"]
        self.assertEqual(window["images_labelled"], 0)
        self.assertTrue(any(c["status"] == "presence_only"
                            for c in window["per_image"]))
        out = run_evaluation(bench_doc(), {})
        window = out["results"]["window"]["window"]
        self.assertTrue(any(c["status"] == "no_predictions"
                            for c in window["per_image"]))

    def test_warnings_name_unmeasured_categories(self):
        doc = run_evaluation(bench_doc(), self.predictions())
        self.assertTrue(any(w.startswith("crack:") for w in doc["warnings"]))

    def test_results_are_reproducible(self):
        first = run_evaluation(bench_doc(), self.predictions())
        second = run_evaluation(bench_doc(), self.predictions())
        for doc in (first, second):
            doc.pop("created_utc")
        self.assertEqual(json.dumps(first, sort_keys=True),
                         json.dumps(second, sort_keys=True))


class TestCollectPredictions(unittest.TestCase):
    LINE = ("<0f>0<|box;x1=10;y1=10;x2=50;y2=60;w=100;h=100|>"
            "<|mask;x=0;y=0;data=2,2,!AB|>")

    def write_run(self, root, run_id, phrase, sha, text):
        run_dir = Path(root) / run_id
        run_dir.mkdir()
        (run_dir / "meta.json").write_text(json.dumps({
            "notes": {"phrase": phrase}, "input_hashes": {"image": sha}}))
        (run_dir / "response.json").write_text(json.dumps({"output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": text}]}]}))

    def test_collects_by_sha_and_phrase(self):
        with TemporaryDirectory() as tmp:
            self.write_run(tmp, "r1", "window", "a" * 64, self.LINE)
            self.write_run(tmp, "r2", "window", "a" * 64, "  ")
            got = collect_predictions(tmp, run_ids=["r1"])
        self.assertEqual(
            got[("a" * 64, "window")]["boxes"],
            [{"x0": 10, "y0": 10, "x1": 50, "y1": 60}])

    def test_unparseable_trace_recorded_not_raised(self):
        with TemporaryDirectory() as tmp:
            self.write_run(tmp, "bad", "window", "a" * 64, "garbage!!")
            got = collect_predictions(tmp)
        self.assertIn(("errors", ""), got)

    def test_mask_backend_decodes_locally(self):
        with TemporaryDirectory() as tmp:
            self.write_run(tmp, "r1", "window", "a" * 64, self.LINE)
            got = collect_predictions(
                tmp, mask_backend=lambda payload, w, h: b"\x01" * (w * h))
        self.assertEqual(got[("a" * 64, "window")]["rasters"], [b"\x01" * 4])


class TestCrops(unittest.TestCase):
    def test_config_validation(self):
        good = {"version": 1, "crops": [{
            "crop_id": "c1", "image": "a.jpg",
            "box": {"x0": 0, "y0": 0, "x1": 5, "y1": 5},
            "purpose": "why", "status": "approved"}]}
        self.assertEqual(validate_crops_config(good), [])
        bad = {"version": 1, "crops": [
            dict(good["crops"][0]),
            dict(good["crops"][0], status="maybe"),
        ]}
        self.assertTrue(validate_crops_config(bad))

    def test_repo_example_config_loads(self):
        path = Path(__file__).resolve().parent.parent / "eval" / "crops_example.json"
        doc = load_crops_config(path)
        # The example crop was eyeballed on its source image: approved.
        self.assertEqual(doc["crops"][0]["status"], "approved")

    def test_box_bounds(self):
        good = {"x0": 0, "y0": 0, "x1": 50, "y1": 40}
        self.assertEqual(validate_crop_box(good, 100, 100), [])
        for bad in [dict(good, x1=101), dict(good, x0=50),
                    dict(good, y1=40, y0=40), "nope"]:
            self.assertTrue(validate_crop_box(bad, 100, 100))

    @unittest.skipUnless(importlib.util.find_spec("PIL") is not None,
                         "materialize needs Pillow")
    def test_materialize_writes_approved_only(self):
        from PIL import Image

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (50, 40), "red").save(root / "a.jpg")
            config = {"version": 1, "crops": [
                {"crop_id": "ok", "image": "a.jpg",
                 "box": {"x0": 5, "y0": 5, "x1": 15, "y1": 15},
                 "purpose": "t", "status": "approved"},
                {"crop_id": "prov", "image": "a.jpg",
                 "box": {"x0": 0, "y0": 0, "x1": 10, "y1": 10},
                 "purpose": "t", "status": "provisional"},
            ]}
            manifest = materialize(config, root, root / "out")
            self.assertEqual([m["crop_id"] for m in manifest], ["ok"])
            self.assertEqual(tuple(manifest[0]["size"]), (10, 10))
            self.assertEqual(len(manifest[0]["sha256"]), 64)
            manifest = materialize(config, root, root / "out2",
                                   include_provisional=True)
            self.assertEqual(len(manifest), 2)

    @unittest.skipUnless(importlib.util.find_spec("PIL") is not None,
                         "materialize needs Pillow")
    def test_materialize_rejects_out_of_bounds(self):
        from PIL import Image

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (50, 40), "red").save(root / "a.jpg")
            config = {"version": 1, "crops": [{
                "crop_id": "bad", "image": "a.jpg",
                "box": {"x0": 0, "y0": 0, "x1": 500, "y1": 10},
                "purpose": "t", "status": "approved"}]}
            with self.assertRaises(ValueError):
                materialize(config, root, root / "out")


if __name__ == "__main__":
    unittest.main()
