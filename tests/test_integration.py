"""Integration tests: seams between existing modules, local fixtures only.

No network, no API, no protected files: every test builds its own tmp
benchmark and tmp run traces in the exact on-disk shapes the live
scripts write, then exercises the real handoff (adapters -> prep ->
measurements -> validation -> eval -> persistence). Anything touching
runs/, eval/*.json, or photo_Sample/ does not belong here.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from neev.adapters.base import BoundingBox
from neev.adapters.sam31 import parse_output_text
from neev.eval.compare import collect_predictions, run_evaluation
from neev.measurements.features import count_storeys, overhang_ratio
from neev.persistence.runs import RunMeta, RunStore
from neev.prep.tracking import Tracker
from neev.validation.schemas import validate_building
from neev.validation.uncertainty import score_building

TWO_FRAMES = (
    "<0f>0<|box;x1=10;y1=10;x2=50;y2=60;w=100;h=100|>"
    "<|mask;x=0;y=0;data=2,2,!AB|>\n"
    "<1f>0<|box;x1=12;y1=12;x2=52;y2=62;w=100;h=100|>"
    "<|mask;x=0;y=0;data=2,2,!AB|>"
)

WINDOWS_3_ROWS = (
    "<0f>0<|box;x1=10;y1=5;x2=20;y2=15;w=100;h=100|>\n"
    "<0f>1<|box;x1=10;y1=40;x2=20;y2=50;w=100;h=100|>\n"
    "<0f>2<|box;x1=10;y1=75;x2=20;y2=85;w=100;h=100|>"
)


def write_trace(root, run_id, phrase, sha, text):
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps({
        "notes": {"phrase": phrase}, "input_hashes": {"image": sha}}))
    (run_dir / "response.json").write_text(json.dumps({"output_text": text}))


class TestAdapterToPrep(unittest.TestCase):
    def test_detections_track_across_frames(self):
        dets = parse_output_text(TWO_FRAMES, "building")
        self.assertEqual(len(dets), 2)
        tracker = Tracker(iou_threshold=0.3)
        first = tracker.update([dets[0].box])
        second = tracker.update([dets[1].box])
        self.assertEqual(first[0], second[0])
        self.assertEqual(dets[0].track_id, dets[1].track_id)


class TestAdapterToMeasurements(unittest.TestCase):
    def test_window_detections_count_storeys(self):
        dets = parse_output_text(WINDOWS_3_ROWS, "window")
        result = count_storeys([d.box for d in dets], image_height_px=100)
        self.assertEqual(result.rows, 3)

    def test_building_boxes_feed_overhang(self):
        upper = BoundingBox(0, 0, 115, 50)
        ground = BoundingBox(0, 50, 100, 100)
        self.assertTrue(overhang_ratio(upper.width, ground.width).flag)


class TestChecklistToScore(unittest.TestCase):
    def test_validate_then_score_order(self):
        items = [
            {"id": "open_ground_storey", "answer": "yes",
             "evidence": "f_0001.jpg", "confidence": 0.9, "reason": "seen"},
        ]
        for cid in ("heavy_overhang", "pounding", "falling_hazards",
                    "short_column", "vertical_irregularity",
                    "apparent_quality", "construction_type"):
            items.append({"id": cid, "answer": "no",
                          "evidence": "f_0001.jpg", "confidence": 0.8,
                          "reason": "clear view"})
        doc = {
            "building_id": "B1",
            "best_frames": ["f_0001.jpg"],
            "storeys": {"measured": 4, "judge": 4},
            "checklist": items,
        }
        self.assertEqual(validate_building(doc), [])
        answers = {item["id"]: item["answer"] for item in doc["checklist"]}
        result = score_building(answers, 4)
        self.assertEqual((result.score, result.band), (45, "medium"))
        self.assertFalse(result.refilm)

    def test_invalid_evidence_blocks_scoring(self):
        doc = {
            "building_id": "B1",
            "best_frames": ["f_0001.jpg"],
            "storeys": {"measured": 4, "judge": 4},
            "checklist": [
                {"id": "open_ground_storey", "answer": "yes",
                 "evidence": "f_9999.jpg", "confidence": 0.9, "reason": "seen"},
            ],
        }
        errors = validate_building(doc)
        self.assertTrue(errors)
        # Callers must check errors before scoring; a yes without valid
        # evidence must never silently become points.
        self.assertIn("f_9999.jpg", " ".join(errors))


class TestFullOfflineLoop(unittest.TestCase):
    SHA_A = "a" * 64
    SHA_B = "b" * 64

    def make_world(self, root):
        benchmark = {
            "version": 1, "name": "loop",
            "images": [
                {"image_id": "a", "path": "a.jpg", "sha256": self.SHA_A,
                 "width": 100, "height": 100, "labels": {"window": {
                     "present": True,
                     "boxes": [{"x0": 10, "y0": 10, "x1": 50, "y1": 60}],
                     "masks": []}}},
                {"image_id": "b", "path": "b.jpg", "sha256": self.SHA_B,
                 "width": 100, "height": 100, "labels": {}},
            ],
        }
        runs = Path(root) / "runs"
        write_trace(runs, "r-a-ok", "window", self.SHA_A,
                    "<0f>0<|box;x1=10;y1=10;x2=50;y2=60;w=100;h=100|>")
        write_trace(runs, "r-m-empty", "window", self.SHA_B, "   ")
        (runs / "r-c-corrupt").mkdir(parents=True)
        (runs / "r-c-corrupt" / "meta.json").write_text("{oops")
        (runs / "r-c-corrupt" / "response.json").write_text("{}")
        write_trace(runs, "r-z-dupe", "window", self.SHA_A,
                    "<0f>0<|box;x1=0;y1=0;x2=5;y2=5;w=100;h=100|>")
        return benchmark, runs

    def test_loop_scores_labelled_and_reports_the_rest(self):
        with TemporaryDirectory() as tmp:
            benchmark, runs = self.make_world(tmp)
            predictions = collect_predictions(runs)
            key = (self.SHA_A, "window")
            # Full detection records survive with their source stamped.
            det = predictions[key]["detections"][0]
            self.assertEqual(det["source"], "r-a-ok")
            self.assertEqual(det["box"], {"x0": 10, "y0": 10, "x1": 50, "y1": 60})
            # Corrupt and duplicate traces are recorded, not silent.
            errs = predictions[("errors", "")]["errors"]
            by_run = {e["run_id"]: e["error"] for e in errs}
            self.assertIn("r-c-corrupt", by_run)
            self.assertIn("duplicate", by_run["r-z-dupe"])
            self.assertIn("r-a-ok", by_run["r-z-dupe"])
            results = run_evaluation(benchmark, predictions)
            window = results["results"]["window"]["window"]
            self.assertEqual((window["tp"], window["fp"], window["fn"]), (1, 0, 0))
            statuses = {c["image_id"]: c["status"] for c in window["per_image"]}
            self.assertEqual(statuses, {"a": "evaluated", "b": "unlabelled"})

    def test_collect_is_deterministic(self):
        with TemporaryDirectory() as tmp:
            _, runs = self.make_world(tmp)
            first = collect_predictions(runs)
            second = collect_predictions(runs)
            self.assertEqual(first, second)

    def test_failing_mask_backend_keeps_box_scoring(self):
        def boom(payload, width, height):
            raise RuntimeError("no decoder here")

        with TemporaryDirectory() as tmp:
            _, runs = self.make_world(tmp)
            predictions = collect_predictions(runs, mask_backend=boom)
            key = (self.SHA_A, "window")
            self.assertEqual(predictions[key]["rasters"], [None])
            self.assertEqual(len(predictions[key]["boxes"]), 1)


class TestPersistenceFailures(unittest.TestCase):
    def test_unserializable_notes_fail_loudly(self):
        with TemporaryDirectory() as tmp:
            store = RunStore(tmp)
            meta = RunMeta(purpose="t", models={}, notes={"raw": b"bytes"})
            with self.assertRaises(TypeError):
                store.save({"a": 1}, {}, meta)

    def test_missing_run_is_key_error(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(KeyError):
                RunStore(tmp).load("nope")


if __name__ == "__main__":
    unittest.main()
