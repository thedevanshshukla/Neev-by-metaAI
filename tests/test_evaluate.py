"""Unit tests: live/evaluate.py. Offline by construction.

Proves the runner spends nothing (exploding transport), scores fixture
labels against synthetic traces, stays deterministic, and reports
unlabelled benchmarks honestly instead of inventing zeros.
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

EVALUATE_PATH = Path(__file__).resolve().parent.parent / "live" / "evaluate.py"

LINE = ("<0f>0<|box;x1=10;y1=10;x2=50;y2=60;w=100;h=100|>"
        "<|mask;x=0;y=0;data=2,2,!AB|>")


def load_evaluate():
    spec = importlib.util.spec_from_file_location("neev_evaluate", EVALUATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate = load_evaluate()


def fixture_benchmark():
    return {
        "version": 1,
        "name": "fixture",
        "images": [
            {"image_id": "a", "path": "a.jpg", "sha256": "0" * 64,
             "width": 100, "height": 100,
             "labels": {"window": {
                 "present": True,
                 "boxes": [{"x0": 10, "y0": 10, "x1": 50, "y1": 60}],
                 "masks": []}}},
            {"image_id": "b", "path": "b.jpg", "sha256": "1" * 64,
             "width": 100, "height": 100, "labels": {}},
        ],
    }


def write_trace(root, run_id, phrase, sha, text):
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps({
        "notes": {"phrase": phrase}, "input_hashes": {"image": sha}}))
    (run_dir / "response.json").write_text(json.dumps({"output_text": text}))


class EvaluateCase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "bench.json").write_text(json.dumps(fixture_benchmark()))
        runs = self.root / "runs"
        write_trace(runs, "r1", "window", "0" * 64, LINE)

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *argv):
        import io
        import os

        with mock.patch.object(sys, "argv", ["evaluate.py", *argv]):
            old = os.getcwd()
            os.chdir(self.root)
            try:
                out = io.StringIO()
                with mock.patch("sys.stdout", out):
                    code = evaluate.main()
                return code, out.getvalue()
            finally:
                os.chdir(old)


class TestEvaluate(EvaluateCase):
    def test_offline_scoring_with_exploding_transport(self):
        with mock.patch(
            "neev.adapters.base.UrllibTransport.post_json",
            side_effect=AssertionError("network!"),
        ):
            code, text = self.run_main(
                "--benchmark", "bench.json", "--out", "results.json")
        self.assertEqual(code, 0)
        doc = json.loads((self.root / "results.json").read_text())
        window = doc["results"]["window"]["window"]
        self.assertEqual((window["tp"], window["fp"], window["fn"]), (1, 0, 0))
        self.assertEqual(window["images_labelled"], 1)
        self.assertIn("P=1.0 R=1.0 F1=1.0", text)

    def test_deterministic_results(self):
        self.run_main("--benchmark", "bench.json", "--out", "r1.json")
        self.run_main("--benchmark", "bench.json", "--out", "r2.json")
        first = json.loads((self.root / "r1.json").read_text())
        second = json.loads((self.root / "r2.json").read_text())
        first.pop("created_utc")
        second.pop("created_utc")
        self.assertEqual(first, second)

    def test_unlabelled_benchmark_reports_nothing_measured(self):
        (self.root / "empty.json").write_text(json.dumps({
            "version": 1, "name": "empty", "images": [
                {"image_id": "a", "path": "a.jpg", "sha256": "0" * 64,
                 "width": 100, "height": 100, "labels": {}}]}))
        code, text = self.run_main("--benchmark", "empty.json", "--out", "r.json")
        self.assertEqual(code, 0)
        self.assertIn("0 labelled cells", text)
        doc = json.loads((self.root / "r.json").read_text())
        self.assertTrue(any(w.startswith("window:") for w in doc["warnings"]))

    def test_invalid_benchmark_fails_loudly(self):
        (self.root / "bad.json").write_text("{}")
        with mock.patch("sys.stderr"):
            code, _ = self.run_main("--benchmark", "bad.json")
        self.assertEqual(code, 2)

    def test_decode_masks_without_package_stays_green(self):
        code, _ = self.run_main(
            "--benchmark", "bench.json", "--out", "r.json", "--decode-masks")
        self.assertEqual(code, 0)
        doc = json.loads((self.root / "r.json").read_text())
        self.assertEqual(doc["results"]["window"]["window"]["mask_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
