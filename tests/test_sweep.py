"""Unit tests: live/sweep_sam31.py. No network, no billing.

Mirrors the smoke-test harness: an exploding transport proves dry runs
send nothing; a canned fake proves manifest statuses, caching, and
trace hygiene. The real .env is never read.
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

SWEEP_PATH = Path(__file__).resolve().parent.parent / "live" / "sweep_sam31.py"

JPEG_BYTES = b"\xff\xd8\xff" + b"\x00" * 100

OFFICIAL_LINE = (
    "<0f>0<|box;x1=10;y1=10;x2=50;y2=60;w=100;h=100|>"
    "<|mask;x=0;y=0;data=2,2,!AB|>"
)


def load_sweep():
    spec = importlib.util.spec_from_file_location("sweep_sam31", SWEEP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = load_sweep()


class ExplodingTransport:
    def post_json(self, url, headers, payload, timeout=0):
        raise AssertionError("network request attempted")


class FakeTransport:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def post_json(self, url, headers, payload, timeout=0):
        self.calls.append(payload)
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


class SweepCase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "a.jpg").write_bytes(JPEG_BYTES)
        (self.root / "b.jpg").write_bytes(JPEG_BYTES + b"\x01")

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *argv, env=None):
        import io
        import os

        with mock.patch.object(sys, "argv", ["sweep_sam31.py", *argv]):
            with mock.patch.dict("os.environ", env or {}, clear=True):
                old = os.getcwd()
                os.chdir(self.root)
                try:
                    out = io.StringIO()
                    with mock.patch("sys.stdout", out):
                        code = sweep.main()
                    return code, out.getvalue()
                finally:
                    os.chdir(old)

    def phrases_of(self, calls):
        return [c["input"][0]["content"][0]["text"] for c in calls]


class TestSweepPlan(SweepCase):
    def test_default_prompts_are_the_docs_sixteen(self):
        self.assertEqual(len(sweep.default_prompts()), 16)
        self.assertEqual(sweep.default_prompts()[0], "building")
        self.assertIn("open ground floor", sweep.default_prompts())
        self.assertIn("license plate", sweep.default_prompts())

    def test_dry_run_sends_nothing_and_counts_uncached(self):
        with mock.patch(
            "neev.adapters.base.UrllibTransport.post_json",
            ExplodingTransport().post_json,
        ):
            code, text = self.run_main(
                "a.jpg", "b.jpg", "--phrase", "building", "--phrase", "window",
                "--runs-dir", str(self.root / "runs"),
            )
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN", text)
        self.assertIn("uncached: 4", text)
        self.assertIn("= $0.0100", text)
        self.assertEqual(list((self.root / "runs").glob("sweep_*.json")), [])

    def test_directory_expands_to_images_only(self):
        (self.root / "note.txt").write_text("nope", encoding="utf-8")
        with mock.patch(
            "neev.adapters.base.UrllibTransport.post_json",
            ExplodingTransport().post_json,
        ):
            code, text = self.run_main(
                ".", "--phrase", "building",
                "--runs-dir", str(self.root / "runs"),
            )
        self.assertEqual(code, 0)
        self.assertIn("images : 2", text)
        self.assertIn("cells: 2", text)

    def test_missing_key_blocks_live(self):
        with mock.patch(
            "neev.adapters.base.UrllibTransport.post_json",
            ExplodingTransport().post_json,
        ):
            with mock.patch("sys.stderr"):
                code, _ = self.run_main("a.jpg", "--confirm-live")
        self.assertEqual(code, 2)

    def test_invalid_image_recorded_without_network(self):
        (self.root / "note.txt").write_text("nope", encoding="utf-8")
        with mock.patch(
            "neev.adapters.base.UrllibTransport.post_json",
            ExplodingTransport().post_json,
        ):
            code, _ = self.run_main(
                "a.jpg", "note.txt", "missing.jpg",
                "--phrase", "building", "--confirm-live",
                "--runs-dir", str(self.root / "runs"),
                env={"MODEL_API_KEY": "fake-key-xyz"},
            )
        self.assertEqual(code, 1)  # invalid cells fail the run, no calls made


class TestSweepLive(SweepCase):
    def run_live(self, transport, *extra):
        with mock.patch(
            "neev.adapters.base.UrllibTransport.post_json", transport.post_json
        ):
            return self.run_main(
                "a.jpg", "--phrase", "building", "--phrase", "window",
                "--confirm-live", "--runs-dir", str(self.root / "runs"),
                *extra, env={"MODEL_API_KEY": "fake-key-xyz"},
            )

    def manifest(self):
        paths = list((self.root / "runs").glob("sweep_*.json"))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text(encoding="utf-8"))

    def test_live_writes_manifest_with_statuses(self):
        fake = FakeTransport([{"output_text": OFFICIAL_LINE}])
        code, _ = self.run_live(fake)
        self.assertEqual(code, 0)
        self.assertEqual(len(fake.calls), 2)
        doc = self.manifest()
        self.assertEqual(doc["summary"], {
            "cells": 2, "ok": 2, "cached": 0, "empty": 0,
            "failed": 0, "invalid": 0, "sent": 2,
        })
        cell = doc["cells"][0]
        self.assertEqual(cell["status"], "ok")
        self.assertEqual(cell["detections"], 1)
        self.assertEqual(cell["masks_with_payload"], 1)
        self.assertEqual(cell["masks_known_encoding"], 1)
        blob = json.dumps(doc)
        self.assertNotIn("fake-key-xyz", blob)

    def test_empty_valid_and_failed_stay_distinct(self):
        fake = FakeTransport([{"output_text": "  "}, {"output_text": "garbage!!"}])
        code, _ = self.run_live(fake)
        self.assertEqual(code, 1)
        doc = self.manifest()
        by_phrase = {c["phrase"]: c for c in doc["cells"]}
        self.assertEqual(by_phrase["building"]["status"], "ok")
        self.assertTrue(by_phrase["building"]["empty"])
        self.assertEqual(by_phrase["window"]["status"], "failed")
        self.assertIn("AdapterError", by_phrase["window"]["error"])

    def test_rerun_is_all_cached_with_no_calls(self):
        fake = FakeTransport([{"output_text": OFFICIAL_LINE}])
        code, _ = self.run_live(fake)
        self.assertEqual(code, 0)
        code, text = self.run_live(ExplodingTransport())
        self.assertEqual(code, 0)
        self.assertIn("cached : 2", text)


@unittest.skipUnless(
    importlib.util.find_spec("PIL") is not None, "overlay test needs Pillow"
)
class TestSweepOverlays(SweepCase):
    def test_overlays_render_per_cell(self):
        from PIL import Image

        Image.new("RGB", (100, 100), "white").save(self.root / "real.jpg")
        fake = FakeTransport([{"output_text": OFFICIAL_LINE}])
        with mock.patch(
            "neev.adapters.base.UrllibTransport.post_json", fake.post_json
        ):
            code, _ = self.run_main(
                "real.jpg", "--phrase", "building", "--confirm-live",
                "--overlays", "--runs-dir", str(self.root / "runs"),
                env={"MODEL_API_KEY": "fake-key-xyz"},
            )
        self.assertEqual(code, 0)
        overlays = list((self.root / "runs").rglob("overlay.png"))
        self.assertEqual(len(overlays), 1)


if __name__ == "__main__":
    unittest.main()
