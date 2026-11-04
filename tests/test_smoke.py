"""Unit tests: live/smoke_sam31.py safeguards. No network, no billing.

Every test either patches the transport to explode on use (proving no
request is attempted) or substitutes a canned fake (proving call counts,
caching, and trace hygiene). The real .env is never read: tests chdir to
a tmp dir and control the environment explicitly.
"""

import base64
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

SMOKE_PATH = Path(__file__).resolve().parent.parent / "live" / "smoke_sam31.py"

JPEG_BYTES = b"\xff\xd8\xff" + b"\x00" * 100

# Official line shape, one record; mask payload is inert test text.
OFFICIAL_LINE = (
    "<0f>0<|box;x1=10;y1=10;x2=50;y2=60;w=100;h=100|>"
    "<|mask;x=0;y=0;data=2,2,!AB|>"
)


def load_smoke():
    spec = importlib.util.spec_from_file_location("smoke_sam31", SMOKE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = load_smoke()


class ExplodingTransport:
    def post_json(self, url, headers, payload, timeout=0):
        raise AssertionError("network request attempted")


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post_json(self, url, headers, payload, timeout=0):
        self.calls.append(payload)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class SmokeCase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "s.jpg").write_bytes(JPEG_BYTES)

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *argv, env=None):
        import os

        with mock.patch.object(sys, "argv", ["smoke_sam31.py", *argv]):
            with mock.patch.dict("os.environ", env or {}, clear=True):
                old = os.getcwd()
                os.chdir(self.root)
                try:
                    return smoke.main()
                finally:
                    os.chdir(old)

    def phrases_of(self, calls):
        return [c["input"][0]["content"][0]["text"] for c in calls]


class TestDryRunSafety(SmokeCase):
    def test_dry_run_makes_no_network_call(self):
        with mock.patch(
            "neev.adapters.base.UrllibTransport.post_json",
            ExplodingTransport().post_json,
        ):
            with mock.patch("sys.stdout"):
                code = self.run_main("--image", "s.jpg")
        self.assertEqual(code, 0)

    def test_dry_run_says_so_and_hides_secrets(self):
        fake_b64 = base64.b64encode(JPEG_BYTES).decode()
        with mock.patch(
            "neev.adapters.base.UrllibTransport.post_json",
            ExplodingTransport().post_json,
        ):
            import io

            out = io.StringIO()
            with mock.patch("sys.stdout", out):
                code = self.run_main("--image", "s.jpg")
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN", text)
        self.assertIn("base64,<", text)
        self.assertNotIn(fake_b64, text)
        self.assertNotIn("Bearer ey", text)

    def test_missing_key_blocks_live(self):
        with mock.patch(
            "neev.adapters.base.UrllibTransport.post_json",
            ExplodingTransport().post_json,
        ):
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                code = self.run_main("--image", "s.jpg", "--confirm-live")
        self.assertEqual(code, 2)

    def test_bad_images_fail_before_any_request(self):
        (self.root / "empty.jpg").write_bytes(b"")
        (self.root / "note.txt").write_text("not an image at all", encoding="utf-8")
        with mock.patch(
            "neev.adapters.base.UrllibTransport.post_json",
            ExplodingTransport().post_json,
        ):
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                self.assertEqual(self.run_main("--image", "nope.jpg"), 2)
                self.assertEqual(self.run_main("--image", "empty.jpg"), 2)
                self.assertEqual(self.run_main("--image", "note.txt"), 2)

    def test_mime_comes_from_bytes_not_extension(self):
        (self.root / "s.png").write_bytes(JPEG_BYTES)
        with mock.patch(
            "neev.adapters.base.UrllibTransport.post_json",
            ExplodingTransport().post_json,
        ):
            import io

            out = io.StringIO()
            with mock.patch("sys.stdout", out):
                self.run_main("--image", "s.png")
        self.assertIn("mime image/jpeg", out.getvalue())


class TestLivePath(SmokeCase):
    def run_live(self, transport):
        with mock.patch(
            "neev.adapters.base.UrllibTransport.post_json", transport.post_json
        ):
            import io

            out = io.StringIO()
            with mock.patch("sys.stdout", out):
                code = self.run_main(
                    "--image", "s.jpg", "--confirm-live",
                    "--runs-dir", str(self.root / "runs"),
                    env={"MODEL_API_KEY": "fake-key-xyz-123"},
                )
        return code, out.getvalue()

    def test_live_sends_exactly_two_requests(self):
        fake = FakeTransport({"output_text": OFFICIAL_LINE})
        code, _ = self.run_live(fake)
        self.assertEqual(code, 0)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(self.phrases_of(fake.calls), ["building", "window"])
        self.assertTrue(all(c["model"] == "sam-3.1" for c in fake.calls))

    def test_traces_hold_envelope_hashes_and_no_key(self):
        fake = FakeTransport(
            {"output_text": OFFICIAL_LINE, "usage": {"x": 1}, "id": "r1"}
        )
        code, _ = self.run_live(fake)
        self.assertEqual(code, 0)
        run_dirs = [p for p in (self.root / "runs").iterdir() if p.is_dir()]
        self.assertEqual(len(run_dirs), 2)
        for run_dir in run_dirs:
            request = json.loads((run_dir / "request.json").read_text())
            media = request["input"][0]["content"][1]["image_url"]
            self.assertIn("sha256", media)  # hash reference, not bytes
            response = json.loads((run_dir / "response.json").read_text())
            self.assertEqual(response["id"], "r1")  # full envelope kept
            blob = "".join(
                (run_dir / name).read_text()
                for name in ("request.json", "response.json", "meta.json")
            )
            self.assertNotIn("fake-key-xyz-123", blob)
            self.assertNotIn("Authorization", blob)

    def test_rerun_hits_cache_and_sends_nothing(self):
        fake = FakeTransport({"output_text": OFFICIAL_LINE})
        code, _ = self.run_live(fake)
        self.assertEqual(code, 0)
        self.assertEqual(len(fake.calls), 2)
        code, text = self.run_live(ExplodingTransport())
        self.assertEqual(code, 0)
        self.assertIn("CACHE HIT", text)

    def test_http200_garbage_is_a_failure(self):
        fake = FakeTransport({"output_text": "no tokens here at all"})
        code, text = self.run_live(fake)
        self.assertEqual(code, 1)
        self.assertIn("FAILED", text)

    def test_http200_without_output_text_is_a_failure(self):
        fake = FakeTransport({"weird": 1})
        code, text = self.run_live(fake)
        self.assertEqual(code, 1)
        self.assertIn("FAILED", text)


if __name__ == "__main__":
    unittest.main()
