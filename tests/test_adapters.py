"""Unit tests: adapter builders and parsers with a fake transport.

No network access: the fake transport returns canned payloads, and the
tests assert request shapes and parsing only. Token fixtures follow the
official grammar (ai.developer.meta.com/docs/sam/reading-segmentation).
"""

import io
import unittest
import urllib.error
from unittest import mock

from neev.adapters.base import (
    AdapterError,
    UrllibTransport,
    extract_output_text,
)
from neev.adapters.sam31 import (
    SAM31_PROMPTS,
    build_image_request,
    build_video_request,
    parse_output_text,
    sanitized_request,
    sniff_image_mime,
)
from neev.adapters.wonder import build_judge_request, extract_json


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post_json(self, url, headers, payload, timeout=0):
        self.calls.append((url, payload))
        return self.response


# Official line shape with deliberately SPARSE ids (0, 2): ids are stable
# object ids, not dense positions. Payloads carry real markers (~, !).
SAMPLE_TOKENS = (
    "<0f>0<|box;x1=353;y1=382;x2=453;y2=427;w=537;h=561|>"
    "<|mask;x=0;y=0;data=46,101,~PAYLOAD1|>"
    ",2<|box;x1=516;y1=117;x2=536;y2=150;w=537;h=561|>"
    "<|mask;x=0;y=0;data=34,21,!PAYLOAD2|>"
)


class TestSamRequests(unittest.TestCase):
    def test_image_request_shape(self):
        req = build_image_request("building", "QUJD", stream=False)
        self.assertEqual(req["model"], "sam-3.1")
        self.assertEqual(req["metadata"], {"mask_encoding": "one_bit"})
        content = req["input"][0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "building"})
        self.assertEqual(content[1]["type"], "input_image")
        self.assertTrue(content[1]["image_url"].startswith("data:image/jpeg;base64,"))

    def test_video_request_uses_input_video(self):
        req = build_video_request("building", "QUJD")
        content = req["input"][0]["content"]
        self.assertEqual(content[1]["type"], "input_video")
        self.assertTrue(content[1]["video_url"].startswith("data:video/mp4;base64,"))
        self.assertTrue(req["stream"])

    def test_prompt_registry_covers_docs_passes(self):
        self.assertEqual(
            set(SAM31_PROMPTS),
            {
                "buildings",
                "ground_floor",
                "storeys",
                "hazards",
                "condition",
                "privacy",
            },
        )
        self.assertIn("open ground floor", SAM31_PROMPTS["ground_floor"])


class TestSamParsing(unittest.TestCase):
    def test_parses_official_line_with_sparse_ids(self):
        dets = parse_output_text(SAMPLE_TOKENS, "building")
        self.assertEqual(len(dets), 2)
        first, second = dets
        # Explicit ids are read, never positionally assigned: 0 and 2.
        self.assertEqual((first.track_id, first.frame_index), ("0", 0))
        self.assertEqual((second.track_id, second.frame_index), ("2", 0))
        self.assertEqual(
            (first.box.x0, first.box.y0, first.box.x1, first.box.y1),
            (353, 382, 453, 427),
        )
        self.assertEqual((first.frame_width, first.frame_height), (537, 561))
        self.assertEqual(first.mask_payload, "~PAYLOAD1")
        self.assertEqual((first.mask_width, first.mask_height), (101, 46))
        self.assertEqual(first.mask_encoding, "lossless")
        self.assertEqual(second.mask_encoding, "one_bit")
        self.assertIsNone(first.confidence)

    def test_confidence_tokens_parse(self):
        line = (
            "<0f>1<|box;x1=444;y1=625;x2=679;y2=850;w=1800;h=1200;c=0.9609375|>"
            "<|mask;x=0;y=0;c=0.9609375;data=226,236,!P|>"
        )
        (det,) = parse_output_text(line, "bus")
        self.assertEqual(det.track_id, "1")
        self.assertAlmostEqual(det.confidence, 0.9609375)
        self.assertEqual((det.mask_width, det.mask_height), (236, 226))

    def test_multiline_video_reads_explicit_frames(self):
        line = (
            "<0f>0<|box;x1=1;y1=1;x2=2;y2=2;w=9;h=9|>\n"
            "<2f>0<|box;x1=3;y1=3;x2=4;y2=4;w=9;h=9|>"
        )
        dets = parse_output_text(line, "car")
        self.assertEqual([d.frame_index for d in dets], [0, 2])

    def test_empty_output_is_zero_detections(self):
        self.assertEqual(parse_output_text("", "building"), [])
        self.assertEqual(parse_output_text("   ", "building"), [])

    def test_garbage_output_raises_not_empty(self):
        # Old expectation (garbage -> []) was wrong: it let API drift
        # masquerade as "nothing found". Non-blank text with no record
        # is a parse failure; only blank text means zero matches.
        with self.assertRaises(AdapterError):
            parse_output_text("no boxes here", "building")


class TestOutputText(unittest.TestCase):
    def test_prefers_top_level_field(self):
        self.assertEqual(extract_output_text({"output_text": "hi"}), "hi")

    def test_reads_nested_message_shape(self):
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "a"}],
                }
            ]
        }
        self.assertEqual(extract_output_text(response), "a")

    def test_missing_output_lists_keys(self):
        with self.assertRaises(AdapterError) as ctx:
            extract_output_text({"weird": 1})
        self.assertIn("weird", str(ctx.exception))


class TestWonder(unittest.TestCase):
    def test_judge_request_orders_features_then_images(self):
        req = build_judge_request({"a": 1}, ["I1", "I2"], "SYS")
        self.assertEqual(req["model"], "muse-wonder-1.0")
        self.assertEqual(req["instructions"], "SYS")
        content = req["input"][0]["content"]
        self.assertEqual(content[0]["type"], "input_text")
        self.assertIn('"a": 1', content[0]["text"])
        self.assertEqual(len(content), 3)
        self.assertTrue(all(c["type"] == "input_image" for c in content[1:]))

    def test_extract_json_tolerates_wrapper_text(self):
        self.assertEqual(extract_json('prefix {"x": 2} suffix'), {"x": 2})

    def test_extract_json_rejects_non_dict(self):
        with self.assertRaises(AdapterError):
            extract_json("[1, 2]")

    def test_extract_json_rejects_prose(self):
        with self.assertRaises(AdapterError):
            extract_json("no json at all")


class TestTransportSafety(unittest.TestCase):
    def test_http_error_scrubs_secret_header_echo(self):
        secret = "Bearer fake-secret-abcdef"
        body = f"bad request, your auth {secret} was weird".encode()
        error = urllib.error.HTTPError(
            "https://x.test/v1/responses", 500, "oops", {}, io.BytesIO(body)
        )
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(AdapterError) as ctx:
                UrllibTransport().post_json(
                    "https://x.test/v1/responses",
                    {"Authorization": secret},
                    {"a": 1},
                )
        message = str(ctx.exception)
        self.assertNotIn("fake-secret-abcdef", message)
        self.assertIn("[REDACTED]", message)
        self.assertIn("HTTP 500", message)

    def test_http_error_keeps_ordinary_detail(self):
        body = b"quota exhausted for team"
        error = urllib.error.HTTPError(
            "https://x.test/v1/responses", 429, "slow", {}, io.BytesIO(body)
        )
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(AdapterError) as ctx:
                UrllibTransport().post_json("https://x.test/v1/responses", {}, {})
        self.assertIn("quota exhausted", str(ctx.exception))

    def test_url_error_names_url(self):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("timed out"),
        ):
            with self.assertRaises(AdapterError) as ctx:
                UrllibTransport().post_json("https://x.test/v1/responses", {}, {})
        self.assertIn("https://x.test/v1/responses", str(ctx.exception))


class TestSniffMime(unittest.TestCase):
    def test_known_magics(self):
        self.assertEqual(sniff_image_mime(b"\xff\xd8\xffrest"), "image/jpeg")
        self.assertEqual(
            sniff_image_mime(b"\x89PNG\r\n\x1a\nrest"), "image/png"
        )
        self.assertEqual(sniff_image_mime(b"GIF87arest"), "image/gif")
        self.assertEqual(sniff_image_mime(b"GIF89arest"), "image/gif")
        self.assertEqual(sniff_image_mime(b"BMrest"), "image/bmp")
        self.assertEqual(
            sniff_image_mime(b"RIFF\x00\x00\x00\x00WEBP"), "image/webp"
        )

    def test_unknown_is_none(self):
        self.assertIsNone(sniff_image_mime(b"hello, this is text"))
        self.assertIsNone(sniff_image_mime(b""))


class TestSanitizedRequest(unittest.TestCase):
    def test_media_becomes_hash_reference(self):
        request = build_image_request("building", "QUJDQUJD")
        shown = sanitized_request(request, "abc123")
        media = shown["input"][0]["content"][1]["image_url"]
        self.assertEqual(media["sha256"], "abc123")
        self.assertNotIn("QUJDQUJD", str(shown))
        # Phrase and model survive; the original is untouched.
        self.assertEqual(shown["input"][0]["content"][0]["text"], "building")
        self.assertEqual(shown["model"], "sam-3.1")
        self.assertIn("QUJDQUJD", request["input"][0]["content"][1]["image_url"])


if __name__ == "__main__":
    unittest.main()
