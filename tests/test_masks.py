"""Unit tests: mask decoding wrapper. No network, no real payloads.

The official codec runs behind an injectable backend, so these tests
cover Neev's validation (markers, dimensions, output shape) with fakes
and never fabricate a successful decode of a real mask.
"""

import unittest

from neev.measurements.masks import MAX_MASK_PIXELS, MaskError, decode_mask


def fake_backend(raster):
    def run(encoding, payload, width, height):
        assert encoding in ("one_bit", "lossless")
        return raster

    return run


class TestDecodeMask(unittest.TestCase):
    def test_valid_raster_passes_through(self):
        raster = decode_mask("!xyz", 2, 2, backend=fake_backend(b"\x00\x01\x01\x00"))
        self.assertEqual(raster, b"\x00\x01\x01\x00")

    def test_lossless_marker_accepted(self):
        seen = {}

        def run(encoding, payload, width, height):
            seen["encoding"] = encoding
            return b"\x01"

        decode_mask("~xyz", 1, 1, backend=run)
        self.assertEqual(seen["encoding"], "lossless")

    def test_unknown_marker_rejected(self):
        with self.assertRaises(MaskError):
            decode_mask("?xyz", 1, 1, backend=fake_backend(b"\x01"))

    def test_empty_payload_rejected(self):
        with self.assertRaises(MaskError):
            decode_mask("", 1, 1, backend=fake_backend(b"\x01"))

    def test_bad_dimensions_rejected(self):
        for width, height in [(0, 4), (4, -1), (4.0, 4), (True, 4), ("4", 4)]:
            with self.assertRaises(MaskError, msg=f"{width!r}x{height!r}"):
                decode_mask("!x", width, height, backend=fake_backend(b"\x01"))

    def test_oversized_mask_rejected_without_decoding(self):
        def boom(encoding, payload, width, height):
            raise AssertionError("backend must not run")

        with self.assertRaises(MaskError):
            decode_mask("!x", MAX_MASK_PIXELS, 2, backend=boom)

    def test_short_raster_rejected(self):
        with self.assertRaises(MaskError):
            decode_mask("!x", 2, 2, backend=fake_backend(b"\x01\x01"))

    def test_non_binary_raster_rejected(self):
        with self.assertRaises(MaskError):
            decode_mask("!x", 2, 1, backend=fake_backend(b"\x01\x02"))

    def test_backend_failure_wrapped(self):
        def fail(encoding, payload, width, height):
            raise ValueError("codec exploded")

        with self.assertRaises(MaskError) as ctx:
            decode_mask("!x", 1, 1, backend=fail)
        self.assertIn("codec exploded", str(ctx.exception))

    def test_missing_package_is_a_clear_mask_error(self):
        import sys
        from unittest import mock

        with mock.patch.dict(sys.modules, {"meta_sam_parser": None}):
            with self.assertRaises(MaskError) as ctx:
                decode_mask("!x", 1, 1)
        self.assertIn("meta-sam-parser is not installed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
