"""Mask raster decoding via Meta's official parser package.

The one_bit/lossless codecs are context-adaptive range coders: porting them
by hand risks silent corruption, so decoding delegates to Meta's own
`meta-sam-parser` package (`pip install meta-sam-parser`). Neev validates
before and after: marker, positive dimensions, area cap, exact H*W length,
binary values. Any failure raises MaskError with the reason. Masks are
never fabricated: undecodable payloads stay undecodable.
"""

from __future__ import annotations

from typing import Callable


class MaskError(Exception):
    """A mask payload cannot be trusted: bad marker, dims, or decode."""


MAX_MASK_PIXELS = 64_000_000  # RAM guard; real masks are far smaller.

_MARKER_TO_ENCODING = {"!": "one_bit", "~": "lossless"}

# backend(encoding, payload, width, height) -> bytes-like raster, row-major.
Backend = Callable[[str, str, int, int], bytes]


def _official_backend(encoding: str, payload: str, width: int, height: int) -> bytes:
    try:
        from meta_sam_parser import SegmentationMask, decode_mask_to_raster
    except (ImportError, ModuleNotFoundError):
        try:
            import sys
            from pathlib import Path
            wheel = Path(__file__).resolve().parents[2] / "third_party" / "meta_sam_parser-0.1.0-py3-none-any.whl"
            if wheel.exists() and str(wheel) not in sys.path:
                sys.path.insert(0, str(wheel))
            from meta_sam_parser import SegmentationMask, decode_mask_to_raster
        except Exception as exc:
            raise MaskError(
                "meta-sam-parser is not installed; enable masks with: "
                "pip install meta-sam-parser"
            ) from exc
    try:
        return bytes(
            decode_mask_to_raster(
                SegmentationMask(
                    encoding=encoding, payload=payload, width=width, height=height
                )
            )
        )
    except Exception as exc:
        raise MaskError(f"official decoder rejected the mask: {exc}") from exc


def _is_clean_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def decode_mask(
    payload: str, width: int, height: int, backend: Backend | None = None
) -> bytes:
    """Decode one mask payload to H*W row-major bytes of 0/1.

    Raises MaskError for unknown markers, bad dimensions, oversized
    rasters, decoder rejection, short/long output, or non-binary values.
    """
    if not isinstance(payload, str) or not payload:
        raise MaskError("mask payload must be a non-empty string")
    encoding = _MARKER_TO_ENCODING.get(payload[0])
    if encoding is None:
        raise MaskError(f"unknown mask marker: {payload[0]!r} (want '!' or '~')")
    if not _is_clean_int(width) or not _is_clean_int(height):
        raise MaskError(f"mask dims must be ints, got {width!r} x {height!r}")
    if width <= 0 or height <= 0:
        raise MaskError(f"mask dims must be positive, got {width} x {height}")
    if width * height > MAX_MASK_PIXELS:
        raise MaskError(f"mask {width}x{height} exceeds {MAX_MASK_PIXELS} pixels")
    run = backend or _official_backend
    try:
        raster = bytes(run(encoding, payload, width, height))
    except MaskError:
        raise
    except Exception as exc:
        raise MaskError(f"mask backend failed: {exc}") from exc
    if len(raster) != width * height:
        raise MaskError(
            f"raster is {len(raster)} bytes for a {width}x{height} mask "
            f"(want {width * height})"
        )
    if any(b not in (0, 1) for b in raster):
        raise MaskError("raster is not binary (values outside 0/1)")
    return raster


def place_box_mask(raster: bytes, mask_w: int, mask_h: int,
                   box: dict[str, int], width: int, height: int) -> bytes:
    """Paste a box-local mask raster onto a full-frame 0/1 raster.

    Same convention as live/visualize.py: NEAREST-resize the small
    raster to the box, paste at the box origin. Box parts outside the
    frame clip defensively (the box itself is reported verbatim
    elsewhere, never moved). Raises MaskError on bad geometry.
    """
    from PIL import Image

    if len(raster) != mask_w * mask_h:
        raise MaskError(
            f"raster is {len(raster)} bytes for a {mask_w}x{mask_h} mask")
    try:
        x0, y0, x1, y1 = (box[k] for k in ("x0", "y0", "x1", "y1"))
    except (KeyError, TypeError) as exc:
        raise MaskError(f"bad box for mask placement: {exc}") from exc
    if not (x0 < x1 and y0 < y1):
        raise MaskError(f"degenerate box for mask placement: {box!r}")
    mask = Image.frombytes("L", (mask_w, mask_h),
                           bytes(b * 255 for b in raster))
    bw, bh = x1 - x0, y1 - y0
    mask = mask.resize((bw, bh), Image.NEAREST)
    frame = Image.new("L", (width, height), 0)
    src = mask.crop((max(0, -x0), max(0, -y0),
                     bw - max(0, x0 + bw - width),
                     bh - max(0, y0 + bh - height)))
    frame.paste(src, (max(0, x0), max(0, y0)))
    return bytes(1 if v else 0 for v in frame.tobytes())
