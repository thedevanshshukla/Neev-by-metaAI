"""View rasterization for multi-view execution (Pillow, local only).

Cuts tile/reinspect views from source bytes. Tiles encode as PNG with
Pillow's default settings -- the same call the tiling pilot used
(``crop.save(path)``) -- so byte-identical re-cuts of the same pixels
hit the same cache keys. No network, no inference.
"""

from __future__ import annotations

import io
from typing import Any


def cut_view(raw: bytes, box: dict[str, int]) -> bytes:
    """Crop ``box`` (source coords) from image bytes. Returns PNG bytes."""
    from PIL import Image

    image = Image.open(io.BytesIO(raw))
    crop = image.crop((box["x0"], box["y0"], box["x1"], box["y1"]))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return buf.getvalue()


def view_bytes_map(raw: bytes, views: list[Any]) -> dict[str, bytes]:
    """Cut every non-whole view once. Returns {view_id: png bytes}."""
    out: dict[str, bytes] = {}
    for view in views:
        if view.kind == "whole":
            continue
        out[view.view_id] = cut_view(raw, view.box)
    return out
