"""Crop experiment configs: compare full images against manual crops.

Crops are NEVER automatic: each one is hand-defined with a purpose, and
only approved crops materialize by default. A crop is a new API input
when it is swept later, so every crop carries an id, source box, and
provenance. Nothing here calls the API; materialize() only writes local
files with Pillow.

Config:
{
  "version": 1,
  "crops": [{"crop_id": "img2-stain-band", "image": "photo_Sample/img2.jpeg",
             "box": {"x0": 40, "y0": 60, "x1": 400, "y1": 300},
             "purpose": "crack recall on stained render, close-up",
             "status": "provisional"}]
}

status provisional = drawn but not yet eyeballed on the source image;
approved = someone verified the box frames the intended region.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CROPS_VERSION = 1
STATUSES = ("provisional", "approved")


def validate_crop_box(box: Any, width: int, height: int) -> list[str]:
    """Pure bounds check: inside the image, non-empty, integer coords."""
    if not isinstance(box, dict):
        return ["crop box must be an object"]
    coords = [box.get(k) for k in ("x0", "y0", "x1", "y1")]
    if not all(isinstance(c, int) and not isinstance(c, bool) for c in coords):
        return [f"crop box coords must be ints: {box!r}"]
    x0, y0, x1, y1 = coords
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        return [f"crop box {box!r} outside {width}x{height}"]
    return []


def validate_crops_config(doc: Any) -> list[str]:
    """Structural validation (bounds need image dims; see materialize)."""
    if not isinstance(doc, dict):
        return ["crops config must be an object"]
    errors: list[str] = []
    if doc.get("version") != CROPS_VERSION:
        errors.append(f"version must be {CROPS_VERSION}")
    crops = doc.get("crops")
    if not isinstance(crops, list):
        return errors + ["crops must be a list"]
    seen: set[str] = set()
    for crop in crops:
        if not isinstance(crop, dict):
            errors.append("crop entries must be objects")
            continue
        crop_id = crop.get("crop_id")
        if not crop_id or not isinstance(crop_id, str):
            errors.append("crop misses string crop_id")
        elif crop_id in seen:
            errors.append(f"duplicate crop_id: {crop_id!r}")
        seen.add(crop_id)
        if not crop.get("image") or not isinstance(crop.get("image"), str):
            errors.append(f"{crop_id}: misses image path")
        if crop.get("status") not in STATUSES:
            errors.append(f"{crop_id}: status must be provisional/approved")
        if not crop.get("purpose"):
            errors.append(f"{crop_id}: misses purpose")
        box = crop.get("box")
        if not isinstance(box, dict):
            errors.append(f"{crop_id}: misses box object")
    return errors


def load_crops_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a crops config. ValueError lists every problem."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_crops_config(doc)
    if errors:
        raise ValueError("invalid crops config:\n- " + "\n- ".join(errors))
    return doc


def materialize(
    config: dict[str, Any],
    src_root: str | Path,
    out_dir: str | Path,
    include_provisional: bool = False,
) -> list[dict[str, Any]]:
    """Write approved (only) crops to out_dir. Returns a crop manifest.

    Each crop box is re-checked against the real image dimensions; an
    out-of-bounds box raises instead of silently clipping.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("materialize needs Pillow: pip install pillow") from exc
    src_root, out_dir = Path(src_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for crop in config.get("crops", []):
        if crop["status"] != "approved" and not include_provisional:
            continue
        source = src_root / crop["image"]
        image = Image.open(source).convert("RGB")
        box = crop["box"]
        errors = validate_crop_box(box, *image.size)
        if errors:
            raise ValueError(f"{crop['crop_id']}: {errors[0]}")
        cut = image.crop((box["x0"], box["y0"], box["x1"], box["y1"]))
        target = out_dir / f"{crop['crop_id']}.jpg"
        cut.save(target, quality=92)
        raw = target.read_bytes()
        manifest.append({
            "crop_id": crop["crop_id"], "path": str(target),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "source": str(crop["image"]), "box": dict(box),
            "size": list(cut.size), "status": crop["status"],
        })
    return manifest
