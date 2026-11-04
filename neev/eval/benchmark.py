"""Local benchmark format v1: manual ground truth for SAM prompt eval.

A benchmark document names images (by content hash, so renames don't
break it) and records, per category, whether the target is present plus
hand-drawn boxes and optional run-length masks:

{
  "version": 1,
  "name": "neev-sam31-v1",
  "coverage": {"crack": "insufficient-labels", ...},
  "images": [
    {"image_id": "villa-01", "path": "photo_Sample/images.jpeg",
     "sha256": "<64 hex>", "width": 447, "height": 447,
     "labels": {
       "building": {"present": true,
                    "boxes": [{"x0": 43, "y0": 27, "x1": 393, "y1": 321}],
                    "masks": []},
       "crack": {"present": null, "boxes": [], "masks": []}
     }}
  ]
}

Presence rules (missing labels stay missing, never negatives):
- category key absent, or present null  -> UNLABELLED, excluded everywhere.
- present false                    -> negative; boxes/masks must be empty.
- present true, boxes empty        -> presence-only; image-level verdict only.
- present true, boxes non-empty    -> full box-level evaluation.
- masks: COCO-style {"width","height","counts"} starting with zeros.

Boxes are source pixels, x1/y1 exclusive, inside the image. Nothing in
this file is ever guessed: an unlabelled image contributes no positives
and no negatives to any metric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BENCHMARK_VERSION = 1

# Canonical measurable categories. Privacy prompts (face, license plate)
# are deliberately absent: they are evaluated separately, if ever.
# "spalling" (missing masonry/render) is separate from "crack" (thin
# traces): the two must never share boxes.
CATEGORIES = (
    "building",
    "car",
    "open_ground_floor",
    "shutter",
    "window",
    "column",
    "balcony",
    "floor_slab",
    "rooftop_unit",
    "hoarding",
    "parapet",
    "crack",
    "spalling",
    "rebar",
)


def _is_clean_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_box(box: Any, width: int, height: int, where: str) -> list[str]:
    """Ground-truth boxes must be well-formed and inside the image."""
    errors: list[str] = []
    if not isinstance(box, dict):
        return [f"{where}: box must be an object"]
    coords = [box.get(k) for k in ("x0", "y0", "x1", "y1")]
    if not all(_is_clean_int(c) for c in coords):
        return [f"{where}: box coords must be ints: {box!r}"]
    x0, y0, x1, y1 = coords
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        errors.append(f"{where}: box {box!r} outside {width}x{height}")
    return errors


def validate_rle(mask: Any, where: str) -> list[str]:
    """COCO-style counts: non-negative ints summing to exactly W*H."""
    errors: list[str] = []
    if not isinstance(mask, dict):
        return [f"{where}: mask must be an object"]
    width, height, counts = mask.get("width"), mask.get("height"), mask.get("counts")
    if not _is_clean_int(width) or not _is_clean_int(height) or width <= 0 or height <= 0:
        return [f"{where}: mask needs positive int width/height"]
    if not isinstance(counts, list) or not counts:
        return [f"{where}: mask counts must be a non-empty list"]
    if not all(_is_clean_int(c) and c >= 0 for c in counts):
        return [f"{where}: mask counts must be non-negative ints"]
    if sum(counts) != width * height:
        errors.append(
            f"{where}: mask counts sum {sum(counts)}, want {width * height}"
        )
    return errors


def validate_label(label: Any, width: int, height: int, where: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(label, dict):
        return [f"{where}: label must be an object"]
    present = label.get("present")
    if present not in (True, False, None):
        return [f"{where}: present must be true/false/null"]
    boxes = label.get("boxes", [])
    masks = label.get("masks", [])
    if not isinstance(boxes, list) or not isinstance(masks, list):
        return [f"{where}: boxes/masks must be lists"]
    if present is not True and (boxes or masks):
        errors.append(f"{where}: unlabelled/negative entries must not hold boxes")
    for index, box in enumerate(boxes):
        errors.extend(validate_box(box, width, height, f"{where}.boxes[{index}]"))
    for index, mask in enumerate(masks):
        errors.extend(validate_rle(mask, f"{where}.masks[{index}]"))
    return errors


def validate_image_entry(entry: Any, seen_ids: set[str]) -> list[str]:
    errors: list[str] = []
    if not isinstance(entry, dict):
        return ["image entries must be objects"]
    image_id = entry.get("image_id")
    if not image_id or not isinstance(image_id, str):
        errors.append("image entry misses string image_id")
    elif image_id in seen_ids:
        errors.append(f"duplicate image_id: {image_id!r}")
    seen_ids.add(image_id)
    where = f"image {image_id!r}"
    width, height = entry.get("width"), entry.get("height")
    if not _is_clean_int(width) or not _is_clean_int(height) or width <= 0 or height <= 0:
        errors.append(f"{where}: needs positive int width/height")
        width, height = 0, 0
    sha = entry.get("sha256", "")
    if not isinstance(sha, str) or len(sha) != 64:
        errors.append(f"{where}: sha256 must be 64 hex chars")
    labels = entry.get("labels", {})
    if not isinstance(labels, dict):
        return errors + [f"{where}: labels must be an object"]
    for category, label in labels.items():
        if category not in CATEGORIES:
            errors.append(f"{where}: unknown category {category!r}")
            continue
        errors.extend(validate_label(label, width, height, f"{where}.{category}"))
    return errors


def validate_benchmark(doc: Any) -> list[str]:
    """Return error strings for a benchmark document (empty means valid)."""
    if not isinstance(doc, dict):
        return ["benchmark must be an object"]
    errors: list[str] = []
    if doc.get("version") != BENCHMARK_VERSION:
        errors.append(f"version must be {BENCHMARK_VERSION}")
    images = doc.get("images")
    if not isinstance(images, list) or not images:
        errors.append("images must be a non-empty list")
        return errors
    seen: set[str] = set()
    for entry in images:
        errors.extend(validate_image_entry(entry, seen))
    return errors


def load_benchmark(path: str | Path) -> dict[str, Any]:
    """Load and validate a benchmark file. ValueError lists every problem."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_benchmark(doc)
    if errors:
        raise ValueError("invalid benchmark:\n- " + "\n- ".join(errors))
    return doc


def label_status(labels: dict, category: str) -> str:
    """One of: unlabelled | negative | presence_only | labelled."""
    label = labels.get(category)
    if not isinstance(label, dict) or label.get("present") is None:
        return "unlabelled"
    if label.get("present") is False:
        return "negative"
    if label.get("boxes"):
        return "labelled"
    return "presence_only"
