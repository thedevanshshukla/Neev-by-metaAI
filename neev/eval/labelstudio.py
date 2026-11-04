"""Native Label Studio export parsing (fidelity-first).

Rotation convention, per the official Label Studio Object Detection
template docs: the saved value.rotation is CLOCKWISE degrees around
(x, y) — the rectangle's top-left corner BEFORE rotation — while x, y,
width, height are percentages of the original image. Whole-image
rotation (image_rotation) must be 0; anything else holds the image out
of evaluation rather than being reinterpreted.

Nothing here modifies the export. Original percentages, rotation, task
ids, and result ids are preserved verbatim on every record, and the
validation report names everything unusual instead of hiding it:
- tiny floating-point edge noise (|v| < 1e-9 outside [0, 100]) is
  snapped to the edge and RECORDED, never silently rounded;
- genuinely out-of-bounds geometry is kept verbatim and CLIPPED only
  for IoU math (nothing exists outside the sensor frame), recorded per
  annotation;
- multi-label results expand to one entry per label (shared geometry);
- images that cannot be mapped unambiguously are held out with errors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FP_SNAP_TOL = 1e-9


@dataclass
class LSRect:
    """One rectangle result, originals preserved verbatim."""

    task_id: Any
    result_id: str
    label: str
    x_pct: float
    y_pct: float
    w_pct: float
    h_pct: float
    rotation_deg: float
    orig_width: int
    orig_height: int
    image_relpath: str
    image_sha256: str
    source: str
    snapped: bool = False
    clipped: bool = False


@dataclass
class LSImage:
    task_id: Any
    image_id: str
    relpath: str
    sha256: str
    width: int
    height: int
    source: str
    is_negative: bool
    rects: list[LSRect] = field(default_factory=list)
    duplicate_paths: list[str] = field(default_factory=list)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return _is_num(value) and value not in (float("inf"), float("-inf")) and value == value


def resolve_image_path(
    task: dict, dataset_root: str | Path
) -> tuple[str | None, str | None, list[str]]:
    """Map a task to its dataset file, or an error (never a guess).

    Matches on the upload basename (with or without Label Studio's
    "<8hex>-" prefix) and requires the data.image reference to agree.
    Several matches with IDENTICAL bytes (e.g. a backup copy) resolve
    deterministically to the first sorted path and are REPORTED as
    duplicates; matches with different bytes hold the image out.
    Returns (relpath, error, duplicate_paths).
    """
    root = Path(dataset_root)
    data_ref = str((task.get("data") or {}).get("image") or "")
    upload = str(task.get("file_upload") or "")
    data_base = data_ref.rsplit("/", 1)[-1]
    if not upload or data_base != upload:
        return None, f"task {task.get('id')}: data.image {data_base!r} disagrees with file_upload {upload!r}", []
    candidates = [upload]
    if "-" in upload:
        candidates.append(upload.split("-", 1)[1])
    hits: list[str] = []
    for candidate in candidates:
        for path in root.rglob(candidate):
            if path.is_file():
                hits.append(str(path.relative_to(root)))
    unique = sorted(set(hits))
    if len(unique) == 1:
        return unique[0], None, []
    if not unique:
        return None, f"task {task.get('id')}: {upload!r} matches 0 files", []
    contents = {h: hashlib.sha256((root / h).read_bytes()).hexdigest() for h in unique}
    if len(set(contents.values())) == 1:
        # Same bytes everywhere: prefer the canonical real_world tree so
        # source typing stays correct. Deterministic; rest is reported.
        ordered = sorted(unique, key=lambda h: (
            Path(h).parts[:1] != ("real_world",), h))
        return ordered[0], None, [h for h in ordered if h != ordered[0]]
    return None, f"task {task.get('id')}: {upload!r} matches {len(unique)} DIFFERENT files {unique}", []


def ls_rotation_to_corners(
    x_px: float, y_px: float, w_px: float, h_px: float, rotation_deg: float
) -> list[tuple[float, float]]:
    """Unrotated rect + clockwise degrees about top-left -> 4 corners.

    Visual-clockwise in y-down image coords: (dx, dy) maps to
    (dx*cos - dy*sin, dx*sin + dy*cos). rotation=0 is the identity.
    """
    import math

    theta = math.radians(rotation_deg % 360.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    corners = []
    for dx, dy in ((0.0, 0.0), (w_px, 0.0), (w_px, h_px), (0.0, h_px)):
        corners.append((x_px + dx * cos_t - dy * sin_t,
                        y_px + dx * sin_t + dy * cos_t))
    return corners


def _snap(value: float) -> tuple[float, bool]:
    if -FP_SNAP_TOL < value < 0:
        return 0.0, True
    if 100 < value < 100 + FP_SNAP_TOL:
        return 100.0, True
    return value, False


def parse_task(task: dict, dataset_root: str | Path) -> tuple[LSImage | None, list[str]]:
    """Parse one task. Errors hold the image out; nothing is discarded."""
    errors: list[str] = []
    task_id = task.get("id")
    relpath, mapping_error, duplicates = resolve_image_path(task, dataset_root)
    if mapping_error:
        return None, [mapping_error]
    annotations = task.get("annotations") or []
    if len(annotations) != 1:
        return None, [f"task {task_id}: want exactly 1 annotation, got {len(annotations)}"]
    results = annotations[0].get("result") or []
    dims = {(r.get("original_width"), r.get("original_height")) for r in results}
    dims.discard((None, None))
    if len(dims) > 1:
        return None, [f"task {task_id}: inconsistent original dims {sorted(dims)}"]
    orig_w, orig_h = next(iter(dims)) if dims else (None, None)
    if results and (
        not isinstance(orig_w, int) or not isinstance(orig_h, int)
        or orig_w <= 0 or orig_h <= 0
    ):
        return None, [f"task {task_id}: invalid original dims {(orig_w, orig_h)}"]

    root = Path(dataset_root)
    image_id = Path(relpath).stem
    source = "real_world" if Path(relpath).parts[:1] == ("real_world",) else "unknown"
    sha = hashlib.sha256((root / relpath).read_bytes()).hexdigest()
    rects: list[LSRect] = []
    for result in results:
        rid = result.get("id", "?")
        where = f"task {task_id} result {rid}"
        if (result.get("type"), result.get("from_name"), result.get("to_name")) != (
            "rectanglelabels", "label", "image"
        ):
            errors.append(f"{where}: not a rectangle label; held out, never skipped")
            continue
        if result.get("image_rotation", 0) != 0:
            errors.append(f"{where}: image_rotation is {result.get('image_rotation')}; held out")
            continue
        value = result.get("value") or {}
        labels = value.get("rectanglelabels") or []
        if not labels:
            errors.append(f"{where}: box without a label; held out")
            continue
        coords = [value.get(k) for k in ("x", "y", "width", "height")]
        rotation = value.get("rotation", 0)
        if not all(_finite(c) for c in coords) or not _finite(rotation):
            errors.append(f"{where}: non-finite coordinates or rotation; held out")
            continue
        x, y, w, h = (float(c) for c in coords)
        if w <= 0 or h <= 0:
            errors.append(f"{where}: non-positive size {(w, h)}; held out")
            continue
        snapped_any = False
        x, s = _snap(x); snapped_any |= s
        y, s = _snap(y); snapped_any |= s
        w_edge, s = _snap(x + w); snapped_any |= s
        h_edge, s = _snap(y + h); snapped_any |= s
        if x < 0 or y < 0 or w_edge > 100 or h_edge > 100:
            # Genuine overflow: keep verbatim, clip only for geometry.
            clipped = True
        else:
            clipped = False
            w, h = w_edge - x, h_edge - y
        for label in labels:
            rects.append(LSRect(
                task_id=task_id, result_id=str(rid), label=str(label),
                x_pct=x, y_pct=y, w_pct=w, h_pct=h,
                rotation_deg=float(rotation), orig_width=orig_w, orig_height=orig_h,
                image_relpath=relpath, image_sha256=sha, source=source,
                snapped=snapped_any, clipped=clipped,
            ))
    if errors:
        return None, errors
    return LSImage(
        task_id=task_id, image_id=image_id, relpath=relpath, sha256=sha,
        width=orig_w or 0, height=orig_h or 0, source=source,
        is_negative=not results, rects=rects, duplicate_paths=duplicates,
    ), []


def rect_polygon(rect: LSRect, img_w: int, img_h: int) -> list[tuple[float, float]]:
    """Percent geometry -> pixel corners (unclipped; originals intact)."""
    return ls_rotation_to_corners(
        rect.x_pct / 100 * img_w, rect.y_pct / 100 * img_h,
        rect.w_pct / 100 * img_w, rect.h_pct / 100 * img_h,
        rect.rotation_deg,
    )


def validate_export(
    tasks: list[dict], dataset_root: str | Path, check_dims: bool = True
) -> dict[str, Any]:
    """Full validation report over the native export (read-only)."""
    images: list[LSImage] = []
    all_errors: list[str] = []
    for task in tasks:
        image, errors = parse_task(task, dataset_root)
        if errors:
            all_errors.extend(errors)
        elif image is not None:
            images.append(image)
    per_class: dict[str, int] = {}
    rotated = snapped = clipped = 0
    for image in images:
        for rect in image.rects:
            per_class[rect.label] = per_class.get(rect.label, 0) + 1
            rotated += bool(rect.rotation_deg)
            snapped += rect.snapped
            clipped += rect.clipped
    dim_mismatches: list[str] = []
    if check_dims:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("dim check needs Pillow: pip install pillow") from exc
        root = Path(dataset_root)
        for image in images:
            if image.is_negative:
                continue
            actual = Image.open(root / image.relpath).size
            if actual != (image.width, image.height):
                dim_mismatches.append(
                    f"task {image.task_id}: export {(image.width, image.height)} vs file {actual}")
    stems = [i.image_id for i in images]
    dupes = {i.task_id: i.duplicate_paths for i in images if i.duplicate_paths}
    return {
        "tasks": len(tasks),
        "images_parsed": len(images),
        "byte_identical_duplicates": dupes,
        "total_annotations": sum(len(i.rects) for i in images),
        "per_class": dict(sorted(per_class.items())),
        "empty_negative_images": sorted(i.task_id for i in images if i.is_negative),
        "rotated_boxes": rotated,
        "snapped_fp_noise": snapped,
        "clipped_overflow": clipped,
        "sources": {s: sum(1 for i in images if i.source == s) for s in {i.source for i in images}},
        "duplicate_image_ids": sorted({s for s in stems if stems.count(s) > 1}),
        "dim_mismatches": dim_mismatches,
        "errors": all_errors,
    }


def load_export(path: str | Path) -> list[dict]:
    """Read the native export. Never writes; the file stays authoritative."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, list):
        raise ValueError("Label Studio export must be a JSON list of tasks")
    return doc
