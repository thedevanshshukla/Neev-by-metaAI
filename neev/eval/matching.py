"""IoU matching and precision/recall/F1 (pure, deterministic).

Greedy best-first matching on box IoU: each ground-truth box matches at
most one prediction, so duplicate detections on one object count as
false positives. Malformed predictions (inverted or zero-area boxes)
are false positives too, listed separately so model faults stay visible
instead of crashing the run. Everything sorts its inputs, so two runs
over the same data produce byte-identical results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neev.adapters.base import BoundingBox
from neev.prep.tracking import iou


def _as_bbox(box: dict[str, Any]) -> BoundingBox | None:
    """BoundingBox for well-formed dicts, else None (malformed)."""
    try:
        x0, y0, x1, y1 = (box[k] for k in ("x0", "y0", "x1", "y1"))
    except (KeyError, TypeError):
        return None
    if (
        not all(isinstance(v, int) and not isinstance(v, bool) for v in (x0, y0, x1, y1))
        or not (x0 < x1 and y0 < y1)
    ):
        return None
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


def rle_decode(counts: list[int], width: int, height: int) -> bytes:
    """COCO-style counts (zeros first) to row-major 0/1 bytes."""
    if sum(counts) != width * height:
        raise ValueError(f"RLE sums to {sum(counts)}, want {width * height}")
    raster = bytearray()
    value = 0
    for run in counts:
        raster.extend([value] * run)
        value = 1 - value
    return bytes(raster)


def mask_iou(a: bytes, b: bytes) -> float:
    """Intersection over union of two equal-length binary rasters."""
    if len(a) != len(b):
        raise ValueError("mask rasters differ in length")
    inter = sum(1 for x, y in zip(a, b) if x and y)
    union = sum(1 for x, y in zip(a, b) if x or y)
    return inter / union if union else 1.0


@dataclass(frozen=True)
class MatchResult:
    tp: int
    fp: int
    fn: int
    matches: list[tuple[int, int, float]] = field(default_factory=list)
    malformed_pred_indices: list[int] = field(default_factory=list)


def match_predictions(
    pred_boxes: list[dict[str, Any]],
    gt_boxes: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Greedy IoU matching. Duplicates and malformed preds are FPs."""
    parsed = [(_as_bbox(box), index) for index, box in enumerate(pred_boxes)]
    malformed = sorted(index for parsed_box, index in parsed if parsed_box is None)
    gt_parsed = [_as_bbox(box) for box in gt_boxes]
    if any(box is None for box in gt_parsed):
        raise ValueError("ground-truth boxes must be well-formed")
    scored: list[tuple[float, int, int]] = []
    for parsed_box, pred_index in parsed:
        if parsed_box is None:
            continue
        for gt_index, gt_box in enumerate(gt_parsed):
            score = iou(parsed_box, gt_box)
            if score >= iou_threshold:
                scored.append((score, pred_index, gt_index))
    scored.sort(key=lambda triple: (-triple[0], triple[1], triple[2]))
    used_preds, used_gts, matches = set(), set(), []
    for score, pred_index, gt_index in scored:
        if pred_index in used_preds or gt_index in used_gts:
            continue
        used_preds.add(pred_index)
        used_gts.add(gt_index)
        matches.append((pred_index, gt_index, score))
    tp = len(matches)
    fp = len(pred_boxes) - tp  # includes malformed and duplicates
    fn = len(gt_boxes) - tp
    return MatchResult(tp=tp, fp=fp, fn=fn, matches=matches,
                       malformed_pred_indices=malformed)


def _to_math(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Image coords (y down) to math coords (y up) for signed geometry."""
    return [(x, -y) for x, y in poly]


def _signed_area(poly: list[tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        total += x0 * y1 - x1 * y0
    return total / 2.0


def _clip_edge(
    subject: list[tuple[float, float]],
    start: tuple[float, float],
    end: tuple[float, float],
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman: clip subject to the left of directed edge."""
    if not subject:
        return []
    out: list[tuple[float, float]] = []
    ex, ey = end[0] - start[0], end[1] - start[1]

    def inside(p: tuple[float, float]) -> bool:
        return ex * (p[1] - start[1]) - ey * (p[0] - start[0]) >= -1e-9

    def cross(s: tuple[float, float], e: tuple[float, float]):
        dx, dy = e[0] - s[0], e[1] - s[1]
        denom = ex * dy - ey * dx
        if abs(denom) < 1e-12:
            return s
        t = -(ex * (s[1] - start[1]) - ey * (s[0] - start[0])) / denom
        return (s[0] + t * dx, s[1] + t * dy)

    prev = subject[-1]
    for point in subject:
        pin, prev_in = inside(point), inside(prev)
        if pin:
            if not prev_in:
                out.append(cross(prev, point))
            out.append(point)
        elif prev_in:
            out.append(cross(prev, point))
        prev = point
    return out


def clip_polygon_to_image(
    poly: list[tuple[float, float]], width: int, height: int
) -> list[tuple[float, float]]:
    """Clip a convex polygon to the [0, W] x [0, H] sensor frame.

    Used for IoU math only: annotations outside the frame cannot match
    anything real, so clipping changes no verdict — it only bounds the
    geometry. Returns math coords converted back to image coords.
    """
    subject = _to_math([(float(x), float(y)) for x, y in poly])
    clip = [(0.0, float(-height)), (float(width), float(-height)),
            (float(width), 0.0), (0.0, 0.0)]
    for start, end in zip(clip, clip[1:] + clip[:1]):
        subject = _clip_edge(subject, start, end)
        if not subject:
            return []
    return [(x, -y) for x, y in subject]


def polygon_area(poly: list[tuple[float, float]]) -> float:
    """Absolute area. Degenerate (<3 points) polygons have area 0."""
    if len(poly) < 3:
        return 0.0
    return abs(_signed_area(_to_math(poly)))


def polygon_intersection_area(
    first: list[tuple[float, float]], second: list[tuple[float, float]]
) -> float:
    """Exact intersection area of two convex polygons (Sutherland-Hodgman)."""
    subject = _to_math(first)
    clip = _to_math(second)
    if _signed_area(clip) < 0:
        clip = clip[::-1]
    for start, end in zip(clip, clip[1:] + clip[:1]):
        subject = _clip_edge(subject, start, end)
        if not subject:
            return 0.0
    return abs(_signed_area(subject))


def rotated_iou(
    first: list[tuple[float, float]], second: list[tuple[float, float]]
) -> float:
    """Exact IoU of two convex quadrilaterals in image coords.

    Works for rotated-vs-rotated and axis-aligned-vs-rotated pairs alike;
    for two identical axis-aligned boxes it agrees with iou(). Zero-area
    inputs yield 0.0 (no measurable overlap), matching iou().
    """
    inter = polygon_intersection_area(first, second)
    union = polygon_area(first) + polygon_area(second) - inter
    return inter / union if union > 1e-12 else 0.0


def _as_polygon(shape: Any) -> list[tuple[float, float]] | None:
    """4 finite corners or None (malformed)."""
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        return None
    corners = []
    for corner in shape:
        if (
            not isinstance(corner, (list, tuple))
            or len(corner) != 2
            or not all(isinstance(v, (int, float)) and v == v for v in corner)
        ):
            return None
        corners.append((float(corner[0]), float(corner[1])))
    return corners


def match_rotated(
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Greedy rotated-IoU matching with a class constraint.

    Each entry holds polygon + category. Only same-category pairs can
    match; each GT matches at most once, so duplicates are false
    positives. Malformed predictions are listed false positives.
    """
    parsed = [(_as_polygon(p.get("polygon")), p.get("category"), i)
              for i, p in enumerate(preds)]
    malformed = sorted(i for poly, _, i in parsed if poly is None)
    gt_parsed = [(_as_polygon(g.get("polygon")), g.get("category"))
                 for g in gts]
    if any(poly is None for poly, _ in gt_parsed):
        raise ValueError("ground-truth polygons must be 4 finite corners")
    scored: list[tuple[float, int, int]] = []
    for poly, category, pred_index in parsed:
        if poly is None:
            continue
        for gt_index, (gt_poly, gt_category) in enumerate(gt_parsed):
            if category != gt_category:
                continue
            score = rotated_iou(poly, gt_poly)
            if score >= iou_threshold:
                scored.append((score, pred_index, gt_index))
    scored.sort(key=lambda triple: (-triple[0], triple[1], triple[2]))
    used_preds, used_gts, matches = set(), set(), []
    for score, pred_index, gt_index in scored:
        if pred_index in used_preds or gt_index in used_gts:
            continue
        used_preds.add(pred_index)
        used_gts.add(gt_index)
        matches.append((pred_index, gt_index, score))
    tp = len(matches)
    return MatchResult(tp=tp, fp=len(preds) - tp, fn=len(gts) - tp,
                       matches=matches, malformed_pred_indices=malformed)


def prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    """Precision/recall/F1. Undefined denominators stay None, never 0.0."""
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}
