"""Mask-instance evaluation for the crack pilot (additive, offline).

Scores predicted crack masks against hand-drawn ground-truth masks with
pairwise raster IoU and greedy class-constrained one-to-one matching,
mirroring the strict box matcher's semantics. Box metrics are untouched;
nothing here converts a box into a mask. Pure and deterministic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from neev.eval.matching import MatchResult, mask_iou, prf

Point = tuple[float, float]


@dataclass(frozen=True)
class MaskInstance:
    """One mask: row-major 0/1 raster at native image dims."""

    image_id: str
    instance_id: str
    width: int
    height: int
    raster: bytes
    category: str = "crack"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("mask dims must be positive")
        if len(self.raster) != self.width * self.height:
            raise ValueError(
                f"raster is {len(self.raster)} bytes, want "
                f"{self.width * self.height}")
        if any(v not in (0, 1) for v in self.raster):
            raise ValueError("raster must hold 0/1 only")

    @property
    def area(self) -> int:
        return sum(self.raster)


def _check_ring(ring: Any) -> list[Point] | None:
    """Validated [(x, y)] ring or None when malformed."""
    if not isinstance(ring, (list, tuple)) or len(ring) < 3:
        return None
    pts: list[Point] = []
    for pt in ring:
        if (not isinstance(pt, (list, tuple)) or len(pt) != 2
                or not all(isinstance(v, (int, float)) and v == v
                           for v in pt)):
            return None
        pts.append((float(pt[0]), float(pt[1])))
    return pts


def rasterize_polygons(polygons: list[Any], width: int, height: int
                       ) -> bytes | None:
    """Even-odd scanline fill of polygon rings; None if any ring is bad.

    A pixel is filled when its center falls inside the ring set.
    Out-of-frame geometry is clipped by construction (only in-frame
    pixels are ever set), so edge-truncated cracks evaluate cleanly.
    """
    if width <= 0 or height <= 0:
        return None
    grid = bytearray(width * height)
    for ring in polygons:
        pts = _check_ring(ring)
        if pts is None:
            return None
        ys = [y for _, y in pts]
        y0 = max(0, math.floor(min(ys)))
        y1 = min(height, math.ceil(max(ys)))
        n = len(pts)
        for y in range(y0, y1):
            yc = y + 0.5
            xs: list[float] = []
            for i in range(n):
                (x1, y1v), (x2, y2v) = pts[i], pts[(i + 1) % n]
                if (y1v <= yc < y2v) or (y2v <= yc < y1v):
                    xs.append(x1 + (yc - y1v) * (x2 - x1) / (y2v - y1v))
            xs.sort()
            for j in range(0, len(xs) - 1, 2):
                xa = max(0, math.ceil(xs[j] - 0.5))
                xb = min(width, math.ceil(xs[j + 1] - 0.5))
                for x in range(xa, xb):
                    grid[y * width + x] ^= 1
    return bytes(grid)


def polygons_to_instance(image_id: str, instance_id: str, width: int,
                         height: int, polygons: list[Any],
                         category: str = "crack") -> MaskInstance | None:
    """GT/pred polygons to an instance; None when malformed or empty-area.

    Empty-area results are None (no measurable mask), letting callers
    count them as missing rather than silently scoring zeros.
    """
    raster = rasterize_polygons(polygons, width, height)
    if raster is None or not any(raster):
        return None
    return MaskInstance(image_id=image_id, instance_id=instance_id,
                        width=width, height=height, raster=raster,
                        category=category)


def pairwise_iou(preds: list[MaskInstance], gts: list[MaskInstance]
                 ) -> list[list[float]]:
    """Raster IoU for every same-image (pred, gt) pair.

    Cross-image or cross-size pairs are caller errors and raise.
    """
    matrix = []
    for p in preds:
        row = []
        for g in gts:
            if (p.image_id != g.image_id or p.width != g.width
                    or p.height != g.height):
                raise ValueError("pred/gt image or dims mismatch: "
                                 f"{p.instance_id} vs {g.instance_id}")
            row.append(mask_iou(p.raster, g.raster) if
                       p.category == g.category else 0.0)
        matrix.append(row)
    return matrix


def match_masks(preds: list[MaskInstance | None], gts: list[MaskInstance],
                iou_threshold: float = 0.5) -> MatchResult:
    """Greedy one-to-one mask matching with a class constraint.

    None predictions are malformed (FPs). GT must be valid instances.
    Sort order ((-score, pred, gt)) matches the box matchers exactly.
    """
    malformed = sorted(i for i, p in enumerate(preds) if p is None)
    well = [(p, i) for i, p in enumerate(preds) if p is not None]
    if any(g is None for g in gts):
        raise ValueError("ground-truth masks must be valid instances")
    scored: list[tuple[float, int, int]] = []
    matrix = pairwise_iou([p for p, _ in well], list(gts))
    for row, (_, pred_index) in zip(matrix, well):
        for gt_index, score in enumerate(row):
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


@dataclass
class ImageScore:
    image_id: str
    iou_threshold: float
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None
    matches: list[dict[str, Any]] = field(default_factory=list)
    malformed_preds: int = 0


def evaluate_image(image_id: str, preds: list[MaskInstance | None],
                   gts: list[MaskInstance], iou_threshold: float = 0.5
                   ) -> ImageScore:
    """Score one image; empty lists are valid (negatives, misses)."""
    match = match_masks(preds, gts, iou_threshold)
    well_ids = [p.instance_id if p is not None else f"malformed#{i}"
                for i, p in enumerate(preds)]
    metrics = prf(match.tp, match.fp, match.fn)
    return ImageScore(
        image_id=image_id, iou_threshold=iou_threshold,
        tp=match.tp, fp=match.fp, fn=match.fn,
        precision=metrics["precision"], recall=metrics["recall"],
        f1=metrics["f1"],
        matches=[{"pred_id": well_ids[pi], "gt_id": gts[gi].instance_id,
                  "iou": round(score, 4)}
                 for pi, gi, score in match.matches],
        malformed_preds=len(match.malformed_pred_indices))


def pool_scores(scores: list[ImageScore]) -> dict[str, Any]:
    """Micro-average image scores (sums; prf over totals)."""
    tp = sum(s.tp for s in scores)
    fp = sum(s.fp for s in scores)
    fn = sum(s.fn for s in scores)
    return {"tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn),
            "images": len(scores)}
