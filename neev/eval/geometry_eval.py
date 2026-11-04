"""Geometry-Aware Defect Evaluation & Oriented Bounding Rectangle (OMBR) Engine.

Provides compatible geometric representations for thin and oriented defects:
- Minimum Bounding Rectangle (OMBR): derives tight rotated bounding boxes from
  SAM segmentation masks via rotating calipers on convex hull vertices.
- Safe Fallback: handles empty masks, disconnected components, and degenerate
  geometries transparently, falling back to axis-aligned boxes when needed.
- Geometry-Aware Matching: evaluates oriented predicted geometry directly against
  Label Studio rotated ground-truth rectangles using exact polygon IoU.
- Dual Comparison: scores axis-aligned and oriented representations side-by-side
  on identical splits and thresholds (0.25, 0.50, 0.75).

No live model calls. Preserves all frozen baseline artifacts.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from neev.eval.labelstudio import LSImage, rect_polygon
from neev.eval.matching import match_rotated, polygon_area, rotated_iou

# Mask decoder signature: (payload, width, height) -> raster bytes
MaskBackend = Callable[[str, int, int], bytes]


def minimum_bounding_rectangle(points: np.ndarray) -> Optional[List[Tuple[float, float]]]:
    """Compute the minimum-area oriented bounding box for 2D points.

    Uses rotating calipers on the 2D convex hull.
    Returns 4 cyclic corners [(x0, y0), (x1, y1), (x2, y2), (x3, y3)],
    or None if points are degenerate (<3 unique non-collinear points).
    """
    if points is None or len(points) < 3:
        return None

    # Filter non-finite points
    valid_mask = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    pts = points[valid_mask]
    if len(pts) < 3:
        return None

    unique_pts = np.unique(pts, axis=0)
    if len(unique_pts) < 3:
        return None

    try:
        hull = ConvexHull(unique_pts)
    except (QhullError, ValueError):
        # Collinear or degenerate 1D point set
        return None

    hull_pts = unique_pts[hull.vertices]
    n_hull = len(hull_pts)
    if n_hull < 3:
        return None

    edges = np.roll(hull_pts, -1, axis=0) - hull_pts
    edge_lens = np.hypot(edges[:, 0], edges[:, 1])
    valid_edges = edge_lens > 1e-9
    if not np.any(valid_edges):
        return None

    u_vecs = edges[valid_edges] / edge_lens[valid_edges, np.newaxis]
    v_vecs = np.column_stack([-u_vecs[:, 1], u_vecs[:, 0]])

    # Project hull points onto edge coordinate frames: (n_edges, n_hull)
    proj_u = np.dot(u_vecs, hull_pts.T)
    proj_v = np.dot(v_vecs, hull_pts.T)

    min_u = np.min(proj_u, axis=1)
    max_u = np.max(proj_u, axis=1)
    min_v = np.min(proj_v, axis=1)
    max_v = np.max(proj_v, axis=1)

    areas = (max_u - min_u) * (max_v - min_v)
    best_idx = np.argmin(areas)

    u = u_vecs[best_idx]
    v = v_vecs[best_idx]
    u0, u1 = min_u[best_idx], max_u[best_idx]
    v0, v1 = min_v[best_idx], max_v[best_idx]

    c0 = u * u0 + v * v0
    c1 = u * u1 + v * v0
    c2 = u * u1 + v * v1
    c3 = u * u0 + v * v1

    corners = [(float(c[0]), float(c[1])) for c in [c0, c1, c2, c3]]
    # Ensure non-zero area
    if polygon_area(corners) <= 1e-6:
        return None
    return corners


def mask_to_oriented_box(
    mask_raster: bytes,
    mask_w: int,
    mask_h: int,
    box: dict[str, Any],
    img_w: int,
    img_h: int,
) -> Optional[List[Tuple[float, float]]]:
    """Derive minimum bounding rectangle from binary mask raster inside a box."""
    if not mask_raster or mask_w <= 0 or mask_h <= 0:
        return None
    if len(mask_raster) != mask_w * mask_h:
        return None

    arr = np.frombuffer(mask_raster, dtype=np.uint8).reshape((mask_h, mask_w))
    ys, xs = np.nonzero(arr)
    if len(xs) < 3:
        return None

    bw = float(box["x1"] - box["x0"])
    bh = float(box["y1"] - box["y0"])
    if bw <= 0 or bh <= 0:
        return None

    scale_x = bw / float(mask_w)
    scale_y = bh / float(mask_h)

    full_xs = box["x0"] + xs.astype(np.float64) * scale_x
    full_ys = box["y0"] + ys.astype(np.float64) * scale_y

    # Clip coordinates to image boundary
    full_xs = np.clip(full_xs, 0.0, float(img_w))
    full_ys = np.clip(full_ys, 0.0, float(img_h))

    pts = np.column_stack([full_xs, full_ys])
    return minimum_bounding_rectangle(pts)


def aabb_to_polygon(box: dict[str, Any]) -> List[Tuple[float, float]]:
    """Convert axis-aligned box to 4 cyclic corners."""
    x0, y0 = float(box["x0"]), float(box["y0"])
    x1, y1 = float(box["x1"]), float(box["y1"])
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


@dataclass(frozen=True)
class DerivedGeometry:
    geometry_type: str  # "oriented_box" | "axis_aligned"
    polygon: List[Tuple[float, float]]
    is_oriented: bool
    mask_pixels: int = 0
    fallback_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometry_type": self.geometry_type,
            "polygon": self.polygon,
            "is_oriented": self.is_oriented,
            "mask_pixels": self.mask_pixels,
            "fallback_reason": self.fallback_reason,
        }


def derive_geometry(
    box: dict[str, Any],
    mask_payload: Optional[str],
    mask_w: Optional[int],
    mask_h: Optional[int],
    img_w: int,
    img_h: int,
    mask_backend: Optional[MaskBackend] = None,
) -> DerivedGeometry:
    """Safely derive oriented box from mask, or fall back to axis-aligned box."""
    aabb_poly = aabb_to_polygon(box)

    if not mask_payload or not mask_w or not mask_h or mask_backend is None:
        return DerivedGeometry(
            geometry_type="axis_aligned",
            polygon=aabb_poly,
            is_oriented=False,
            fallback_reason="no_mask_payload_or_backend",
        )

    try:
        raster = mask_backend(mask_payload, mask_w, mask_h)
    except Exception as exc:
        return DerivedGeometry(
            geometry_type="axis_aligned",
            polygon=aabb_poly,
            is_oriented=False,
            fallback_reason=f"decode_error: {exc}",
        )

    if not raster:
        return DerivedGeometry(
            geometry_type="axis_aligned",
            polygon=aabb_poly,
            is_oriented=False,
            fallback_reason="empty_raster",
        )

    arr = np.frombuffer(raster, dtype=np.uint8)
    on_px = int(np.sum(arr))
    if on_px < 3:
        return DerivedGeometry(
            geometry_type="axis_aligned",
            polygon=aabb_poly,
            is_oriented=False,
            mask_pixels=on_px,
            fallback_reason="insufficient_foreground_pixels",
        )

    poly = mask_to_oriented_box(raster, mask_w, mask_h, box, img_w, img_h)
    if poly is None:
        return DerivedGeometry(
            geometry_type="axis_aligned",
            polygon=aabb_poly,
            is_oriented=False,
            mask_pixels=on_px,
            fallback_reason="degenerate_contour",
        )

    return DerivedGeometry(
        geometry_type="oriented_box",
        polygon=poly,
        is_oriented=True,
        mask_pixels=on_px,
        fallback_reason=None,
    )


@dataclass
class MetricThresholdSummary:
    threshold: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


@dataclass
class GeometryAwareReport:
    category: str
    prompt: str
    total_gt: int
    total_predictions: int
    eligible_masks: int
    fallback_boxes: int
    metrics_by_threshold: dict[float, dict[str, MetricThresholdSummary]]
    per_image: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_geometry_aware(
    gt_images: list[LSImage],
    predictions: dict[tuple[str, str], dict[str, Any]],
    category: str = "crack",
    prompt: str = "crack in wall",
    thresholds: tuple[float, ...] = (0.25, 0.50, 0.75),
    mask_backend: Optional[MaskBackend] = None,
) -> GeometryAwareReport:
    """Evaluate predictions using both axis-aligned and mask-oriented representations."""
    total_gt = 0
    total_preds = 0
    eligible_masks = 0
    fallback_boxes = 0

    image_entries: list[dict[str, Any]] = []

    for img in gt_images:
        rects = [r for r in img.rects if r.label == category]
        total_gt += len(rects)

        gt_polys = [rect_polygon(r, img.width, img.height) for r in rects]
        gt_items = [{"polygon": p, "category": category} for p in gt_polys]

        pred_data = predictions.get((img.sha256, prompt), {})
        dets = pred_data.get("detections", [])
        total_preds += len(dets)

        aabb_items = []
        ombr_items = []

        for d in dets:
            box = d.get("box", {})
            aabb_items.append({
                "polygon": aabb_to_polygon(box),
                "category": category,
            })

            geom = derive_geometry(
                box=box,
                mask_payload=d.get("mask_payload"),
                mask_w=d.get("mask_width"),
                mask_h=d.get("mask_height"),
                img_w=img.width,
                img_h=img.height,
                mask_backend=mask_backend,
            )

            if geom.is_oriented:
                eligible_masks += 1
            else:
                fallback_boxes += 1

            ombr_items.append({
                "polygon": geom.polygon,
                "category": category,
            })

        image_entries.append({
            "image_id": img.image_id,
            "gt_items": gt_items,
            "aabb_items": aabb_items,
            "ombr_items": ombr_items,
        })

    metrics: dict[float, dict[str, MetricThresholdSummary]] = {}
    per_image_gains: dict[str, dict[str, Any]] = {}

    for thresh in thresholds:
        aabb_tp, aabb_fp, aabb_fn = 0, 0, 0
        ombr_tp, ombr_fp, ombr_fn = 0, 0, 0

        for entry in image_entries:
            iid = entry["image_id"]
            gt = entry["gt_items"]
            aabb_p = entry["aabb_items"]
            ombr_p = entry["ombr_items"]

            m_aabb = match_rotated(aabb_p, gt, iou_threshold=thresh)
            aabb_tp += m_aabb.tp
            aabb_fp += m_aabb.fp
            aabb_fn += m_aabb.fn

            m_ombr = match_rotated(ombr_p, gt, iou_threshold=thresh)
            ombr_tp += m_ombr.tp
            ombr_fp += m_ombr.fp
            ombr_fn += m_ombr.fn

            if thresh == 0.50:
                per_image_gains[iid] = {
                    "gt_count": len(gt),
                    "pred_count": len(aabb_p),
                    "aabb_tp": m_aabb.tp,
                    "ombr_tp": m_ombr.tp,
                    "delta_tp": m_ombr.tp - m_aabb.tp,
                }

        def _calc_summary(tp: int, fp: int, fn: int) -> MetricThresholdSummary:
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            return MetricThresholdSummary(
                threshold=thresh,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=round(p, 4),
                recall=round(r, 4),
                f1=round(f1, 4),
            )

        metrics[thresh] = {
            "aabb": _calc_summary(aabb_tp, aabb_fp, aabb_fn),
            "ombr": _calc_summary(ombr_tp, ombr_fp, ombr_fn),
        }

    return GeometryAwareReport(
        category=category,
        prompt=prompt,
        total_gt=total_gt,
        total_predictions=total_preds,
        eligible_masks=eligible_masks,
        fallback_boxes=fallback_boxes,
        metrics_by_threshold=metrics,
        per_image=per_image_gains,
    )
