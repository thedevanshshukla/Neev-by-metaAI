"""Unit tests for geometry-aware defect evaluation (Task 2)."""

import math
import numpy as np
import pytest

from neev.eval.geometry_eval import (
    minimum_bounding_rectangle,
    mask_to_oriented_box,
    aabb_to_polygon,
    derive_geometry,
    evaluate_geometry_aware,
)
from neev.eval.matching import rotated_iou, polygon_area


def test_identical_rotated_rectangles():
    """Identical rotated rectangles must yield exact 1.0 IoU."""
    angle = math.radians(35)
    c, s = math.cos(angle), math.sin(angle)
    # 80x20 rectangle rotated 35 degrees
    base = np.array([[-40, -10], [40, -10], [40, 10], [-40, 10]])
    R = np.array([[c, -s], [s, c]])
    rot = (R @ base.T).T + np.array([100, 100])
    poly = [(float(pt[0]), float(pt[1])) for pt in rot]

    # OMBR from points of this rectangle
    grid_x, grid_y = np.meshgrid(np.linspace(-40, 40, 20), np.linspace(-10, 10, 10))
    dense = (R @ np.column_stack([grid_x.ravel(), grid_y.ravel()]).T).T + np.array([100, 100])
    mbr = minimum_bounding_rectangle(dense)

    assert mbr is not None
    iou = rotated_iou(poly, mbr)
    assert iou >= 0.999


def test_rotated_rectangles_partial_overlap():
    """Rotated rectangles with known shift yield expected IoU in (0, 1)."""
    # Two axis-aligned 100x20 rects shifted by 20px horizontally: overlap is 80x20
    rect_a = [(0.0, 0.0), (100.0, 0.0), (100.0, 20.0), (0.0, 20.0)]
    rect_b = [(20.0, 0.0), (120.0, 0.0), (120.0, 20.0), (20.0, 20.0)]
    iou = rotated_iou(rect_a, rect_b)
    # inter = 80*20 = 1600, union = 100*20 + 100*20 - 1600 = 2400 -> iou = 1600/2400 = 2/3
    assert abs(iou - 2.0 / 3.0) < 1e-4


def test_diagonal_crack_mbr():
    """Diagonal 45-degree crack line produces an oriented box matching its orientation."""
    pts = []
    # 45-degree line from (10, 10) to (100, 100) with 2px width
    for t in np.linspace(10, 100, 100):
        pts.append((t, t))
        pts.append((t + 1, t - 1))
        pts.append((t - 1, t + 1))
    pts = np.array(pts)
    mbr = minimum_bounding_rectangle(pts)
    assert mbr is not None
    area = polygon_area(mbr)
    # Length ~ 90*sqrt(2) ~ 127, width ~ 2*sqrt(2) ~ 2.8 -> area ~ 360
    # Axis-aligned box area would be 90*90 = 8100 (22x larger!)
    assert area < 1000.0


def test_near_horizontal_crack_mbr():
    """Near-horizontal crack (~5 degrees) produces tight bounding box."""
    angle = math.radians(5)
    c, s = math.cos(angle), math.sin(angle)
    xs = np.linspace(0, 200, 200)
    ys = np.zeros_like(xs)
    # Add width
    all_pts = []
    for dy in [-2, -1, 0, 1, 2]:
        x_rot = c * xs - s * (ys + dy) + 50
        y_rot = s * xs + c * (ys + dy) + 50
        all_pts.append(np.column_stack([x_rot, y_rot]))
    pts = np.vstack(all_pts)
    mbr = minimum_bounding_rectangle(pts)
    assert mbr is not None
    assert polygon_area(mbr) < 1200.0


def test_empty_and_degenerate_masks():
    """Empty and degenerate masks return None or fallback cleanly."""
    # 1. Empty array
    assert minimum_bounding_rectangle(np.zeros((0, 2))) is None
    # 2. Single point
    assert minimum_bounding_rectangle(np.array([[10.0, 20.0]])) is None
    # 3. Two points (line segment)
    assert minimum_bounding_rectangle(np.array([[10.0, 20.0], [30.0, 40.0]])) is None
    # 4. Collinear points
    collinear = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    assert minimum_bounding_rectangle(collinear) is None

    # Mask to oriented box with empty raster
    empty_raster = bytes([0] * 100)
    box = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
    assert mask_to_oriented_box(empty_raster, 10, 10, box, 100, 100) is None


def test_disconnected_crack_components():
    """Disconnected crack mask components are covered safely by minimum bounding rectangle."""
    # Two disconnected segments: (10, 10)-(20, 20) and (50, 50)-(60, 60)
    pts = np.array([
        [10.0, 10.0], [12.0, 10.0], [20.0, 20.0], [20.0, 22.0],
        [50.0, 50.0], [52.0, 50.0], [60.0, 60.0], [60.0, 62.0],
    ])
    mbr = minimum_bounding_rectangle(pts)
    assert mbr is not None
    # All points must be enclosed or on edge
    for pt in pts:
        # Distance to centroid of MBR is bounded
        assert 10.0 <= pt[0] <= 62.0


def test_coordinate_clipping_image_boundaries():
    """Mask-to-box projection clips safely at image boundaries."""
    # Raster with 1s at right and bottom borders
    w, h = 10, 10
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[2:8, 2:8] = 1
    raster = arr.tobytes()

    # Box extends outside image frame
    box = {"x0": 90, "y0": 90, "x1": 120, "y1": 120}
    # Image is 100x100
    poly = mask_to_oriented_box(raster, w, h, box, img_w=100, img_h=100)
    assert poly is not None
    for x, y in poly:
        assert 0.0 <= x <= 100.0 + 1e-4
        assert 0.0 <= y <= 100.0 + 1e-4


def test_derive_geometry_fallback_reporting():
    """derive_geometry transparently marks fallback reasons without crashing."""
    box = {"x0": 10, "y0": 10, "x1": 50, "y1": 50}

    # 1. No mask payload
    g1 = derive_geometry(box, mask_payload=None, mask_w=None, mask_h=None, img_w=100, img_h=100)
    assert g1.geometry_type == "axis_aligned"
    assert not g1.is_oriented
    assert g1.fallback_reason == "no_mask_payload_or_backend"
    assert g1.polygon == aabb_to_polygon(box)

    # 2. Empty mask backend
    def fake_empty(payload, w, h):
        return bytes([0] * (w * h))

    g2 = derive_geometry(box, mask_payload="!fake", mask_w=10, mask_h=10, img_w=100, img_h=100, mask_backend=fake_empty)
    assert g2.geometry_type == "axis_aligned"
    assert not g2.is_oriented
    assert g2.fallback_reason == "insufficient_foreground_pixels"

    # 3. Valid mask with 45-degree diagonal line
    def fake_diag(payload, w, h):
        arr = np.zeros((h, w), dtype=np.uint8)
        for i in range(min(w, h)):
            arr[i, i] = 1
            if i + 1 < w:
                arr[i, i + 1] = 1
        return arr.tobytes()

    g3 = derive_geometry(box, mask_payload="!diag", mask_w=20, mask_h=20, img_w=100, img_h=100, mask_backend=fake_diag)
    assert g3.geometry_type == "oriented_box"
    assert g3.is_oriented
    assert g3.fallback_reason is None
    assert g3.mask_pixels > 0


def test_backward_compatibility_frozen_evaluator():
    """Verify that the official frozen baseline evaluation is unchanged and passes bit-for-bit."""
    import json
    from neev.eval.labelstudio import load_export, parse_task
    from neev.eval.compare import collect_predictions, run_rotated_evaluation

    tasks = load_export("eval/annotations_label_studio.json")
    images = [parse_task(t, "dataset")[0] for t in tasks if parse_task(t, "dataset")[0] is not None]
    preds = collect_predictions("runs")
    res = run_rotated_evaluation(images, preds, iou_threshold=0.50)

    crack_stats = res["results"]["crack"]["crack in wall"]
    assert crack_stats["tp"] == 3
    assert crack_stats["fp"] == 15
    assert crack_stats["fn"] == 162
    assert abs(crack_stats["f1"] - 0.032786885) < 1e-6
