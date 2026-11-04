"""Offline unit tests for defect physics, structural mechanics, and dual-metric evaluation."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neev.eval.centerline_eval import evaluate_dual_tolerance
from neev.eval.labelstudio import LSImage, LSRect
from neev.measurements.defect_physics import (
    ORIENTATION_DIAGONAL_SHEAR,
    ORIENTATION_HORIZONTAL_FLEXURE,
    ORIENTATION_VERTICAL_SETTLEMENT,
    SEVERITY_COSMETIC,
    SEVERITY_STRUCTURAL,
    analyze_crack_physics,
)


def test_defect_physics_box_fallback():
    # 45-degree diagonal box (100x100)
    box = {"x0": 0, "y0": 0, "x1": 100, "y1": 100}
    rep = analyze_crack_physics(box)

    assert rep.orientation_type == ORIENTATION_DIAGONAL_SHEAR
    assert rep.seismic_risk_factor == 1.5
    assert rep.aspect_ratio == 1.0

    # Horizontal box (100x10) -> flexural
    horiz_box = {"x0": 0, "y0": 0, "x1": 100, "y1": 10}
    h_rep = analyze_crack_physics(horiz_box)
    assert h_rep.orientation_type == ORIENTATION_HORIZONTAL_FLEXURE
    assert h_rep.seismic_risk_factor == 1.0

    # Vertical box (10x100) -> settlement
    vert_box = {"x0": 0, "y0": 0, "x1": 10, "y1": 100}
    v_rep = analyze_crack_physics(vert_box)
    assert v_rep.orientation_type == ORIENTATION_VERTICAL_SETTLEMENT


def test_defect_physics_with_raster():
    # Synthetic 20x20 binary raster with a 2-pixel wide diagonal stripe
    w, h = 20, 20
    raster = bytearray(w * h)
    for i in range(20):
        raster[i * w + i] = 1
        if i + 1 < 20:
            raster[i * w + (i + 1)] = 1

    box = {"x0": 0, "y0": 0, "x1": 20, "y1": 20}
    rep = analyze_crack_physics(box, bytes(raster), w, h, gsd_mm_per_px=0.1)

    assert rep.orientation_type == ORIENTATION_DIAGONAL_SHEAR
    assert rep.seismic_risk_factor == 1.5
    assert rep.pixel_area > 0
    assert rep.max_width_px > 0
    assert rep.estimated_width_mm is not None
    assert rep.estimated_length_mm is not None


def test_centerline_eval_dual_tolerance():
    # Synthetic image with 2 cracks
    img = LSImage(
        task_id=1,
        image_id="img1",
        relpath="img1.png",
        sha256="sha1",
        width=1000,
        height=1000,
        source="test",
        is_negative=False,
        rects=[
            LSRect(1, "r1", "crack", 10.0, 10.0, 5.0, 5.0, 0.0, 1000, 1000, "p", "s", "src"),
            LSRect(1, "r2", "crack", 40.0, 40.0, 10.0, 10.0, 0.0, 1000, 1000, "p", "s", "src"),
        ],
    )

    # Prediction 1 matches r1 at IoU ~0.35 (strict 0.50 fails, relaxed 0.25 passes)
    # Prediction 2 matches r2 at IoU 1.0 (both pass)
    preds = {
        ("sha1", "crack"): {
            "boxes": [
                {"x0": 100, "y0": 115, "x1": 150, "y1": 185},  # partial overlap on r1 (100..150, 100..150)
                {"x0": 400, "y0": 400, "x1": 500, "y1": 500},  # exact match on r2
            ]
        }
    }

    report = evaluate_dual_tolerance([img], preds, category="crack", prompt="crack")
    assert report.total_gt == 2
    assert report.total_preds == 2

    # Strict: only 1 TP (exact match)
    assert report.strict_iou_050.tp == 1

    # Relaxed: 2 TPs (both match!)
    assert report.relaxed_iou_025.tp == 2
    assert report.relaxed_iou_025.recall == 1.0
    assert report.relaxed_iou_025.f1 > report.strict_iou_050.f1


def test_crack_dsp_frangi_and_skeleton():
    import numpy as np
    from neev.measurements.crack_dsp import compute_frangi_vesselness, extract_skeleton_topology

    # Synthetic image: 60x60 white wall with dark diagonal crack
    img = np.ones((60, 60), dtype=np.float32) * 0.8
    for i in range(15, 45):
        img[i, i] = 0.1
        img[i, i + 1] = 0.1

    vessel = compute_frangi_vesselness(img)
    assert vessel.shape == (60, 60)
    # Ridge response should be strong along the crack
    assert vessel[25, 25] > 0.4
    # Wall background response should be near 0
    assert vessel[5, 5] < 0.1

    # Extract skeleton topology on binary crack
    binary = img < 0.5
    skel = extract_skeleton_topology(binary)
    assert skel.length_px > 0
    assert len(skel.endpoints) >= 2
    assert skel.tortuosity >= 1.0


def test_crack_dsp_aperture_distribution():
    import numpy as np
    from neev.measurements.crack_dsp import extract_aperture_profile

    # Synthetic stripe of thickness 4 pixels
    binary = np.zeros((50, 50), dtype=bool)
    binary[10:14, 10:40] = True

    aperture = extract_aperture_profile(binary, gsd_mm_per_px=0.5)
    assert aperture.max_width_px >= 3.5
    assert aperture.p95_width_px > 0
    assert aperture.estimated_max_mm is not None
    assert aperture.estimated_max_mm > 0


def test_map_cracking_classification():
    import numpy as np
    from neev.measurements.defect_physics import ORIENTATION_MAP_CRACKING

    # Cross / star pattern with multiple branches
    w, h = 40, 40
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[20, 5:35] = 1
    arr[5:35, 20] = 1
    # Add diagonal arms to ensure >= 3 junctions and tortuosity
    for i in range(12, 28):
        arr[i, i] = 1
        arr[i, 40 - i - 1] = 1

    box = {"x0": 5, "y0": 5, "x1": 35, "y1": 35}
    rep = analyze_crack_physics(box, bytes(arr), w, h)
    assert rep.orientation_type == ORIENTATION_MAP_CRACKING
    assert "Shrinkage" in rep.failure_mechanism
    assert rep.branch_points >= 2
    assert len(rep.crack_tips) > 0
