"""Offline unit tests for error analysis and size stratification (Phase 1)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neev.eval.error_analysis import (
    STATUS_INSUFFICIENT_DATA,
    STATUS_SUPPORTED_EVALUATED,
    STATUS_UNSUPPORTED_UNMAPPED,
    STATUS_ZERO_DETECTIONS,
    audit_categories,
    classify_size,
    run_error_analysis,
)
from neev.eval.labelstudio import LSImage, LSRect


def _make_ls_rect(rid: str, label: str, x: float, y: float, w: float, h: float, rot: float = 0.0) -> LSRect:
    return LSRect(
        task_id=1,
        result_id=rid,
        label=label,
        x_pct=x,
        y_pct=y,
        w_pct=w,
        h_pct=h,
        rotation_deg=rot,
        orig_width=1000,
        orig_height=1000,
        image_relpath="test.png",
        image_sha256="test_sha",
        source="test",
    )


def test_classify_size_bins():
    assert classify_size(10.0) == "tiny"
    assert classify_size(31.9) == "tiny"
    assert classify_size(32.0) == "small"
    assert classify_size(63.9) == "small"
    assert classify_size(64.0) == "medium"
    assert classify_size(127.9) == "medium"
    assert classify_size(128.0) == "large"
    assert classify_size(500.0) == "large"


def test_audit_categories_classifies_correctly():
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
            _make_ls_rect("r1", "crack", 10, 10, 5, 5),
            _make_ls_rect("r2", "crack", 20, 20, 5, 5),
            _make_ls_rect("r3", "crack", 30, 30, 5, 5),
            _make_ls_rect("r4", "spalling", 40, 40, 5, 5),
            _make_ls_rect("r5", "rare_defect", 50, 50, 5, 5),
            _make_ls_rect("r6", "floor_slab", 60, 60, 5, 5),
            _make_ls_rect("r7", "floor_slab", 70, 70, 5, 5),
            _make_ls_rect("r8", "floor_slab", 80, 80, 5, 5),
        ],
    )
    prompt_to_cat = {
        "crack in wall": "crack",
        "floor slab": "floor_slab",
        "rare prompt": "rare_defect",
    }
    preds = {
        ("sha1", "crack in wall"): {"boxes": [{"x0": 100, "y0": 100, "x1": 150, "y1": 150}]},
        ("sha1", "floor slab"): {"boxes": []},
        ("sha1", "rare prompt"): {"boxes": [{"x0": 1, "y0": 1, "x1": 2, "y1": 2}]},
    }
    audit = audit_categories([img], prompt_to_cat, preds)

    assert audit["crack"]["status"] == STATUS_SUPPORTED_EVALUATED
    assert audit["crack"]["gt_instances"] == 3
    assert audit["crack"]["total_predictions"] == 1

    assert audit["floor_slab"]["status"] == STATUS_ZERO_DETECTIONS
    assert audit["floor_slab"]["total_predictions"] == 0

    assert audit["spalling"]["status"] == STATUS_UNSUPPORTED_UNMAPPED
    assert audit["spalling"]["mapped_prompts"] == []

    assert audit["rare_defect"]["status"] == STATUS_INSUFFICIENT_DATA


def test_run_error_analysis_synthetic_breakdown():
    # 1000x1000 image
    # GT 1: Tiny (w=10, h=100) -> short=10px (<32px) at x=100, y=100-200
    # GT 2: Small (w=40, h=100) -> short=40px (32-64px) at x=300, y=300-400
    # GT 3: Large (w=200, h=200) -> short=200px (>=128px) at x=600, y=600-800
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
            _make_ls_rect("r1", "crack", 10.0, 10.0, 1.0, 10.0),   # 10x100px (Tiny)
            _make_ls_rect("r2", "crack", 30.0, 30.0, 4.0, 10.0),   # 40x100px (Small)
            _make_ls_rect("r3", "crack", 60.0, 60.0, 20.0, 20.0),  # 200x200px (Large)
        ],
    )

    # Pred 1 matches GT 3 perfectly (IoU ~1.0) -> TP
    # Pred 2 overlaps GT 2 partially (x=300..340, y=320..420) -> IoU around 0.30 -> Near-miss
    # Pred 3 is away at (x=900, y=900) -> IoU 0.0 -> Hallucination FP
    preds = {
        ("sha1", "crack in wall"): {
            "boxes": [
                {"x0": 600, "y0": 600, "x1": 800, "y1": 800},  # matches GT3
                {"x0": 300, "y0": 350, "x1": 340, "y1": 500},  # partial overlap on GT2
                {"x0": 900, "y0": 900, "x1": 950, "y1": 950},  # hallucination
            ]
        }
    }

    report = run_error_analysis([img], preds, category="crack", prompt="crack in wall", iou_threshold=0.5)

    assert report.total_gt == 3
    assert report.total_predictions == 3
    assert report.tp == 1
    assert report.fp == 2
    assert report.fn == 2

    # Check size stratification
    assert report.size_stratification["tiny"]["gt_count"] == 1
    assert report.size_stratification["tiny"]["tp_count"] == 0
    assert report.size_stratification["tiny"]["missed_count"] == 1

    assert report.size_stratification["small"]["gt_count"] == 1
    assert report.size_stratification["small"]["tp_count"] == 0
    assert report.size_stratification["small"]["near_miss_count"] == 1

    assert report.size_stratification["large"]["gt_count"] == 1
    assert report.size_stratification["large"]["tp_count"] == 1

    # Check FP breakdown
    assert report.fp_breakdown["total_fp"] == 2
    assert report.fp_breakdown["counts"]["near_miss_granularity"] == 1
    assert report.fp_breakdown["counts"]["hallucination"] == 1


def test_run_error_analysis_empty_handling():
    img = LSImage(
        task_id=1,
        image_id="img1",
        relpath="img1.png",
        sha256="sha1",
        width=1000,
        height=1000,
        source="test",
        is_negative=True,
        rects=[],
    )
    preds = {}
    report = run_error_analysis([img], preds, category="crack", prompt="crack in wall")
    assert report.total_gt == 0
    assert report.total_predictions == 0
    assert report.tp == 0 and report.fp == 0 and report.fn == 0
    assert report.precision == 0.0 and report.recall == 0.0
