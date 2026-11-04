"""Unit tests for Experiment 3: Core Detection Engine Upgrade.

Verifies:
1. CrackUNetAdapter sliding-window inference with edge padding and instance extraction.
2. WonderVerifier semantic classification, rubric logic, and manifest matching.
3. Multi-scale tile coordinate mapping and cross-view suppression integrity.
4. Experiment 3 evaluation metric bounds and bipartite matching invariants.
5. Integrity and preservation of frozen Baseline A reference outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from neev.adapters.crack_unet import CrackUNetAdapter, DEFAULT_CHECKPOINT
from neev.adapters.wonder_verifier import WonderVerifier, WonderVerificationResult
from neev.eval.bipartite_matching import evaluate_detection_validity_bipartite
from neev.eval.tiles import aabb_iou, compute_tiles, merge_cross_tile, translate_box


@pytest.fixture
def unet_adapter():
    if not DEFAULT_CHECKPOINT.exists():
        pytest.skip("U-Net checkpoint not available")
    return CrackUNetAdapter(device="cpu")


@pytest.fixture
def wonder_verifier():
    return WonderVerifier()


def test_unet_sliding_window_aspect_padding(unet_adapter):
    """Test 1: U-Net sliding window handles asymmetric images where one side < patch size."""
    # Synthetic 515x388 RGB image (similar to cr1)
    img = np.full((388, 515, 3), 128, dtype=np.uint8)
    res = unet_adapter.segment(img, threshold=0.5, extract_instances=False)
    assert res.probs.shape == (388, 515)
    assert res.binary.shape == (388, 515)
    assert np.all(res.probs >= 0.0) and np.all(res.probs <= 1.0)
    assert res.tiles >= 1


def test_wonder_verifier_manifest_and_rubric(wonder_verifier):
    """Test 2: WonderVerifier correctly classifies cracks vs architectural false alarms."""
    # Synthetic crack crop: elongated with tortuous path
    im = Image.new("RGB", (200, 200), (180, 180, 180))
    res_crack = wonder_verifier.verify_crop(
        im, box={"x0": 20, "y0": 10, "x1": 50, "y1": 180},
        dsp_features={"tortuosity": 1.6, "fill_factor": 0.25, "energy": 800.0}
    )
    assert isinstance(res_crack, WonderVerificationResult)
    assert res_crack.is_crack is True
    assert res_crack.defect_type == "crack"
    assert res_crack.confidence >= 0.70

    # Synthetic window frame: straight vertical with uniform thickness
    res_mullion = wonder_verifier.verify_crop(
        im, box={"x0": 50, "y0": 0, "x1": 54, "y1": 200},
        dsp_features={"tortuosity": 1.01, "fill_factor": 0.20, "energy": 300.0}
    )
    assert res_mullion.is_crack is False
    assert res_mullion.defect_type == "architectural_edge"
    assert res_mullion.rejection_reason is not None

    # Synthetic surface stain: high fill factor
    res_stain = wonder_verifier.verify_crop(
        im, box={"x0": 30, "y0": 30, "x1": 150, "y1": 150},
        dsp_features={"tortuosity": 1.1, "fill_factor": 0.85, "energy": 200.0}
    )
    assert res_stain.is_crack is False
    assert res_stain.defect_type == "surface_stain"


def test_multiscale_tile_mapping_and_suppression():
    """Test 3: Multi-scale tile coordinate offsets and duplicate suppression."""
    tiles = compute_tiles(width=2400, height=1792, rows=2, cols=2, overlap=0.25)
    assert len(tiles) == 4
    for t in tiles:
        assert 0 <= t.x0 < t.x1 <= 2400
        assert 0 <= t.y0 < t.y1 <= 1792

    # Translate a tile box into native canvas coordinates
    local_box = {"x0": 50, "y0": 60, "x1": 120, "y1": 200}
    native_box = translate_box(local_box, dx=1000, dy=500)
    assert native_box == {"x0": 1050, "y0": 560, "x1": 1120, "y1": 700}

    # Verify duplicate cross-tile box suppression at IoU >= 0.50
    duplicate_box = {"x0": 1055, "y0": 565, "x1": 1120, "y1": 705}
    distinct_box = {"x0": 100, "y0": 100, "x1": 200, "y1": 300}
    kept, suppressed = merge_cross_tile([native_box, duplicate_box, distinct_box], iou_threshold=0.50)
    assert len(kept) == 2
    assert len(suppressed) == 1


def test_bipartite_evaluation_metric_invariants():
    """Test 4: Mathematical invariants of bipartite matching under cardinality_first."""
    gt_raster = np.zeros((100, 100), dtype=bool)
    gt_raster[20:80, 40:60] = True
    gt_poly = [(40.0, 20.0), (60.0, 20.0), (60.0, 80.0), (40.0, 80.0)]

    # Perfect prediction box
    pred_boxes = [{"x0": 38, "y0": 18, "x1": 62, "y1": 82}]
    res = evaluate_detection_validity_bipartite(
        prediction_boxes=pred_boxes,
        gt_rasters=[gt_raster],
        gt_polygons=[gt_poly],
        min_coverage_thresh=0.50,
        matching_mode="cardinality_first",
        image_width=100,
        image_height=100,
    )
    assert res.total_predictions == 1
    assert res.total_ground_truths == 1
    assert res.true_positives == 1
    assert res.false_positives == 0
    assert res.false_negatives == 0
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1 == 1.0


def test_frozen_baseline_provenance_preservation():
    """Test 5: Verify that Experiment 2.3 baseline records remain intact and immutable."""
    val_res_path = Path("runs/experiments/experiment_2_3_validation/validation_results.json")
    assert val_res_path.exists(), "Experiment 2.3 validation results missing"
    data = json.loads(val_res_path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    assert meta.get("actual_spend_usd") == 0.03
    assert meta.get("live_api_calls_made") == 12

    # Verify that Experiment 3 results were exported properly
    exp3_res_path = Path("runs/experiments/experiment_3_engine_upgrade/experiment_3_results.json")
    assert exp3_res_path.exists(), "Experiment 3 results missing"
    exp3_data = json.loads(exp3_res_path.read_text(encoding="utf-8"))
    assert "development_benchmark" in exp3_data
    assert "validation_benchmark" in exp3_data
