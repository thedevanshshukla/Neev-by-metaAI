"""Unit tests for CUBIT-Seg matching, mask metrics, and evaluation integrity."""

import numpy as np
import pytest

from neev.eval.bipartite_matching import (
    evaluate_detection_validity_bipartite,
    DetectionEvaluationResult,
)


def create_box_mask_and_poly(w: int, h: int, box: list[int]):
    m = np.zeros((h, w), dtype=bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    poly = [(float(x0), float(y0)), (float(x1), float(y0)), (float(x1), float(y1)), (float(x0), float(y1))]
    return m, poly


def test_perfect_match():
    w, h = 512, 512
    gt_box = [100, 100, 200, 200]
    pred_box = {"x0": 100, "y0": 100, "x1": 200, "y1": 200, "confidence": 0.9}
    
    m, p = create_box_mask_and_poly(w, h, gt_box)
    gt_rasters = [m]
    gt_polys = [p]
    preds = [pred_box]
    
    res = evaluate_detection_validity_bipartite(
        prediction_boxes=preds,
        gt_rasters=gt_rasters,
        gt_polygons=gt_polys,
        min_coverage_thresh=0.50,
        matching_mode="cardinality_first",
        image_width=w,
        image_height=h,
    )
    
    assert res.total_predictions == 1
    assert res.total_ground_truths == 1
    assert res.true_positives == 1
    assert res.false_positives == 0
    assert res.false_negatives == 0
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1 == 1.0


def test_duplicate_prediction_rejection():
    w, h = 512, 512
    gt_box = [100, 100, 200, 200]
    pred1 = {"x0": 100, "y0": 100, "x1": 200, "y1": 200, "confidence": 0.95}
    pred2 = {"x0": 105, "y0": 105, "x1": 205, "y1": 205, "confidence": 0.85}
    
    m, p = create_box_mask_and_poly(w, h, gt_box)
    gt_rasters = [m]
    gt_polys = [p]
    preds = [pred1, pred2]
    
    res = evaluate_detection_validity_bipartite(
        prediction_boxes=preds,
        gt_rasters=gt_rasters,
        gt_polygons=gt_polys,
        min_coverage_thresh=0.50,
        matching_mode="cardinality_first",
        image_width=w,
        image_height=h,
    )
    
    assert res.total_predictions == 2
    assert res.total_ground_truths == 1
    assert res.true_positives == 1
    assert res.false_positives == 1  # Duplicate becomes a false positive!
    assert res.false_negatives == 0
    assert res.precision == 0.5
    assert res.recall == 1.0


def test_negative_image_clean():
    w, h = 512, 512
    res = evaluate_detection_validity_bipartite(
        prediction_boxes=[],
        gt_rasters=[],
        gt_polygons=[],
        min_coverage_thresh=0.50,
        matching_mode="cardinality_first",
        image_width=w,
        image_height=h,
    )
    
    assert res.total_predictions == 0
    assert res.total_ground_truths == 0
    assert res.true_positives == 0
    assert res.false_positives == 0
    assert res.false_negatives == 0
    assert res.precision == 1.0
    assert res.recall == 1.0


def test_negative_image_with_false_alarms():
    w, h = 512, 512
    preds = [
        {"x0": 50, "y0": 50, "x1": 150, "y1": 150, "confidence": 0.8},
        {"x0": 200, "y0": 200, "x1": 300, "y1": 300, "confidence": 0.7},
    ]
    
    res = evaluate_detection_validity_bipartite(
        prediction_boxes=preds,
        gt_rasters=[],
        gt_polygons=[],
        min_coverage_thresh=0.50,
        matching_mode="cardinality_first",
        image_width=w,
        image_height=h,
    )
    
    assert res.total_predictions == 2
    assert res.total_ground_truths == 0
    assert res.true_positives == 0
    assert res.false_positives == 2
    assert res.precision == 0.0
    assert res.recall == 1.0



def test_multiple_independent_cracks():
    w, h = 512, 512
    gt1 = [20, 20, 80, 80]
    gt2 = [150, 150, 220, 220]
    gt3 = [350, 350, 450, 450]
    
    pred1 = {"x0": 20, "y0": 20, "x1": 80, "y1": 80, "confidence": 0.9}
    pred2 = {"x0": 150, "y0": 150, "x1": 220, "y1": 220, "confidence": 0.9}
    
    m1, p1 = create_box_mask_and_poly(w, h, gt1)
    m2, p2 = create_box_mask_and_poly(w, h, gt2)
    m3, p3 = create_box_mask_and_poly(w, h, gt3)
    
    gt_rasters = [m1, m2, m3]
    gt_polys = [p1, p2, p3]
    preds = [pred1, pred2]
    
    res = evaluate_detection_validity_bipartite(
        prediction_boxes=preds,
        gt_rasters=gt_rasters,
        gt_polygons=gt_polys,
        min_coverage_thresh=0.50,
        matching_mode="cardinality_first",
        image_width=w,
        image_height=h,
    )
    
    assert res.total_predictions == 2
    assert res.total_ground_truths == 3
    assert res.true_positives == 2
    assert res.false_positives == 0
    assert res.false_negatives == 1
    assert res.precision == 1.0
    assert round(res.recall, 3) == round(2 / 3, 3)


def test_oversized_box_cannot_claim_multiple_cracks():
    w, h = 512, 512
    gt1 = [20, 20, 50, 50]
    gt2 = [100, 100, 130, 130]
    gt3 = [200, 200, 230, 230]
    
    # One giant box covering all three
    pred_giant = {"x0": 10, "y0": 10, "x1": 250, "y1": 250, "confidence": 0.99}
    
    m1, p1 = create_box_mask_and_poly(w, h, gt1)
    m2, p2 = create_box_mask_and_poly(w, h, gt2)
    m3, p3 = create_box_mask_and_poly(w, h, gt3)
    
    gt_rasters = [m1, m2, m3]
    gt_polys = [p1, p2, p3]
    preds = [pred_giant]
    
    res = evaluate_detection_validity_bipartite(
        prediction_boxes=preds,
        gt_rasters=gt_rasters,
        gt_polygons=gt_polys,
        min_coverage_thresh=0.50,
        matching_mode="cardinality_first",
        image_width=w,
        image_height=h,
    )
    
    assert res.total_predictions == 1
    assert res.total_ground_truths == 3
    # In one-to-one matching, exactly ONE GT is matched; the other 2 remain false negatives!
    assert res.true_positives == 1
    assert res.false_negatives == 2
    assert res.false_positives == 0
