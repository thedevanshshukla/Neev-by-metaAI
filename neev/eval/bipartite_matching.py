"""Bipartite matching and detection-level evaluation framework.

Implements rigorous one-to-one prediction-to-ground-truth assignment via
maximum weight bipartite matching (Hungarian algorithm), accounting for
duplicate predictions, false positives, false negatives, precision, recall,
and detection-level F1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np
from scipy.optimize import linear_sum_assignment

from neev.eval.matching import rotated_iou


@dataclass(frozen=True)
class MatchedPair:
    pred_index: int
    gt_index: int
    score: float
    iou: float
    mask_cov: float = 0.0
    mask_iou: float = 0.0


@dataclass
class DetectionEvaluationResult:
    total_predictions: int
    total_ground_truths: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    unconstrained_covered_gts: int
    unconstrained_coverage_pct: float
    one_to_one_matched_gts: int
    one_to_one_recall_pct: float
    inflation_factor: float
    matching_mode: str = "cardinality_first"
    unconstrained_cov_25: int = 0
    unconstrained_cov_50: int = 0
    unconstrained_cov_75: int = 0
    unconstrained_cov_90: int = 0
    near_whole_image_predictions_count: int = 0
    total_pred_area_pct_of_image: float = 0.0
    iou_ge_010: int = 0
    iou_ge_025: int = 0
    iou_ge_050: int = 0
    mean_box_area: float = 0.0
    mean_proposal_to_gt_ratio: float = 0.0
    matched_pairs: list[MatchedPair] = field(default_factory=list)
    unmatched_predictions: list[int] = field(default_factory=list)
    unmatched_ground_truths: list[int] = field(default_factory=list)
    multi_claim_predictions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_predictions": self.total_predictions,
            "total_ground_truths": self.total_ground_truths,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "unconstrained_covered_gts": self.unconstrained_covered_gts,
            "unconstrained_coverage_pct": round(self.unconstrained_coverage_pct, 1),
            "one_to_one_matched_gts": self.one_to_one_matched_gts,
            "one_to_one_recall_pct": round(self.one_to_one_recall_pct, 1),
            "inflation_factor": round(self.inflation_factor, 2),
            "matching_mode": self.matching_mode,
            "unconstrained_cov_25": self.unconstrained_cov_25,
            "unconstrained_cov_50": self.unconstrained_cov_50,
            "unconstrained_cov_75": self.unconstrained_cov_75,
            "unconstrained_cov_90": self.unconstrained_cov_90,
            "near_whole_image_predictions_count": self.near_whole_image_predictions_count,
            "total_pred_area_pct_of_image": round(self.total_pred_area_pct_of_image, 2),
            "iou_ge_010": self.iou_ge_010,
            "iou_ge_025": self.iou_ge_025,
            "iou_ge_050": self.iou_ge_050,
            "mean_box_area": round(self.mean_box_area, 1),
            "mean_proposal_to_gt_ratio": round(self.mean_proposal_to_gt_ratio, 1),
            "matched_pairs": [
                {
                    "pred_index": p.pred_index,
                    "gt_index": p.gt_index,
                    "score": round(p.score, 4),
                    "iou": round(p.iou, 4),
                    "mask_cov": round(p.mask_cov, 4),
                    "mask_iou": round(p.mask_iou, 4),
                }
                for p in self.matched_pairs
            ],
            "unmatched_predictions": self.unmatched_predictions,
            "unmatched_ground_truths": self.unmatched_ground_truths,
            "multi_claim_predictions": self.multi_claim_predictions,
        }


def evaluate_detection_validity_bipartite(
    prediction_boxes: Sequence[dict[str, int]],
    gt_rasters: Sequence[np.ndarray],
    gt_polygons: Sequence[list[tuple[float, float]]],
    min_coverage_thresh: float = 0.50,
    pred_masks: Sequence[np.ndarray] | None = None,
    matching_mode: str = "cardinality_first",
    image_width: int | None = None,
    image_height: int | None = None,
) -> DetectionEvaluationResult:
    """Evaluate detection validity with threshold-constrained maximum-cardinality bipartite matching.

    Parameters:
    - prediction_boxes: list of bounding boxes {'x0', 'y0', 'x1', 'y1'}
    - gt_rasters: list of 2D boolean masks for each ground-truth object
    - gt_polygons: list of polygon coordinate vertices for each ground-truth object
    - min_coverage_thresh: overlap threshold for eligibility (default: 0.50)
    - pred_masks: optional list of full-canvas decoded binary masks for prediction objects
    - matching_mode: 'cardinality_first' (default, guarantees maximum valid matches) or
                     'weight_sum' (historical unconstrained sum maximization)
    - image_width, image_height: optional dimensions to evaluate near-whole-image prediction metrics
    """
    n_preds = len(prediction_boxes)
    n_gts = len(gt_rasters)

    img_area = (image_width * image_height) if (image_width and image_height) else None

    if n_preds == 0 and n_gts == 0:
        return DetectionEvaluationResult(
            total_predictions=0,
            total_ground_truths=0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            unconstrained_covered_gts=0,
            unconstrained_coverage_pct=100.0,
            one_to_one_matched_gts=0,
            one_to_one_recall_pct=100.0,
            inflation_factor=1.0,
            matching_mode=matching_mode,
        )

    if n_preds == 0:
        return DetectionEvaluationResult(
            total_predictions=0,
            total_ground_truths=n_gts,
            true_positives=0,
            false_positives=0,
            false_negatives=n_gts,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            unconstrained_covered_gts=0,
            unconstrained_coverage_pct=0.0,
            one_to_one_matched_gts=0,
            one_to_one_recall_pct=0.0,
            inflation_factor=1.0,
            matching_mode=matching_mode,
            unmatched_ground_truths=list(range(n_gts)),
        )

    # Compute prediction areas and whole-image stats
    pred_areas = []
    near_whole_img_count = 0
    total_pred_area = 0
    for pb in prediction_boxes:
        bw = max(1, pb["x1"] - pb["x0"])
        bh = max(1, pb["y1"] - pb["y0"])
        p_area = bw * bh
        pred_areas.append(p_area)
        total_pred_area += p_area
        if img_area and (p_area / img_area >= 0.50):
            near_whole_img_count += 1

    total_pred_area_ratio = (total_pred_area / img_area * 100.0) if img_area else 0.0

    if n_gts == 0:
        mean_area = float(np.mean(pred_areas)) if pred_areas else 0.0
        return DetectionEvaluationResult(
            total_predictions=n_preds,
            total_ground_truths=0,
            true_positives=0,
            false_positives=n_preds,
            false_negatives=0,
            precision=0.0,
            recall=1.0,
            f1=0.0,
            unconstrained_covered_gts=0,
            unconstrained_coverage_pct=100.0,
            one_to_one_matched_gts=0,
            one_to_one_recall_pct=100.0,
            inflation_factor=1.0,
            matching_mode=matching_mode,
            mean_box_area=mean_area,
            near_whole_image_predictions_count=near_whole_img_count,
            total_pred_area_pct_of_image=total_pred_area_ratio,
            unmatched_predictions=list(range(n_preds)),
        )

    # 1. Compute Coverage Matrix, IoU Matrix, and Mask Overlap Matrices
    cov_matrix = np.zeros((n_preds, n_gts), dtype=float)
    iou_matrix = np.zeros((n_preds, n_gts), dtype=float)
    mask_cov_matrix = np.zeros((n_preds, n_gts), dtype=float)
    mask_iou_matrix = np.zeros((n_preds, n_gts), dtype=float)

    for i, pb in enumerate(prediction_boxes):
        p_poly = [
            (float(pb["x0"]), float(pb["y0"])),
            (float(pb["x1"]), float(pb["y0"])),
            (float(pb["x1"]), float(pb["y1"])),
            (float(pb["x0"]), float(pb["y1"])),
        ]
        p_mask = pred_masks[i] if (pred_masks is not None and i < len(pred_masks)) else None

        for j in range(n_gts):
            g_arr = gt_rasters[j]
            g_area = max(1, int(g_arr.sum()))
            cov_px = int(g_arr[pb["y0"]:pb["y1"], pb["x0"]:pb["x1"]].sum())
            cov_matrix[i, j] = cov_px / g_area
            iou_matrix[i, j] = rotated_iou(p_poly, gt_polygons[j])

            if p_mask is not None:
                mask_inter = int((p_mask & g_arr).sum())
                mask_union = int((p_mask | g_arr).sum())
                mask_cov_matrix[i, j] = mask_inter / g_area
                mask_iou_matrix[i, j] = mask_inter / max(1, mask_union)

    # 2. Historical Unconstrained Coverage at multiple thresholds
    max_cov_per_gt = cov_matrix.max(axis=0)
    cov_25 = int((max_cov_per_gt >= 0.25).sum())
    cov_50 = int((max_cov_per_gt >= 0.50).sum())
    cov_75 = int((max_cov_per_gt >= 0.75).sum())
    cov_90 = int((max_cov_per_gt >= 0.90).sum())

    unconstrained_covered = int((max_cov_per_gt >= min_coverage_thresh).sum())
    unconstrained_pct = (unconstrained_covered / n_gts) * 100.0

    # 3. Detect Multi-Claim Predictions (Single prediction covering multiple GT networks)
    multi_claims = []
    for i in range(n_preds):
        covered_gt_indices = [int(j) for j in range(n_gts) if cov_matrix[i, j] >= min_coverage_thresh]
        if len(covered_gt_indices) > 1:
            multi_claims.append({
                "pred_index": i,
                "box": prediction_boxes[i],
                "claimed_gt_count": len(covered_gt_indices),
                "claimed_gt_indices": covered_gt_indices,
            })

    # 4. Maximum Bipartite Matching
    if matching_mode == "cardinality_first":
        # Threshold-constrained maximum-cardinality matching:
        # Choose K sufficiently large so matching an additional eligible edge strictly beats any weight sum
        K = max(n_preds, n_gts) + 10.0
        # Transformed weight matrix: Eligible edges get K + coverage, ineligible get 0.0
        W = np.where(cov_matrix >= min_coverage_thresh, K + cov_matrix, 0.0)
        row_ind, col_ind = linear_sum_assignment(-W)
    elif matching_mode == "weight_sum":
        # Historical unconstrained weight-sum maximization
        row_ind, col_ind = linear_sum_assignment(-cov_matrix)
    else:
        raise ValueError(f"Unknown matching_mode '{matching_mode}'. Must be 'cardinality_first' or 'weight_sum'.")

    matched_pairs: list[MatchedPair] = []
    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    area_ratios = []

    for r, c in zip(row_ind, col_ind):
        cov = cov_matrix[r, c]
        if cov >= min_coverage_thresh:
            iou_val = float(iou_matrix[r, c])
            m_cov = float(mask_cov_matrix[r, c])
            m_iou = float(mask_iou_matrix[r, c])
            matched_pairs.append(
                MatchedPair(
                    pred_index=int(r),
                    gt_index=int(c),
                    score=float(cov),
                    iou=iou_val,
                    mask_cov=m_cov,
                    mask_iou=m_iou,
                )
            )
            matched_preds.add(int(r))
            matched_gts.add(int(c))
            gt_area = max(1, int(gt_rasters[c].sum()))
            area_ratios.append(pred_areas[r] / gt_area)

    tp = len(matched_pairs)
    fp = n_preds - tp
    fn = n_gts - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    one_to_one_pct = (tp / n_gts) * 100.0
    inflation = unconstrained_covered / max(1, tp)

    # IoU metrics across matched pairs
    iou_010 = sum(1 for p in matched_pairs if p.iou >= 0.10)
    iou_025 = sum(1 for p in matched_pairs if p.iou >= 0.25)
    iou_050 = sum(1 for p in matched_pairs if p.iou >= 0.50)

    unmatched_preds = [i for i in range(n_preds) if i not in matched_preds]
    unmatched_gts = [j for j in range(n_gts) if j not in matched_gts]

    mean_box_area = float(np.mean(pred_areas)) if pred_areas else 0.0
    mean_ratio = float(np.mean(area_ratios)) if area_ratios else 0.0

    return DetectionEvaluationResult(
        total_predictions=n_preds,
        total_ground_truths=n_gts,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        unconstrained_covered_gts=unconstrained_covered,
        unconstrained_coverage_pct=unconstrained_pct,
        one_to_one_matched_gts=tp,
        one_to_one_recall_pct=one_to_one_pct,
        inflation_factor=inflation,
        matching_mode=matching_mode,
        unconstrained_cov_25=cov_25,
        unconstrained_cov_50=cov_50,
        unconstrained_cov_75=cov_75,
        unconstrained_cov_90=cov_90,
        near_whole_image_predictions_count=near_whole_img_count,
        total_pred_area_pct_of_image=total_pred_area_ratio,
        iou_ge_010=iou_010,
        iou_ge_025=iou_025,
        iou_ge_050=iou_050,
        mean_box_area=mean_box_area,
        mean_proposal_to_gt_ratio=mean_ratio,
        matched_pairs=matched_pairs,
        unmatched_predictions=unmatched_preds,
        unmatched_ground_truths=unmatched_gts,
        multi_claim_predictions=multi_claims,
    )
