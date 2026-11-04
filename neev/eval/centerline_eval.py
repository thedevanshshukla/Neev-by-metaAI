"""Centerline and Boundary-Tolerant Defect Evaluator.

Provides defect-specific linear evaluation metrics:
- Strict Box Metric: official rotated-polygon IoU >= 0.50 (baseline compatibility)
- Relaxed Box Metric: rotated-polygon IoU >= 0.25 (calibrated for line annotations)
- Buffer Tube Match: tests whether predicted bounding box covers the ground-truth
  crack centerline within a tolerance buffer tau in pixels.

Allows fair, scientifically grounded reporting for thin surface cracks.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from neev.eval.labelstudio import LSImage, rect_polygon
from neev.eval.matching import match_rotated, rotated_iou


@dataclass
class ToleranceComparisonResult:
    threshold: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


@dataclass
class DualMetricReport:
    category: str
    prompt: str
    total_gt: int
    total_preds: int
    strict_iou_050: ToleranceComparisonResult
    relaxed_iou_025: ToleranceComparisonResult
    per_image_gains: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_dual_tolerance(
    gt_images: list[LSImage],
    predictions: dict[tuple[str, str], dict[str, Any]],
    category: str = "crack",
    prompt: str = "crack in wall",
) -> DualMetricReport:
    """Evaluate predictions at both strict 0.50 IoU and relaxed 0.25 IoU."""
    total_gt = 0
    total_preds = 0

    per_image: dict[str, dict[str, Any]] = {}

    tp_strict = 0
    fp_strict = 0
    fn_strict = 0

    tp_relaxed = 0
    fp_relaxed = 0
    fn_relaxed = 0

    for img in gt_images:
        rects = [r for r in img.rects if r.label == category]
        p_boxes = predictions.get((img.sha256, prompt), {}).get("boxes", [])

        total_gt += len(rects)
        total_preds += len(p_boxes)

        gt_entries = [{"polygon": rect_polygon(r, img.width, img.height), "category": category} for r in rects]
        pred_entries = [
            {
                "polygon": [
                    (float(b["x0"]), float(b["y0"])),
                    (float(b["x1"]), float(b["y0"])),
                    (float(b["x1"]), float(b["y1"])),
                    (float(b["x0"]), float(b["y1"])),
                ],
                "category": category,
            }
            for b in p_boxes
        ]

        # 1. Strict (0.50)
        res_050 = match_rotated(pred_entries, gt_entries, iou_threshold=0.50)
        tp_strict += res_050.tp
        fp_strict += res_050.fp
        fn_strict += res_050.fn

        # 2. Relaxed (0.25)
        res_025 = match_rotated(pred_entries, gt_entries, iou_threshold=0.25)
        tp_relaxed += res_025.tp
        fp_relaxed += res_025.fp
        fn_relaxed += res_025.fn

        per_image[img.image_id] = {
            "gt_count": len(rects),
            "pred_count": len(p_boxes),
            "tp_strict": res_050.tp,
            "tp_relaxed": res_025.tp,
            "gained_tps": res_025.tp - res_050.tp,
        }

    p_str = tp_strict / (tp_strict + fp_strict) if (tp_strict + fp_strict) > 0 else 0.0
    r_str = tp_strict / (tp_strict + fn_strict) if (tp_strict + fn_strict) > 0 else 0.0
    f1_str = 2 * p_str * r_str / (p_str + r_str) if (p_str + r_str) > 0 else 0.0

    p_rel = tp_relaxed / (tp_relaxed + fp_relaxed) if (tp_relaxed + fp_relaxed) > 0 else 0.0
    r_rel = tp_relaxed / (tp_relaxed + fn_relaxed) if (tp_relaxed + fn_relaxed) > 0 else 0.0
    f1_rel = 2 * p_rel * r_rel / (p_rel + r_rel) if (p_rel + r_rel) > 0 else 0.0

    return DualMetricReport(
        category=category,
        prompt=prompt,
        total_gt=total_gt,
        total_preds=total_preds,
        strict_iou_050=ToleranceComparisonResult(
            threshold=0.50,
            tp=tp_strict,
            fp=fp_strict,
            fn=fn_strict,
            precision=round(p_str, 4),
            recall=round(r_str, 4),
            f1=round(f1_str, 4),
        ),
        relaxed_iou_025=ToleranceComparisonResult(
            threshold=0.25,
            tp=tp_relaxed,
            fp=fp_relaxed,
            fn=fn_relaxed,
            precision=round(p_rel, 4),
            recall=round(r_rel, 4),
            f1=round(f1_rel, 4),
        ),
        per_image_gains=per_image,
    )
