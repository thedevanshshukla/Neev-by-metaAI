"""Offline error analysis and defect-size stratification engine.

Pure, deterministic diagnostics for defect detection failures.
Turns raw evaluation outputs into actionable engineering classifications:
- Size-stratified recall: Tiny (<32px), Small (32-64px), Medium (64-128px), Large (>=128px)
- Failure mode taxonomy: Missed defects vs. localization/granularity near-misses vs. duplicates
- False-positive audit: Real-defect near-misses (0.10 <= IoU < 0.50) vs. hallucinations (IoU = 0)
- Category registry: Disentangles detection failures from unmapped/unsupported evaluation coverage.

No network calls, no model execution, no writes to protected benchmark artifacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neev.eval.labelstudio import LSImage, LSRect, rect_polygon
from neev.eval.matching import match_rotated, polygon_area, rotated_iou

# Standard size boundaries on shorter side (px)
SIZE_BINS = {
    "tiny": (0.0, 32.0),
    "small": (32.0, 64.0),
    "medium": (64.0, 128.0),
    "large": (128.0, float("inf")),
}

# Category audit classifications
STATUS_SUPPORTED_EVALUATED = "supported_evaluated"
STATUS_ZERO_DETECTIONS = "zero_detections"
STATUS_UNSUPPORTED_UNMAPPED = "unsupported_unmapped"
STATUS_INSUFFICIENT_DATA = "insufficient_data"


def classify_size(short_side_px: float) -> str:
    """Classify defect by shorter side dimension."""
    for name, (low, high) in SIZE_BINS.items():
        if low <= short_side_px < high:
            return name
    return "large"


@dataclass(frozen=True)
class GTDefectRecord:
    image_id: str
    result_id: str
    short_side_px: float
    long_side_px: float
    area_px: float
    area_frac: float
    size_bin: str
    rotation_deg: float
    matched: bool
    best_iou: float
    best_pred_index: int | None
    failure_mode: str  # "true_positive" | "near_miss_granularity" | "missed_defect"


@dataclass(frozen=True)
class FPRecord:
    image_id: str
    pred_index: int
    width_px: float
    height_px: float
    area_px: float
    best_iou: float
    fp_type: str  # "near_miss_granularity" (0.10<=IoU<0.50) | "marginal" (0<IoU<0.10) | "hallucination" (IoU==0)


@dataclass
class SizeStratificationSummary:
    gt_count: int = 0
    tp_count: int = 0
    recall: float = 0.0
    near_miss_count: int = 0  # 0.10 <= IoU < 0.50
    missed_count: int = 0     # IoU < 0.10


@dataclass
class ErrorAnalysisReport:
    category: str
    prompt: str
    total_gt: int
    total_predictions: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    size_stratification: dict[str, dict[str, Any]]
    iou_distribution: dict[str, int]
    fp_breakdown: dict[str, Any]
    gt_records: list[dict[str, Any]] = field(default_factory=list)
    fp_records: list[dict[str, Any]] = field(default_factory=list)
    image_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_categories(
    gt_images: list[LSImage],
    prompt_to_category: dict[str, str],
    predictions: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Audit all ground-truth categories: supported vs unmapped vs zero-detection."""
    # Count ground-truth instances per label
    gt_counts: dict[str, int] = {}
    for img in gt_images:
        for rect in img.rects:
            gt_counts[rect.label] = gt_counts.get(rect.label, 0) + 1

    # Invert mapping: category -> list of prompts
    cat_to_prompts: dict[str, list[str]] = {}
    for p, c in prompt_to_category.items():
        cat_to_prompts.setdefault(c, []).append(p)

    out: dict[str, dict[str, Any]] = {}
    all_categories = sorted(set(gt_counts.keys()) | set(cat_to_prompts.keys()))

    for cat in all_categories:
        count = gt_counts.get(cat, 0)
        prompts = cat_to_prompts.get(cat, [])
        if not prompts:
            status = STATUS_UNSUPPORTED_UNMAPPED
            total_preds = 0
        else:
            # Check predictions across images for these prompts
            total_preds = 0
            for img in gt_images:
                for p in prompts:
                    boxes = predictions.get((img.sha256, p), {}).get("boxes", [])
                    total_preds += len(boxes)
            if count < 3:
                status = STATUS_INSUFFICIENT_DATA
            elif total_preds == 0:
                status = STATUS_ZERO_DETECTIONS
            else:
                status = STATUS_SUPPORTED_EVALUATED

        out[cat] = {
            "category": cat,
            "status": status,
            "gt_instances": count,
            "mapped_prompts": prompts,
            "total_predictions": total_preds,
        }
    return out


def run_error_analysis(
    gt_images: list[LSImage],
    predictions: dict[tuple[str, str], dict[str, Any]],
    category: str = "crack",
    prompt: str = "crack in wall",
    iou_threshold: float = 0.5,
) -> ErrorAnalysisReport:
    """Perform fine-grained error analysis for a target category and prompt."""
    gt_records: list[GTDefectRecord] = []
    fp_records: list[FPRecord] = []
    image_breakdown: dict[str, dict[str, Any]] = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_preds = 0

    iou_hist = {
        "0.0": 0,
        "0.0_0.10": 0,
        "0.10_0.25": 0,
        "0.25_0.50": 0,
        "0.50_0.75": 0,
        "0.75_1.0": 0,
    }

    def record_iou(score: float):
        if score <= 0.0:
            iou_hist["0.0"] += 1
        elif score < 0.10:
            iou_hist["0.0_0.10"] += 1
        elif score < 0.25:
            iou_hist["0.10_0.25"] += 1
        elif score < 0.50:
            iou_hist["0.25_0.50"] += 1
        elif score < 0.75:
            iou_hist["0.50_0.75"] += 1
        else:
            iou_hist["0.75_1.0"] += 1

    strat_map: dict[str, SizeStratificationSummary] = {
        name: SizeStratificationSummary() for name in SIZE_BINS
    }

    for img in gt_images:
        rects = [r for r in img.rects if r.label == category]
        gt_polys = []
        for r in rects:
            poly = rect_polygon(r, img.width, img.height)
            gt_polys.append(poly)

        p_boxes = predictions.get((img.sha256, prompt), {}).get("boxes", [])
        pred_polys = []
        for b in p_boxes:
            pred_polys.append([
                (float(b["x0"]), float(b["y0"])),
                (float(b["x1"]), float(b["y0"])),
                (float(b["x1"]), float(b["y1"])),
                (float(b["x0"]), float(b["y1"])),
            ])

        total_gt += len(rects)
        total_preds += len(p_boxes)

        # Rotated matching using official matching module
        gt_entries = [{"polygon": p, "category": category} for p in gt_polys]
        pred_entries = [{"polygon": p, "category": category} for p in pred_polys]

        if gt_entries and pred_entries:
            match_res = match_rotated(pred_entries, gt_entries, iou_threshold=iou_threshold)
            matched_gt_indices = {g_idx: (p_idx, score) for p_idx, g_idx, score in match_res.matches}
            matched_pred_indices = {p_idx for p_idx, _, _ in match_res.matches}
            img_tp = match_res.tp
            img_fp = match_res.fp
            img_fn = match_res.fn
        else:
            matched_gt_indices = {}
            matched_pred_indices = set()
            img_tp = 0
            img_fp = len(p_boxes)
            img_fn = len(rects)

        total_tp += img_tp
        total_fp += img_fp
        total_fn += img_fn

        # Detailed GT analysis
        for g_idx, (r, g_poly) in enumerate(zip(rects, gt_polys)):
            w_px = r.w_pct / 100 * img.width
            h_px = r.h_pct / 100 * img.height
            short_s = min(w_px, h_px)
            long_s = max(w_px, h_px)
            area = polygon_area(g_poly)
            area_frac = area / (img.width * img.height) if img.width * img.height > 0 else 0.0
            size_bin = classify_size(short_s)

            # Find best overlap against any candidate prediction
            best_iou = 0.0
            best_p_idx = None
            for p_idx, p_poly in enumerate(pred_polys):
                score = rotated_iou(p_poly, g_poly)
                if score > best_iou:
                    best_iou = score
                    best_p_idx = p_idx

            record_iou(best_iou)

            matched = g_idx in matched_gt_indices
            if matched:
                fail_mode = "true_positive"
                strat_map[size_bin].tp_count += 1
            elif best_iou >= 0.10:
                fail_mode = "near_miss_granularity"
                strat_map[size_bin].near_miss_count += 1
            else:
                fail_mode = "missed_defect"
                strat_map[size_bin].missed_count += 1

            strat_map[size_bin].gt_count += 1

            gt_records.append(GTDefectRecord(
                image_id=img.image_id,
                result_id=r.result_id,
                short_side_px=round(short_s, 2),
                long_side_px=round(long_s, 2),
                area_px=round(area, 1),
                area_frac=round(area_frac, 6),
                size_bin=size_bin,
                rotation_deg=round(r.rotation_deg, 1),
                matched=matched,
                best_iou=round(best_iou, 4),
                best_pred_index=best_p_idx,
                failure_mode=fail_mode,
            ))

        # Detailed FP analysis
        for p_idx, (b, p_poly) in enumerate(zip(p_boxes, pred_polys)):
            if p_idx in matched_pred_indices:
                continue
            best_iou = max((rotated_iou(p_poly, g) for g in gt_polys), default=0.0)
            pw = float(b["x1"] - b["x0"])
            ph = float(b["y1"] - b["y0"])
            p_area = pw * ph

            if best_iou >= 0.10:
                fp_type = "near_miss_granularity"
            elif best_iou > 0.0:
                fp_type = "marginal"
            else:
                fp_type = "hallucination"

            fp_records.append(FPRecord(
                image_id=img.image_id,
                pred_index=p_idx,
                width_px=pw,
                height_px=ph,
                area_px=p_area,
                best_iou=round(best_iou, 4),
                fp_type=fp_type,
            ))

        image_breakdown[img.image_id] = {
            "width": img.width,
            "height": img.height,
            "gt_count": len(rects),
            "pred_count": len(p_boxes),
            "tp": img_tp,
            "fp": img_fp,
            "fn": img_fn,
        }

    # Finalize size stratification metrics
    size_strat_out: dict[str, dict[str, Any]] = {}
    for name, s in strat_map.items():
        rec = s.tp_count / s.gt_count if s.gt_count > 0 else 0.0
        size_strat_out[name] = {
            "gt_count": s.gt_count,
            "tp_count": s.tp_count,
            "recall": round(rec, 4),
            "near_miss_count": s.near_miss_count,
            "missed_count": s.missed_count,
        }

    # Summarize FPs
    fp_counts = {
        "near_miss_granularity": sum(1 for f in fp_records if f.fp_type == "near_miss_granularity"),
        "marginal": sum(1 for f in fp_records if f.fp_type == "marginal"),
        "hallucination": sum(1 for f in fp_records if f.fp_type == "hallucination"),
    }

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return ErrorAnalysisReport(
        category=category,
        prompt=prompt,
        total_gt=total_gt,
        total_predictions=total_preds,
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        size_stratification=size_strat_out,
        iou_distribution=iou_hist,
        fp_breakdown={
            "total_fp": len(fp_records),
            "counts": fp_counts,
            "near_miss_pct": round(fp_counts["near_miss_granularity"] / len(fp_records) * 100, 1) if fp_records else 0.0,
            "hallucination_pct": round(fp_counts["hallucination"] / len(fp_records) * 100, 1) if fp_records else 0.0,
        },
        gt_records=[asdict(r) for r in gt_records],
        fp_records=[asdict(f) for f in fp_records],
        image_breakdown=image_breakdown,
    )
