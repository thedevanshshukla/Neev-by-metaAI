"""Offline Error Audit CLI: Detailed diagnosis of detection failures.

Analyzes cached predictions against Label Studio ground truth:
- Category coverage audit (supported vs unsupported vs zero-detection)
- Size-stratified defect recall for cracks
- False-positive root causes (near-miss granularity vs hallucination)
- Per-image defect breakdown and IoU margin distribution

Usage:
  python live/error_audit.py
  python live/error_audit.py --prompt crack
  python live/error_audit.py --out runs/crack_error_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neev.eval.compare import PROMPT_TO_CATEGORY, collect_predictions  # noqa: E402
from neev.eval.error_analysis import audit_categories, run_error_analysis  # noqa: E402
from neev.eval.labelstudio import load_export, parse_task  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", default="eval/annotations_label_studio.json")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--category", default="crack")
    parser.add_argument("--prompt", default="crack in wall")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    tasks = load_export(args.export)
    gt_images = []
    for task in tasks:
        image, errors = parse_task(task, args.dataset)
        if not errors and image is not None:
            gt_images.append(image)

    predictions = collect_predictions(args.runs_dir)

    print("=" * 70)
    print("NEEV DEFECT DETECTION ENGINE: ERROR AUDIT & DIAGNOSTIC REPORT")
    print("=" * 70)

    # 1. Category Registry Audit
    cat_audit = audit_categories(gt_images, PROMPT_TO_CATEGORY, predictions)
    print("\n[1] CATEGORY COVERAGE & DETECTION AUDIT:")
    print(f"{'Category':<22} | {'Status':<22} | {'GT Count':<8} | {'Preds':<8} | {'Prompts'}")
    print("-" * 75)
    for cat, info in cat_audit.items():
        print(f"{cat:<22} | {info['status']:<22} | {info['gt_instances']:<8} | {info['total_predictions']:<8} | {', '.join(info['mapped_prompts']) or 'NONE'}")

    # 2. Defect Error Analysis for Target Category
    report = run_error_analysis(
        gt_images,
        predictions,
        category=args.category,
        prompt=args.prompt,
        iou_threshold=args.iou,
    )

    print(f"\n[2] {args.category.upper()} DETECTION SUMMARY (Prompt: '{args.prompt}', IoU >= {args.iou}):")
    print(f"  Ground Truth: {report.total_gt} | Predictions: {report.total_predictions}")
    print(f"  TP: {report.tp} | FP: {report.fp} | FN: {report.fn}")
    print(f"  Precision: {report.precision:.1%} | Recall: {report.recall:.1%} | F1: {report.f1:.3f}")

    print("\n[3] SIZE-STRATIFIED RECALL (Defect Thickness / Shorter Side):")
    print(f"{'Size Bin':<12} | {'Range':<15} | {'GT':<6} | {'TP':<6} | {'Recall':<8} | {'Near-Miss (IoU 0.1-0.5)':<22} | {'Missed (<0.1)'}")
    print("-" * 85)
    ranges = {"tiny": "<32px", "small": "32-64px", "medium": "64-128px", "large": ">=128px"}
    for bin_name, s in report.size_stratification.items():
        print(f"{bin_name:<12} | {ranges.get(bin_name, ''):<15} | {s['gt_count']:<6} | {s['tp_count']:<6} | {s['recall']:<8.1%} | {s['near_miss_count']:<22} | {s['missed_count']}")

    print("\n[4] FALSE POSITIVE ROOT CAUSE AUDIT (Total FP = {}):".format(report.fp_breakdown["total_fp"]))
    counts = report.fp_breakdown["counts"]
    print(f"  - Real Defect Near-Misses (IoU 0.10 - 0.49): {counts['near_miss_granularity']} ({report.fp_breakdown['near_miss_pct']}%)")
    print(f"  - Marginal Overlaps (0 < IoU < 0.10):        {counts['marginal']}")
    print(f"  - True Hallucinations (IoU == 0):             {counts['hallucination']} ({report.fp_breakdown['hallucination_pct']}%)")

    print("\n[5] IoU MARGIN DISTRIBUTION ACROSS ALL GT DEFECTS:")
    for k, v in report.iou_distribution.items():
        pct = v / report.total_gt * 100 if report.total_gt else 0.0
        print(f"  IoU [{k}]: {v:3d} ({pct:5.1f}%)")

    if args.out:
        out_data = {
            "category_audit": cat_audit,
            "error_analysis": report.to_dict(),
        }
        Path(args.out).write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"\nWrote full audit to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
