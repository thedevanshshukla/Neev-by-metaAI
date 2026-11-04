"""Evaluate cached SAM predictions against Label Studio ground truth.

Offline, no network: replays runs/ traces and scores them with exact
rotated-polygon IoU against the native export. Prompts that never ran
on an image are reported, never zero-filled. Writes a results JSON.

Usage:
  python live/evaluate_ls.py
  python live/evaluate_ls.py --out runs/ls_eval.json --iou 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neev.eval.compare import collect_predictions, run_rotated_evaluation  # noqa: E402
from neev.eval.labelstudio import load_export, parse_task  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", default="eval/annotations_label_studio.json")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default=None)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    try:
        tasks = load_export(args.export)
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot load export: {exc}", file=sys.stderr)
        return 2
    gt_images, held_out = [], []
    for task in tasks:
        image, errors = parse_task(task, args.dataset)
        if errors:
            held_out.append({"task_id": task.get("id"), "errors": errors})
        elif image is not None:
            gt_images.append(image)
    if held_out:
        print(f"HELD OUT {len(held_out)} task(s) with errors:", file=sys.stderr)
        for entry in held_out:
            for error in entry["errors"]:
                print(f"  task {entry['task_id']}: {error}", file=sys.stderr)
        return 1

    predictions = collect_predictions(args.runs_dir)
    results = run_rotated_evaluation(gt_images, predictions, iou_threshold=args.iou)
    results["held_out_tasks"] = []

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = Path(args.out or str(Path(args.runs_dir) / f"ls_eval_{stamp}.json"))
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    print("=== LS evaluation (no network) ===")
    print(f"images: {len(gt_images)}  iou: {args.iou} "
          f"({results['iou_definition'][:60]}...)")
    measured = 0
    for category in sorted(results["results"]):
        for prompt in sorted(results["results"][category]):
            entry = results["results"][category][prompt]
            if entry["images_labelled"] == 0:
                continue
            measured += 1
            print(f"{category} :: {prompt}: P={entry['precision']} "
                  f"R={entry['recall']} F1={entry['f1']} "
                  f"(TP={entry['tp']} FP={entry['fp']} FN={entry['fn']})")
    if measured == 0:
        print("0 measured cells: no cached predictions overlap this dataset yet.")
    neg = results["negative_fp"]
    print(f"negative-image FPs: total={neg['total']} by_category={neg['by_category']}")
    if results["unevaluated_phrases"]:
        print(f"unevaluated phrases: {results['unevaluated_phrases']}")
    for warning in results["warnings"]:
        print(f"warning: {warning}")
    print(f"results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
