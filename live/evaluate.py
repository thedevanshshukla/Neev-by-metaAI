"""Offline evaluation: join benchmark labels with cached runs.

Reads a benchmark file plus runs/ traces, writes a results JSON. Makes NO
network calls and spends nothing: predictions come only from traces that
already exist. Unlabelled cells are excluded with warnings; nothing is
measured until labels exist.

Usage:
  python live/evaluate.py                                              # defaults
  python live/evaluate.py --benchmark eval/benchmark_v1.json --out results.json
  python live/evaluate.py --decode-masks --parser-wheel <wheel>        # + mask IoU
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neev.eval.benchmark import load_benchmark  # noqa: E402
from neev.eval.compare import collect_predictions, run_evaluation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="eval/benchmark_v1.json")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default=None, help="Results JSON path")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--decode-masks", action="store_true",
                        help="Decode pred masks locally for mask IoU (no API)")
    parser.add_argument("--parser-wheel", default=None)
    args = parser.parse_args()

    if args.parser_wheel:
        sys.path.insert(0, args.parser_wheel)
    try:
        benchmark = load_benchmark(args.benchmark)
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot load benchmark: {exc}", file=sys.stderr)
        return 2

    backend = None
    if args.decode_masks:
        from neev.measurements.masks import decode_mask

        def backend(payload: str, width: int, height: int) -> bytes:
            return decode_mask(payload, width, height)

    predictions = collect_predictions(args.runs_dir, mask_backend=backend)
    results = run_evaluation(benchmark, predictions, iou_threshold=args.iou)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = Path(args.out or str(Path(args.runs_dir) / f"eval_{stamp}.json"))
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    print("=== offline evaluation (no network) ===")
    print(f"benchmark: {results['benchmark']} "
          f"({len(benchmark['images'])} images)  iou: {args.iou}")
    measured = 0
    for category in sorted(results["results"]):
        for prompt in sorted(results["results"][category]):
            entry = results["results"][category][prompt]
            if entry["images_labelled"] == 0:
                continue
            measured += 1
            print(f"{category} :: {prompt}: "
                  f"P={entry['precision']} R={entry['recall']} F1={entry['f1']} "
                  f"(TP={entry['tp']} FP={entry['fp']} FN={entry['fn']}, "
                  f"labelled={entry['images_labelled']})")
    if measured == 0:
        print("0 labelled cells: nothing measurable yet.")
    print(f"excluded: {results['excluded']}")
    for warning in results["warnings"]:
        print(f"warning: {warning}")
    print(f"results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
