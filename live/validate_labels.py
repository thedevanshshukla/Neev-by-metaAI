"""Validate the native Label Studio export (read-only, offline).

Parses every task, maps uploads to dataset files, checks geometry,
dimensions, rotations, duplicates, and negative examples, then prints
the validation report. Exits 1 when any image fails mapping or
validation; clipping/FP-noise notes are warnings, not failures.

Usage:
  python live/validate_labels.py
  python live/validate_labels.py --export eval/annotations_label_studio.json --dataset dataset --no-dim-check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neev.eval.labelstudio import load_export, validate_export  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", default="eval/annotations_label_studio.json")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--no-dim-check", action="store_true")
    args = parser.parse_args()

    try:
        tasks = load_export(args.export)
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot load export: {exc}", file=sys.stderr)
        return 2
    report = validate_export(tasks, args.dataset, check_dims=not args.no_dim_check)

    print("=== Label Studio validation (read-only) ===")
    print(f"tasks: {report['tasks']}  parsed: {report['images_parsed']}  "
          f"annotations: {report['total_annotations']}")
    print(f"empty negatives (task ids): {report['empty_negative_images']}")
    print(f"rotated boxes: {report['rotated_boxes']}  "
          f"fp-snapped: {report['snapped_fp_noise']}  "
          f"clipped overflow: {report['clipped_overflow']}")
    print(f"sources: {report['sources']}")
    print("per-class:")
    for label, count in report["per_class"].items():
        print(f"  {label}: {count}")
    for mismatch in report["dim_mismatches"]:
        print(f"DIM MISMATCH: {mismatch}")
    if report["duplicate_image_ids"]:
        print(f"DUP IMAGE IDS: {report['duplicate_image_ids']}")
    if report["byte_identical_duplicates"]:
        print(f"byte-identical duplicates (resolved, reported): "
              f"{len(report['byte_identical_duplicates'])} tasks")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    if report["errors"] or report["dim_mismatches"] or report["duplicate_image_ids"]:
        return 1
    print("VALID: all images mapped, geometry within policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
