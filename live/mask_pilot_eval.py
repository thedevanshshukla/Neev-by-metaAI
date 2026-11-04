"""Offline mask-pilot evaluation (no network, no API calls, no inference).

Scores cached SAM mask rasters (decoded offline) against human mask
polygons for a small image subset. Prompts, thresholds, and images are
explicit; box metrics are read from the existing experiment file for
context only and never recomputed here. Refuses to overwrite output.

Usage:
  python live/mask_pilot_eval.py --out runs/mask_pilot_eval_v1.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neev.eval.compare import collect_predictions  # noqa: E402
from neev.eval.labelstudio import load_export, parse_task  # noqa: E402
from neev.eval.mask_eval import (  # noqa: E402
    MaskInstance,
    evaluate_image,
    pool_scores,
    polygons_to_instance,
)
from neev.measurements.masks import MaskError, decode_mask  # noqa: E402

PILOT_IMAGES = ["cr5", "cr4", "dm1", "k1", "in1"]
PROMPTS = ["crack", "crack in wall"]
THRESHOLDS = [0.25, 0.5]


def _load_validator():
    path = (Path(__file__).resolve().parent.parent / "eval" / "mask_pilot"
            / "validate.py")
    spec = importlib.util.spec_from_file_location("mask_pilot_validate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _place_box_mask(raster: bytes, mask_w: int, mask_h: int,
                    box: dict[str, Any], width: int, height: int) -> bytes:
    """Box-local raster to a full-frame 0/1 raster.

    Mirrors live/visualize.py exactly: NEAREST resize to the box, paste
    at the box origin; out-of-frame box parts clip defensively.
    """
    from PIL import Image

    mask = Image.frombytes("L", (mask_w, mask_h),
                           bytes(b * 255 for b in raster))
    bw = max(1, box["x1"] - box["x0"])
    bh = max(1, box["y1"] - box["y0"])
    mask = mask.resize((bw, bh), Image.NEAREST)
    frame = Image.new("L", (width, height), 0)
    x0, y0 = box["x0"], box["y0"]
    src = mask.crop((max(0, -x0), max(0, -y0),
                     bw - max(0, x0 + bw - width),
                     bh - max(0, y0 + bh - height)))
    frame.paste(src, (max(0, x0), max(0, y0)))
    return bytes(1 if v else 0 for v in frame.tobytes())


def _pred_instances(pred: dict[str, Any], image_id: str, width: int,
                    height: int) -> tuple[list[MaskInstance | None], dict]:
    """Cached detections to full-frame mask instances.

    Every quality problem (missing payload, decode failure, frame
    mismatch, bad box, empty raster) becomes None (a malformed FP)
    with an explicit reason counted; nothing is silently dropped.
    """
    insts: list[MaskInstance | None] = []
    quality = {"ok": 0, "missing_payload": 0, "decode_failed": 0,
               "frame_mismatch": 0, "bad_box": 0, "empty_raster": 0}
    dets = pred.get("detections", [])
    boxes = pred.get("boxes", [])
    rasters = pred.get("rasters", [None] * len(dets))
    for i, det in enumerate(dets):
        box = boxes[i] if i < len(boxes) else {}
        raster = rasters[i] if i < len(rasters) else None
        if ((det.get("frame_width"), det.get("frame_height"))
                not in (None, (None, None))
                and (det.get("frame_width"), det.get("frame_height"))
                != (width, height)):
            quality["frame_mismatch"] += 1
            insts.append(None)
        elif (not all(isinstance(box.get(k), int) for k in
                      ("x0", "y0", "x1", "y1"))
                or not (box["x0"] < box["x1"] and box["y0"] < box["y1"])):
            quality["bad_box"] += 1
            insts.append(None)
        elif not det.get("mask_payload"):
            quality["missing_payload"] += 1
            insts.append(None)
        elif raster is None:
            quality["decode_failed"] += 1
            insts.append(None)
        else:
            placed = _place_box_mask(raster, det["mask_width"],
                                     det["mask_height"], box, width, height)
            if not any(placed):
                quality["empty_raster"] += 1
                insts.append(None)
            else:
                quality["ok"] += 1
                insts.append(MaskInstance(
                    image_id=image_id, instance_id=f"pred#{i}", width=width,
                    height=height, raster=placed))
    return insts, quality


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", default="eval/annotations_label_studio.json")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--artifact", default="eval/mask_pilot/gt_masks.v1.json")
    parser.add_argument("--box-results", default="runs/crack_exp_eval.json")
    parser.add_argument("--out", default="runs/mask_pilot_eval_v1.json")
    parser.add_argument("--parser-wheel", default=None,
                        help="meta_sam_parser wheel (added to sys.path, "
                             "never installed), as in live/visualize.py")
    args = parser.parse_args()

    if args.parser_wheel:
        sys.path.insert(0, args.parser_wheel)
    try:
        import meta_sam_parser  # noqa: F401
    except ImportError:
        print("ERROR: meta_sam_parser unavailable; pass --parser-wheel",
              file=sys.stderr)
        return 2

    out_path = Path(args.out)
    if out_path.exists():
        print(f"ERROR: refusing to overwrite {out_path}", file=sys.stderr)
        return 2
    try:
        artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot load artifact: {exc}", file=sys.stderr)
        return 2
    validator = _load_validator()
    gt_by_image = {e["image_id"]: e for e in artifact.get("images", [])}
    actual_dims = {iid: (e["width"], e["height"])
                   for iid, e in gt_by_image.items()}
    errors, warnings = validator.validate_artifact(artifact, actual_dims)
    if errors:
        print("ERROR: artifact invalid:", file=sys.stderr)
        for emsg in errors:
            print(f"  {emsg}", file=sys.stderr)
        return 2

    tasks = load_export(args.export)
    ls_images: dict[str, Any] = {}
    for task in tasks:
        image, errs = parse_task(task, args.dataset)
        if errs or image is None:
            print(f"ERROR: task {task.get('id')}: {errs}", file=sys.stderr)
            return 2
        ls_images[image.image_id] = image
    try:
        predictions = collect_predictions(
            args.runs_dir,
            mask_backend=lambda p, w, h: decode_mask(p, w, h))
    except (MaskError, ImportError) as exc:
        print(f"ERROR: mask decode unavailable: {exc}", file=sys.stderr)
        return 2
    box_results = json.loads(Path(args.box_results).read_text(encoding="utf-8"))

    results: dict[str, Any] = {
        "version": 1,
        "subset": PILOT_IMAGES,
        "excluded": ["h2", "i1"],
        "thresholds": THRESHOLDS,
        "matching": ("greedy class-constrained one-to-one mask-IoU "
                     "matching; malformed preds are FPs"),
        "modes": {},
    }
    for prompt in PROMPTS:
        per_image: dict[str, Any] = {}
        pooled_input: dict[str, list] = {}
        mask_quality: dict[str, dict] = {}
        skipped: dict[str, str] = {}
        for iid in PILOT_IMAGES:
            g = ls_images.get(iid)
            entry = gt_by_image.get(iid)
            if g is None or entry is None:
                skipped[iid] = "missing ground truth or image record"
                continue
            pred = predictions.get((g.sha256, prompt))
            if pred is None:
                skipped[iid] = f"no cached '{prompt}' predictions"
                continue
            gt_insts = []
            for inst in entry["instances"]:
                made = polygons_to_instance(
                    iid, inst["instance_id"], entry["width"], entry["height"],
                    inst["polygons"])
                if made is None:
                    skipped.setdefault(
                        iid, f"unmeasurable GT {inst['instance_id']}")
                else:
                    gt_insts.append(made)
            if iid in skipped:
                continue
            from PIL import Image

            with Image.open(Path(args.dataset) / g.relpath) as im:
                native_w, native_h = im.size
            if (entry["width"], entry["height"]) != (native_w, native_h):
                skipped[iid] = ("artifact dims differ from measured "
                                "source image")
                continue
            insts, quality = _pred_instances(pred, iid, native_w, native_h)
            mask_quality[iid] = {"run_id": pred.get("run_id"), **quality}
            cell: dict[str, Any] = {
                "image_id": iid, "task_id": g.task_id,
                "run_id": pred.get("run_id"),
                "gt_instances": len(gt_insts), "preds": len(insts),
                "by_threshold": {},
            }
            for thr in THRESHOLDS:
                score = evaluate_image(iid, insts, gt_insts, thr)
                cell["by_threshold"][str(thr)] = {
                    "tp": score.tp, "fp": score.fp, "fn": score.fn,
                    "precision": score.precision, "recall": score.recall,
                    "f1": score.f1, "matches": score.matches,
                    "malformed_preds": score.malformed_preds,
                }
                pooled_input.setdefault(str(thr), []).append(score)
            per_image[iid] = cell
        pooled = {thr: pool_scores(scores)
                  for thr, scores in pooled_input.items()}
        box_entry = box_results["prompts"].get(prompt, {})
        box_subset = {"tp": 0, "fp": 0, "fn": 0}
        for cell in box_entry.get("per_image", []):
            if cell.get("image_id") in PILOT_IMAGES and \
                    cell.get("status") == "evaluated":
                for k in box_subset:
                    box_subset[k] += cell[k]
        results["modes"][prompt] = {
            "per_image": per_image, "pooled": pooled,
            "mask_quality": mask_quality, "skipped": skipped,
            "box_context_5img": box_subset,
        }
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True),
                        encoding="utf-8")
    for prompt in PROMPTS:
        for thr in THRESHOLDS:
            p = results["modes"][prompt]["pooled"][str(thr)]
            print(f"{prompt} IoU>={thr}: TP={p['tp']} FP={p['fp']} FN={p['fn']} "
                  f"P={p['precision']} R={p['recall']} F1={p['f1']}")
    print(f"results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
