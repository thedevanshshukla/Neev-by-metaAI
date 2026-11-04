"""Prompt-variant comparison over cached runs. No API calls, ever.

collect_predictions replays saved run traces (meta + response envelope)
through the same parser the live path uses, keyed by (image sha256,
prompt). run_evaluation joins those predictions with a benchmark file
and scores every (category, prompt) pair with IoU matching. Variant
comparison is then just grouping: two prompts mapped to one category
are scored on the same labelled images from the same cache.

Results document (JSON-serializable):

{
  "version": 1, "created_utc": ..., "iou_threshold": 0.5,
  "benchmark": "neev-sam31-v1",
  "results": {"window": {"window": {
     "tp": 12, "fp": 3, "fn": 2, "precision": ..., "recall": ...,
     "f1": ..., "images_labelled": 4, "images_excluded": 1,
     "mean_mask_iou": 0.81, "mask_pairs": 9,
     "per_image": [{"image_id": ..., "status": "evaluated", ...}]}}},
  "overall_micro": {"window": {"tp": ..., ...}},   # per prompt, pooled
  "excluded": {"unlabelled": 7, "presence_only": 1, "no_predictions": 2},
  "warnings": ["crack: insufficient labels (0 labelled images)"]
}

Unlabelled cells are excluded, never negatives. Statuses per image:
evaluated | negative | presence_only | unlabelled | no_predictions.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from neev.adapters.base import extract_output_text
from neev.adapters.sam31 import parse_output_text
from neev.eval.benchmark import CATEGORIES, label_status
from neev.eval.matching import mask_iou, match_predictions, prf, rle_decode

RESULTS_VERSION = 1

# Prompt text -> measurable category. Privacy prompts (face, license
# plate) are intentionally unmapped: evaluated separately, if ever.
PROMPT_TO_CATEGORY = {
    "building": "building",
    "car parked under building": "car",
    "open ground floor": "open_ground_floor",
    "shop shutter": "shutter",
    "column": "column",
    "window": "window",
    "floor slab": "floor_slab",
    "balcony": "balcony",
    "water tank": "rooftop_unit",
    "air conditioner unit": "rooftop_unit",
    "hoarding": "hoarding",
    "parapet": "parapet",
    "crack in wall": "crack",
    "exposed steel bar": "rebar",
}

# Extended mapping covering defect variations and spalling/concrete damage
EXTENDED_PROMPT_TO_CATEGORY = {
    **PROMPT_TO_CATEGORY,
    "crack": "crack",
    "wall crack": "crack",
    "cracked wall": "crack",
    "damaged wall": "spalling",
    "spalling": "spalling",
    "spalled concrete": "spalling",
}

# backend(payload, width, height) -> raster bytes (wraps masks.decode_mask).
MaskBackend = Callable[[str, int, int], bytes]


def collect_predictions(
    runs_dir: str | Path,
    run_ids: list[str] | None = None,
    mask_backend: MaskBackend | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Replay cached traces into {(image sha256, phrase): predictions}.

    Predictions hold boxes plus mask payloads/dims; rasters decode only
    when a backend is given (local decode, still no API calls). A trace
    that no longer parses is recorded under "errors", never crashed on.
    """
    root = Path(runs_dir)
    ids = run_ids if run_ids is not None else sorted(
        p.name for p in root.iterdir() if p.is_dir()
    )
    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    for run_id in sorted(ids):
        run_dir = root / run_id
        try:
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
            response = json.loads((run_dir / "response.json").read_text(encoding="utf-8"))
            phrase = meta["notes"]["phrase"]
            image_sha = meta["input_hashes"]["image"]
            detections = parse_output_text(extract_output_text(response), phrase)
        except Exception as exc:
            errors.append({"run_id": run_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        key = (image_sha, phrase)
        if key in predictions:
            first = predictions[key]["run_id"]
            errors.append({
                "run_id": run_id,
                "error": f"duplicate (image, phrase) already collected "
                         f"from {first}; kept first",
            })
            continue
        boxes, rasters, full = [], [], []
        for det in detections:
            entry = asdict(det)
            entry["source"] = run_id
            full.append(entry)
            boxes.append(det.box.to_dict())
            raster = None
            if det.mask_payload and mask_backend is not None:
                try:
                    raster = bytes(mask_backend(
                        det.mask_payload, det.mask_width, det.mask_height))
                except Exception:
                    raster = None
            rasters.append(raster)
        predictions[key] = {
            "run_id": run_id, "boxes": boxes, "rasters": rasters,
            "detections": full,
        }
    if errors:
        predictions[("errors", "")] = {"errors": errors}
    return predictions


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _box_to_polygon(box: dict[str, Any]) -> list[tuple[float, float]] | None:
    try:
        x0, y0, x1, y1 = (box[k] for k in ("x0", "y0", "x1", "y1"))
    except (KeyError, TypeError):
        return None
    if not all(isinstance(v, (int, float)) and v == v for v in (x0, y0, x1, y1)):
        return None
    if not (x0 < x1 and y0 < y1):
        return None
    return [(float(x0), float(y0)), (float(x1), float(y0)),
            (float(x1), float(y1)), (float(x0), float(y1))]


def run_rotated_evaluation(
    gt_images: list[Any],
    predictions: dict[tuple[str, str], dict[str, Any]],
    iou_threshold: float = 0.5,
    prompt_to_category: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Score cached predictions against Label Studio ground truth.

    gt_images are LSImage records (rotated geometry preserved). Matching
    is greedy rotated-IoU with a class constraint; predictions come only
    from the given cache. Non-empty images are treated as COMPLETELY
    labelled (an absent class is a negative for that class); empty
    images are explicit negatives whose detections are false alarms.
    A prompt that never ran on an image yields no_predictions there —
    never a silent zero. Classes without GT positives report undefined
    metrics, never 0.0. There are no GT masks in this export, so mask
    IoU is reported not-applicable, never fabricated.
    """
    from neev.eval.labelstudio import rect_polygon
    from neev.eval.matching import clip_polygon_to_image, match_rotated

    mapping = prompt_to_category if prompt_to_category is not None else PROMPT_TO_CATEGORY

    by_sha: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for (sha, phrase), pred in predictions.items():
        if sha == "errors":
            continue
        by_sha.setdefault(sha, []).append((phrase, pred))

    prompts_by_category: dict[str, list[str]] = {}
    for prompt, category in sorted(mapping.items()):
        prompts_by_category.setdefault(category, []).append(prompt)

    results: dict[str, dict[str, Any]] = {}
    unevaluated: dict[str, int] = {}
    negative_fp: dict[str, Any] = {"total": 0, "by_category": {}, "per_image": []}
    for image in gt_images:
        if image.is_negative:
            entry: dict[str, Any] = {"image_id": image.image_id,
                                     "task_id": image.task_id, "status": "negative",
                                     "fp": 0, "by_prompt": {}}
            for phrase, pred in by_sha.get(image.sha256, []):
                category = mapping.get(phrase, "unmapped")
                n = len(pred["boxes"])
                entry["fp"] += n
                entry["by_prompt"][phrase] = n
                negative_fp["total"] += n
                bucket = negative_fp["by_category"]
                bucket[category] = bucket.get(category, 0) + n
                if category == "unmapped":
                    unevaluated[phrase] = unevaluated.get(phrase, 0) + n
            negative_fp["per_image"].append(entry)

    for category in sorted(prompts_by_category):
        results[category] = {}
        for prompt in prompts_by_category[category]:
            cells = []
            for image in gt_images:
                if image.is_negative:
                    continue
                gt_here = [r for r in image.rects if r.label == category]
                pred = predictions.get((image.sha256, prompt))
                if pred is None:
                    cells.append({"image_id": image.image_id, "task_id": image.task_id,
                                  "status": "no_predictions"})
                    continue
                gt_entries = [{
                    "polygon": clip_polygon_to_image(
                        rect_polygon(r, image.width, image.height),
                        image.width, image.height),
                    "category": category, "result_id": r.result_id,
                } for r in gt_here]
                pred_entries, malformed = [], 0
                for box in pred["boxes"]:
                    poly = _box_to_polygon(box)
                    if poly is None:
                        malformed += 1
                    else:
                        pred_entries.append({
                            "polygon": clip_polygon_to_image(
                                poly, image.width, image.height),
                            "category": category})
                match = match_rotated(pred_entries, gt_entries, iou_threshold)
                matches = [{
                    "pred_index": pi, "result_id": gt_entries[gi]["result_id"],
                    "iou": round(score, 4),
                } for pi, gi, score in match.matches]
                cells.append({
                    "image_id": image.image_id, "task_id": image.task_id,
                    "status": "evaluated", "tp": match.tp,
                    "fp": match.fp + malformed, "fn": match.fn,
                    "malformed": malformed + len(match.malformed_pred_indices),
                    "matches": matches,
                })
            pooled = {"tp": 0, "fp": 0, "fn": 0}
            for cell in cells:
                if cell["status"] == "evaluated":
                    pooled["tp"] += cell["tp"]
                    pooled["fp"] += cell["fp"]
                    pooled["fn"] += cell["fn"]
            results[category][prompt] = {
                **pooled, **prf(pooled["tp"], pooled["fp"], pooled["fn"]),
                "images_labelled": sum(1 for c in cells if c["status"] == "evaluated"),
                "images_excluded": sum(1 for c in cells if c["status"] != "evaluated"),
                "per_image": cells,
            }

    for (sha, phrase) in predictions:
        if sha != "errors" and phrase not in PROMPT_TO_CATEGORY:
            unevaluated[phrase] = unevaluated.get(phrase, 0) + len(
                predictions[(sha, phrase)]["boxes"])

    warnings = []
    gt_label_counts: dict[str, int] = {}
    for image in gt_images:
        if image.is_negative:
            continue
        for rect in image.rects:
            gt_label_counts[rect.label] = gt_label_counts.get(rect.label, 0) + 1
    for label in sorted(gt_label_counts):
        if label not in prompts_by_category:
            warnings.append(f"{label}: {gt_label_counts[label]} ground-truth "
                            f"boxes but no prompts mapped; unscored")
    for category in sorted(prompts_by_category):
        positives = sum(len([r for r in i.rects if r.label == category])
                        for i in gt_images if not i.is_negative)
        if positives == 0:
            warnings.append(f"{category}: insufficient positives "
                            f"(0 ground-truth boxes); metrics undefined")

    by_source: dict[str, dict[str, int]] = {}
    for image in gt_images:
        bucket = by_source.setdefault(image.source, {"images": 0, "boxes": 0})
        bucket["images"] += 1
        bucket["boxes"] += len(image.rects)

    return {
        "version": RESULTS_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iou_definition": "exact rotated-polygon IoU (Sutherland-Hodgman), "
                          "greedy best-first, class-constrained",
        "iou_threshold": iou_threshold,
        "image_completeness": "non-empty images treated as completely labelled; "
                              "empty images are explicit negatives",
        "mask_iou": "not_applicable_no_gt_masks",
        "results": results,
        "negative_fp": negative_fp,
        "unevaluated_phrases": dict(sorted(unevaluated.items())),
        "by_source": by_source,
        "warnings": warnings,
    }


def run_evaluation(
    benchmark: dict[str, Any],
    predictions: dict[tuple[str, str], dict[str, Any]],
    iou_threshold: float = 0.5,
    prompt_to_category: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Score every (category, prompt) pair on labelled images only."""
    mapping = prompt_to_category if prompt_to_category is not None else PROMPT_TO_CATEGORY
    prompts_by_category: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    for prompt, category in sorted(mapping.items()):
        if category in prompts_by_category:
            prompts_by_category[category].append(prompt)

    results: dict[str, dict[str, Any]] = {}
    excluded = {"unlabelled": 0, "presence_only": 0, "no_predictions": 0}
    for category in CATEGORIES:
        results[category] = {}
        for prompt in prompts_by_category[category]:
            cells, mask_ious = [], []
            for entry in benchmark["images"]:
                status = label_status(entry.get("labels", {}), category)
                pred = predictions.get((entry["sha256"], prompt))
                if status == "unlabelled":
                    excluded["unlabelled"] += 1
                    cells.append({"image_id": entry["image_id"],
                                  "status": "unlabelled"})
                    continue
                if pred is None:
                    excluded["no_predictions"] += 1
                    cells.append({"image_id": entry["image_id"],
                                  "status": "no_predictions"})
                    continue
                label = entry["labels"][category]
                if status == "negative":
                    match = match_predictions(pred["boxes"], [], iou_threshold)
                    cells.append({
                        "image_id": entry["image_id"], "status": "negative",
                        "tp": 0, "fp": match.fp, "fn": 0,
                        "malformed": len(match.malformed_pred_indices),
                    })
                    continue
                if status == "presence_only":
                    excluded["presence_only"] += 1
                    cells.append({
                        "image_id": entry["image_id"], "status": "presence_only",
                        "image_hit": bool(pred["boxes"]),
                    })
                    continue
                match = match_predictions(pred["boxes"], label["boxes"], iou_threshold)
                for pred_index, gt_index, _ in match.matches:
                    gt_masks = label.get("masks", [])
                    pred_raster = (pred.get("rasters") or [None] * len(pred["boxes"]))[pred_index]
                    if gt_index < len(gt_masks) and pred_raster is not None:
                        gt = gt_masks[gt_index]
                        try:
                            mask_ious.append(mask_iou(
                                rle_decode(gt["counts"], gt["width"], gt["height"]),
                                pred_raster))
                        except ValueError:
                            pass
                cells.append({
                    "image_id": entry["image_id"], "status": "evaluated",
                    "tp": match.tp, "fp": match.fp, "fn": match.fn,
                    "malformed": len(match.malformed_pred_indices),
                })
            pooled = {"tp": 0, "fp": 0, "fn": 0}
            for cell in cells:
                if cell["status"] in ("evaluated", "negative"):
                    pooled["tp"] += cell["tp"]
                    pooled["fp"] += cell["fp"]
                    pooled["fn"] += cell["fn"]
            results[category][prompt] = {
                **pooled,
                **prf(pooled["tp"], pooled["fp"], pooled["fn"]),
                "images_labelled": sum(
                    1 for c in cells if c["status"] in ("evaluated", "negative")),
                "images_excluded": sum(
                    1 for c in cells if c["status"] not in ("evaluated", "negative")),
                "mean_mask_iou": _mean(mask_ious),
                "mask_pairs": len(mask_ious),
                "per_image": cells,
            }

    overall: dict[str, Any] = {}
    for prompt, category in sorted(mapping.items()):
        if category in results and prompt in results[category]:
            entry = results[category][prompt]
            overall.setdefault(prompt, {"tp": 0, "fp": 0, "fn": 0})
            overall[prompt]["tp"] += entry["tp"]
            overall[prompt]["fp"] += entry["fp"]
            overall[prompt]["fn"] += entry["fn"]
    for prompt in overall:
        overall[prompt].update(prf(**overall[prompt]))

    warnings = []
    for category in CATEGORIES:
        prompts = prompts_by_category[category]
        labelled_in_benchmark = sum(
            1 for entry in benchmark["images"]
            if label_status(entry.get("labels", {}), category)
            in ("labelled", "negative")
        )
        scored = sum(
            results[category][p]["images_labelled"] for p in prompts
        )
        if labelled_in_benchmark == 0:
            warnings.append(f"{category}: insufficient labels (0 labelled images)")
        elif not prompts:
            warnings.append(
                f"{category}: {labelled_in_benchmark} labelled image(s) "
                f"but no prompts mapped")
        elif scored == 0:
            warnings.append(
                f"{category}: {labelled_in_benchmark} labelled image(s) "
                f"but no cached predictions")

    return {
        "version": RESULTS_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iou_threshold": iou_threshold,
        "benchmark": benchmark.get("name", "?"),
        "prompt_to_category": dict(sorted(mapping.items())),
        "results": results,
        "overall_micro": overall,
        "excluded": excluded,
        "warnings": warnings,
    }
