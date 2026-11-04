"""Offline crack coverage diagnostics (no network, no API calls).

Scores cached SAM predictions with the unchanged strict matcher plus
additive geometric coverage diagnostics (neev.eval.coverage). Modes:
four crack prompts on the shared 16-image subset, dm1/k1 full-frame vs
merged-tile, and cr4/cr5 full-frame controls. Refuses to overwrite output.

Usage:
  python live/coverage_ls.py --out runs/crack_coverage_diag.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neev.eval.compare import collect_predictions  # noqa: E402
from neev.eval.coverage import GRAPH_THRESHOLD, THRESHOLDS, pool, summarize  # noqa: E402
from neev.eval.labelstudio import load_export, parse_task, rect_polygon  # noqa: E402
from neev.eval.matching import clip_polygon_to_image  # noqa: E402
from neev.eval.tiles import clip_box, merge_cross_tile, translate_box  # noqa: E402

CATEGORY = "crack"
SUBSET_16 = ["cr1", "cr2", "cr3", "cr4", "cr5", "dm1", "dv2", "fa3", "h1",
             "h2", "i1", "i2", "k1", "k2", "in1", "in3"]
PROMPTS = ["crack in wall", "crack", "wall crack", "damaged wall"]


def box_to_polygon(box: dict[str, Any]) -> list[tuple[float, float]] | None:
    """Axis-aligned box dict to corners; None when malformed.

    Mirrors compare._box_to_polygon semantics for runner-side use.
    """
    try:
        x0, y0, x1, y1 = (box[k] for k in ("x0", "y0", "x1", "y1"))
    except (KeyError, TypeError):
        return None
    if not all(isinstance(v, (int, float)) and v == v
               for v in (x0, y0, x1, y1)):
        return None
    if not (x0 < x1 and y0 < y1):
        return None
    return [(float(x0), float(y0)), (float(x1), float(y0)),
            (float(x1), float(y1)), (float(x0), float(y1))]


def gt_polys(image: Any) -> tuple[list[list[tuple[float, float]]], list[str]]:
    polys, ids = [], []
    for r in image.rects:
        if r.label != CATEGORY:
            continue
        polys.append(clip_polygon_to_image(
            rect_polygon(r, image.width, image.height),
            image.width, image.height))
        ids.append(r.result_id)
    return polys, ids


def score_mode(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool per-image summaries; gather top fragmentation cases."""
    pooled = pool([c["summary"] for c in cells])
    frag: list[dict[str, Any]] = []
    for cell in cells:
        for comp in cell["summary"]["graph"]["detail"]:
            if len(comp["pred_ids"]) == 1 and len(comp["gt_ids"]) >= 2:
                frag.append({"image_id": cell["image_id"],
                             "task_id": cell["task_id"],
                             "pred_id": comp["pred_ids"][0],
                             "gt_ids": comp["gt_ids"],
                             "n_gt": len(comp["gt_ids"])})
    frag.sort(key=lambda e: (-e["n_gt"], e["image_id"], e["pred_id"]))
    shared: list[dict[str, Any]] = []
    for cell in cells:
        for thr, entries in cell["summary"]["many_to_one"]["shared"].items():
            for e in entries:
                shared.append({"image_id": cell["image_id"], "threshold": thr,
                               **e})
    shared.sort(key=lambda e: (-e["n_gt"], e["image_id"], e["pred_id"]))
    return {"pooled": pooled, "top_fragmentation": frag[:10],
            "many_to_one_shared": shared, "per_image": cells}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", default="eval/annotations_label_studio.json")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default="runs/crack_coverage_diag.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        print(f"ERROR: refusing to overwrite {out_path}", file=sys.stderr)
        return 2
    tasks = load_export(args.export)
    gt_images: dict[str, Any] = {}
    for task in tasks:
        image, errors = parse_task(task, args.dataset)
        if errors or image is None:
            print(f"ERROR: task {task.get('id')}: {errors}", file=sys.stderr)
            return 2
        gt_images[image.image_id] = image
    missing = [i for i in SUBSET_16 if i not in gt_images]
    if missing:
        print(f"ERROR: subset images missing: {missing}", file=sys.stderr)
        return 2
    predictions = collect_predictions(args.runs_dir)

    modes: dict[str, Any] = {}
    for prompt in PROMPTS:
        cells = []
        for iid in SUBSET_16:
            g = gt_images[iid]
            pred = predictions.get((g.sha256, prompt))
            if pred is None:
                cells.append({"image_id": iid, "task_id": g.task_id,
                              "status": "no_predictions"})
                continue
            polys = [clip_polygon_to_image(p, g.width, g.height)
                     if p is not None else None
                     for p in (box_to_polygon(b) for b in pred["boxes"])]
            gpolys, gids = gt_polys(g)
            cells.append({
                "image_id": iid, "task_id": g.task_id, "status": "evaluated",
                "run_id": pred.get("run_id"),
                "summary": summarize(
                    polys, [f"{pred.get('run_id')}#{i}"
                            for i in range(len(polys))],
                    gpolys, gids, CATEGORY),
            })
        evaluated = [c for c in cells if c["status"] == "evaluated"]
        modes[f"prompt:{prompt}"] = {
            "subset": SUBSET_16, "images_evaluated": len(evaluated),
            **score_mode(evaluated),
        }

    # Merged-tile modes for dm1/k1 (first-wins per tile file).
    tile_spec = json.loads(
        (Path(args.runs_dir) / "crack_tiling_tiles.json").read_text(
            encoding="utf-8"))
    tile_runs: dict[str, list[tuple[str, list]]] = {}
    for run_dir in sorted(Path(args.runs_dir).iterdir()):
        meta_p = run_dir / "meta.json"
        if not run_dir.is_dir() or not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        notes = meta.get("notes", {})
        if notes.get("phrase") != "crack":
            continue
        img = notes.get("image", "")
        if "crack_tiles" not in img.replace("\\", "/"):
            continue
        stem = Path(img.replace("\\", "/")).stem
        key = (meta["input_hashes"]["image"], "crack")
        tile_runs.setdefault(stem, []).append(
            (run_dir.name, predictions[key]["boxes"]))
    for img_id in ["dm1", "k1"]:
        g = gt_images[img_id]
        spec = tile_spec[img_id]
        assert [spec["width"], spec["height"]] == [g.width, g.height]
        raw, excluded = [], []
        for t in spec["tiles"]:
            stem = f"{img_id}_{t['tile_id']}"
            entries = tile_runs.get(stem, [])
            if not entries:
                print(f"ERROR: no run for tile {stem}", file=sys.stderr)
                return 2
            (run_id, boxes), rest = entries[0], entries[1:]
            excluded += [r for r, _ in rest]
            for i, b in enumerate(boxes):
                moved = translate_box(
                    {"x0": b["x0"], "y0": b["y0"], "x1": b["x1"],
                     "y1": b["y1"]}, t["x0"], t["y0"])
                clipped = clip_box(moved, g.width, g.height)
                if clipped is not None:
                    clipped["tile_id"] = t["tile_id"]
                    clipped["tile_pred_index"] = i
                    raw.append(clipped)
        kept, suppressed = merge_cross_tile(raw, 0.5)
        gpolys, gids = gt_polys(g)
        polys = [clip_polygon_to_image(box_to_polygon(b), g.width, g.height)
                 for b in kept]
        ids = [f"{b['tile_id']}#{b['tile_pred_index']}" for b in kept]
        modes[f"merged-tiles:{img_id}"] = {
            "subset": [img_id], "raw_tile_preds": len(raw),
            "kept": len(kept), "suppressed": len(suppressed),
            "excluded_duplicate_runs": sorted(excluded),
            **score_mode([{
                "image_id": img_id, "task_id": g.task_id,
                "status": "evaluated",
                "summary": summarize(polys, ids, gpolys, gids, CATEGORY)}]),
        }

    results = {
        "version": 1,
        "category": CATEGORY,
        "thresholds": list(THRESHOLDS),
        "graph_threshold": GRAPH_THRESHOLD,
        "disclaimer": ("Coverage and graph statistics are geometric overlap "
                       "diagnostics, not proof of correctness. Strict "
                       "one-to-one IoU>=0.5 metrics remain official."),
        "modes": modes,
    }
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True),
                        encoding="utf-8")
    for name, mode in modes.items():
        s = mode["pooled"]["strict"]
        print(f"{name}: strict TP={s['tp']} FP={s['fp']} FN={s['fn']} "
              f"P={s['precision']} R={s['recall']} F1={s['f1']}")
    print(f"results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
