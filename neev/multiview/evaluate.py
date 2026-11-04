"""Offline multi-view evaluation adapter (no inference, no protected edits).

Builds reconciled multi-view prediction cells from cached runs and
scores them through the untouched rotated evaluator
(``run_rotated_evaluation``, official IoU 0.5). Ground truth and
matching semantics are never modified: merged cells simply override
single-view cells under the same ``(image_sha, prompt)`` keys before
the frozen scoring path runs.

Report shape keeps official benchmark metrics separate from
exploratory diagnostics::

    {"official": <run_rotated_evaluation output verbatim>,
     "exploratory": {per-cell kept/suppressed counts, notes},
     "cost": {requests, cached_cells, uncached_cells, projected_usd}}
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from neev.adapters.sam31 import PRICE_PER_IMAGE_USD


def estimate_cost(n_requests: int) -> float:
    """Projected spend for new (uncached) view requests."""
    return round(n_requests * PRICE_PER_IMAGE_USD, 4)


def sha_of_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tile_boxes_for_image(tiles_doc: dict[str, Any],
                         image_id: str) -> dict[str, dict[str, int]]:
    """{tile_id: source-coords box} from a tiling manifest document."""
    entry = tiles_doc[image_id]
    return {t["tile_id"]: {"x0": t["x0"], "y0": t["y0"],
                           "x1": t["x1"], "y1": t["y1"]}
            for t in entry["tiles"]}


def merge_views(source_sha: str, phrase: str,
                view_cells: list[dict[str, Any]],
                src_width: int, src_height: int) -> dict[str, Any]:
    """Reconcile cached view cells into one merged prediction cell.

    Each view cell holds ``view_id``, ``view_kind``, ``tile_box``, and a
    ``collect_predictions``-shaped ``cell`` (``detections`` + ``run_id``).
    Returns a merged cell in the same shape (``run_id`` None: derived,
    not a model response) plus ``merged_from``, ``suppressed``,
    ``rejected``, and ``views`` provenance.
    """
    from neev.multiview import reconcile as mv_reconcile

    dets: list[dict[str, Any]] = []
    for view in view_cells:
        cell = view["cell"]
        for det in cell.get("detections", []):
            if not isinstance(det, dict):
                continue
            dets.append({
                "phrase": phrase,
                "box": det.get("box"),
                "view": view["view_id"],
                "view_kind": view.get("view_kind", "tile"),
                "tile_box": dict(view["tile_box"]),
                "frame_width": det.get("frame_width"),
                "frame_height": det.get("frame_height"),
                "run_id": det.get("source"),
                "mask_payload": det.get("mask_payload"),
                "mask_width": det.get("mask_width"),
                "mask_height": det.get("mask_height"),
                "mask_encoding": det.get("mask_encoding"),
            })
    kept, suppressed, rejected = mv_reconcile.reconcile(
        dets, src_width, src_height, source_sha)
    return {
        "run_id": None,
        "boxes": [dict(k["source_box"]) for k in kept],
        "rasters": [None] * len(kept),
        "detections": kept,
        "merged_from": sorted({v["cell"].get("run_id") for v in view_cells
                               if v["cell"].get("run_id")}),
        "suppressed": suppressed,
        "rejected": rejected,
        "views": [{"view_id": v["view_id"],
                   "view_kind": v.get("view_kind", "tile"),
                   "run_id": v["cell"].get("run_id")}
                  for v in view_cells],
    }


def override_predictions(
    predictions: dict[tuple[str, str], dict[str, Any]],
    merged: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Copy of ``predictions`` with merged cells swapped in. Never mutates."""
    out = dict(predictions)
    out.update(merged)
    return out


def score_multiview(gt_images: list[Any],
                    predictions: dict[tuple[str, str], dict[str, Any]],
                    merged: dict[tuple[str, str], dict[str, Any]],
                    iou_threshold: float = 0.5) -> dict[str, Any]:
    """Score overridden predictions with the frozen rotated evaluator."""
    from neev.eval.compare import run_rotated_evaluation

    official = run_rotated_evaluation(
        gt_images, override_predictions(predictions, merged), iou_threshold)
    return official


def cost_report(n_cached: int, n_uncached: int) -> dict[str, Any]:
    """Request accounting for a multi-view arm (no billing here)."""
    return {
        "cached_cells": n_cached,
        "uncached_cells": n_uncached,
        "requests_if_confirmed": n_uncached,
        "projected_usd": estimate_cost(n_uncached),
    }


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    """Write an eval report. Refuses to overwrite, like build_report."""
    out = Path(path)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    return out
