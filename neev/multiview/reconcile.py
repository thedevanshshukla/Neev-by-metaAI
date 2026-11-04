"""Coordinate reconciliation for multi-view detections (pure functions).

Tile-local boxes translate by integer tile offsets into source-image
coordinates, clip to the frame, and stay integers end to end (the
evaluator rejects anything else). Same-prompt cross-view duplicates
suppress at IoU >= 0.5 reusing ``merge_cross_tile``; every suppression
records its surviving counterpart. Different prompts never merge, and
nearby distinct defects (IoU < 0.5) always survive.

Detection dicts carry: ``phrase``, ``box`` (tile-local ints), ``view``
(view id), ``tile_box`` (view footprint in source coords),
``frame_width``/``frame_height`` (box-token echo of the sent view dims),
plus optional ``run_id`` and mask fields. Kept detections gain
``source_box`` and ``provenance``; inputs are never mutated.
"""

from __future__ import annotations

from typing import Any

from neev.adapters.base import extract_output_text
from neev.adapters.sam31 import parse_output_text
from neev.eval.tiles import (
    aabb_iou,
    aabb_iomin,
    clip_box,
    merge_cross_tile,
    translate_box,
)

# Audited suppression threshold (Phase 8 plan; matches the tiling pilot).
SUPPRESS_IOU = 0.5


def _is_clean_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_frame(frame_width: Any, frame_height: Any,
                   view_width: int, view_height: int) -> list[str]:
    """Box-token dims must equal the sent view dims (catches resize drift)."""
    if not _is_clean_int(frame_width) or not _is_clean_int(frame_height):
        return [f"frame dims must be ints: {frame_width!r}x{frame_height!r}"]
    if frame_width != view_width or frame_height != view_height:
        return [f"frame {frame_width}x{frame_height} != "
                f"view {view_width}x{view_height}"]
    return []


def to_source_box(pred_box: Any, tile_box: dict[str, int],
                  src_width: int, src_height: int) -> dict[str, int] | None:
    """Translate a tile-local box to clipped source coords. None if empty.

    Raises ValueError for malformed (non-int or inverted) boxes: caller
    decides whether that is a rejection or a failure.
    """
    if not isinstance(pred_box, dict):
        raise ValueError(f"pred box must be an object: {pred_box!r}")
    coords = [pred_box.get(k) for k in ("x0", "y0", "x1", "y1")]
    if not all(_is_clean_int(c) for c in coords):
        raise ValueError(f"pred box coords must be ints: {pred_box!r}")
    x0, y0, x1, y1 = coords
    if not (x0 < x1 and y0 < y1):
        raise ValueError(f"pred box inverted or empty: {pred_box!r}")
    moved = translate_box({"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                          tile_box["x0"], tile_box["y0"])
    clipped = clip_box(moved, src_width, src_height)
    if clipped is None:
        return None
    return {"x0": clipped["x0"], "y0": clipped["y0"],
            "x1": clipped["x1"], "y1": clipped["y1"]}


def project_mask(raster: bytes, mask_w: int, mask_h: int,
                 pred_box: dict[str, int], tile_box: dict[str, int],
                 src_width: int, src_height: int) -> bytes:
    """Paste a tile-decoded mask raster into a full-frame 0/1 raster.

    Composes the tile offset with the tile-local box, then reuses
    ``place_box_mask`` (NEAREST resize, defensive clipping). Raises
    MaskError on bad geometry, like the single-view path.
    """
    from neev.measurements.masks import place_box_mask

    composed = {"x0": tile_box["x0"] + pred_box["x0"],
                "y0": tile_box["y0"] + pred_box["y0"],
                "x1": tile_box["x0"] + pred_box["x1"],
                "y1": tile_box["y0"] + pred_box["y1"]}
    return place_box_mask(raster, mask_w, mask_h, composed,
                          src_width, src_height)


def _survivor_ref(det: dict[str, Any]) -> dict[str, Any]:
    return {"mv_id": det["mv_id"], "view": det["view"],
            "box": dict(det["source_box"])}


def reconcile(detections: list[dict[str, Any]], src_width: int,
              src_height: int, source_sha: str = "",
              iomin_threshold: float | None = None
              ) -> tuple[list[dict], list[dict], list[dict]]:
    """Reconcile multi-view detections. Returns (kept, suppressed, rejected).

    - kept: detections with ``source_box`` + ``provenance`` attached.
    - suppressed: ``{detection, reason, survivor, iou}`` for every
      exact duplicate and IoU >= 0.5 same-prompt overlap (plus IoMin >= threshold
      if iomin_threshold is specified).
    - rejected: ``{detection, reason}`` for malformed boxes, frame
      mismatches, and boxes clipping to nothing. Never silent.
    """
    from neev.multiview.provenance import finding_provenance

    staged: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, det in enumerate(detections):
        if not isinstance(det, dict):
            rejected.append({"detection": det,
                             "reason": "detection must be an object"})
            continue
        entry = dict(det)
        entry["mv_id"] = f"mv{index}"
        tile_box = det.get("tile_box")
        if not isinstance(tile_box, dict):
            rejected.append({"detection": det, "reason": "missing tile_box"})
            continue
        view_w = tile_box.get("x1", 0) - tile_box.get("x0", 0)
        view_h = tile_box.get("y1", 0) - tile_box.get("y0", 0)
        frame_errors = validate_frame(det.get("frame_width"),
                                      det.get("frame_height"), view_w, view_h)
        if frame_errors:
            rejected.append({"detection": det, "reason": frame_errors[0]})
            continue
        try:
            source_box = to_source_box(det.get("box"), tile_box,
                                       src_width, src_height)
        except ValueError as exc:
            rejected.append({"detection": det, "reason": str(exc)})
            continue
        if source_box is None:
            rejected.append({"detection": det,
                             "reason": "box clips to nothing in source frame"})
            continue
        entry["source_box"] = source_box
        view_id = det.get("view")
        kind = det.get("view_kind")
        entry["provenance"] = finding_provenance(
            view_id if isinstance(view_id, str) else "unknown",
            kind if isinstance(kind, str) else "tile",
            tile_box, source_sha,
            det.get("phrase", "") if isinstance(det.get("phrase"), str) else "",
            det.get("run_id"))
        staged.append(entry)

    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    by_phrase: dict[str, list[dict[str, Any]]] = {}
    for entry in staged:
        by_phrase.setdefault(entry.get("phrase", ""), []).append(entry)
    for phrase in sorted(by_phrase):
        group = by_phrase[phrase]
        # Exact duplicates first: same box, keep lowest mv_id.
        seen: dict[tuple[int, int, int, int], dict[str, Any]] = {}
        unique: list[dict[str, Any]] = []
        for entry in sorted(group, key=lambda e: e["mv_id"]):
            key = (entry["source_box"]["x0"], entry["source_box"]["y0"],
                   entry["source_box"]["x1"], entry["source_box"]["y1"])
            if key in seen:
                suppressed.append({
                    "detection": entry, "reason": "exact-duplicate",
                    "survivor": _survivor_ref(seen[key]), "iou": 1.0})
            else:
                seen[key] = entry
                unique.append(entry)
        # Cross-view NMS on source coords; map results back by identity.
        boxes = [dict(e["source_box"]) for e in unique]
        kept_boxes, _ = merge_cross_tile(boxes, SUPPRESS_IOU,
                                         iomin_threshold=iomin_threshold)
        box_to_entry = {id(b): e for b, e in zip(boxes, unique)}
        group_kept = [box_to_entry[id(b)] for b in kept_boxes]
        kept_set = {e["mv_id"] for e in group_kept}
        for entry in unique:
            if entry["mv_id"] in kept_set:
                continue
            best, best_score, best_reason = None, -1.0, "duplicate-iou"
            for cand in group_kept:  # deterministic counterpart search
                score = aabb_iou(entry["source_box"], cand["source_box"])
                score_min = aabb_iomin(entry["source_box"], cand["source_box"])
                if score >= SUPPRESS_IOU and score > best_score:
                    best, best_score, best_reason = cand, score, "duplicate-iou"
                elif (iomin_threshold is not None
                      and score_min >= iomin_threshold
                      and score_min > best_score):
                    best, best_score, best_reason = cand, score_min, "duplicate-containment"
            if best is None:
                for cand in group_kept:
                    score = aabb_iou(entry["source_box"], cand["source_box"])
                    if score > best_score:
                        best, best_score = cand, score
            suppressed.append({
                "detection": entry, "reason": best_reason,
                "survivor": _survivor_ref(best) if best else None,
                "iou": round(best_score, 4)})
        kept.extend(group_kept)
    kept.sort(key=lambda e: e["mv_id"])
    return kept, suppressed, rejected


def detections_from_run(runs: Any, run_id: str, phrase: str, view_id: str,
                        view_kind: str, tile_box: dict[str, int]
                        ) -> list[dict[str, Any]]:
    """Parse one cached run response into reconcile-ready detections.

    Box frame echoes come from the wire tokens; a view whose dims the
    model misreports is caught later by ``validate_frame`` during
    ``reconcile``. Unparseable responses raise (same rule as the live
    path: drift never masquerades as "nothing found").
    """
    stored = runs.load(run_id)
    parsed = parse_output_text(
        extract_output_text(stored["response"]), phrase)
    return [{
        "phrase": phrase,
        "box": det.box.to_dict(),
        "view": view_id,
        "view_kind": view_kind,
        "tile_box": dict(tile_box),
        "frame_width": det.frame_width,
        "frame_height": det.frame_height,
        "run_id": run_id,
        "mask_payload": det.mask_payload,
        "mask_width": det.mask_width,
        "mask_height": det.mask_height,
        "mask_encoding": det.mask_encoding,
    } for det in parsed]
