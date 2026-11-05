"""Cached-first inspection execution over the existing SAM pipeline.

Default prompt set (deterministic, documented): the 16 docs.md prompts
in ``SAM31_PROMPTS`` table order -- the same set ``live/sweep_sam31.py``
runs by default. Every cell reuses ``SAM31Adapter``, ``RunStore``, and
the sanitized-trace cache form, so app runs and CLI runs share cache
entries in both directions.

Safety: this module never sends a request unless the caller passes
``confirm_live=True`` AND valid API configuration exists. Cached-only
runs, dashboard loads, and report rendering never touch the network.
There are no retries and no automatic loops: each uncached phrase gets
at most one live attempt per call.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from neev.adapters.base import Transport, UrllibTransport
from neev.adapters.sam31 import (
    PRICE_PER_IMAGE_USD,
    SAM31Adapter,
    SAM31_PROMPTS,
    build_image_request,
    sanitized_request,
)
from neev.app.store import InspectionStore, io_lock, utcnow
from neev.config import Config, auth_headers
from neev.evidence.priority import TINY_AREA_FRAC
from neev.multiview import provenance as prov
from neev.multiview import raster as mv_raster
from neev.multiview import reconcile as mv_reconcile
from neev.multiview import views as mv_views
from neev.persistence.runs import RunMeta, RunStore, sha256_bytes


def default_prompts() -> list[str]:
    """The 16 documented prompts in table order."""
    return [phrase for phrases in SAM31_PROMPTS.values() for phrase in phrases]


# Largest custom phrase list accepted per run call (cost guardrail).
MAX_PHRASES_PER_RUN = 32

_UNET_ADAPTER = None

try:
    import spaces
    _gpu_decorator = spaces.GPU
except (ImportError, Exception):
    def _gpu_decorator(func):
        return func


def _get_unet_adapter():
    global _UNET_ADAPTER
    if _UNET_ADAPTER is None:
        from neev.adapters.crack_unet import CrackUNetAdapter
        _UNET_ADAPTER = CrackUNetAdapter()
    return _UNET_ADAPTER


@_gpu_decorator
def _run_unet_detections(raw: bytes) -> list[dict[str, Any]]:
    import io
    import logging
    import numpy as np
    from PIL import Image

    logger = logging.getLogger("neev.runner")
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        unet = _get_unet_adapter()
        res = unet.segment(np.array(pil_img))
        kept = []
        for i, inst in enumerate(res.instances):
            kept.append({
                "source_box": dict(inst.box),
                "run_id": f"unet_{i}",
                "view": "whole",
                "view_kind": "whole",
                "provenance": {"model": "crack_unet", "prob": round(float(inst.mean_prob), 3), "area_px": inst.area_px},
                "mask_payload": None,
                "mask_width": inst.box["x1"] - inst.box["x0"],
                "mask_height": inst.box["y1"] - inst.box["y0"],
                "mask_encoding": "one_bit",
            })
        return kept
    except Exception as exc:
        logger.error("U-Net crack segmentation failed during inspection: %s", exc, exc_info=True)
        return []


def _fuse_hybrid_detections(sam_dets: list[dict[str, Any]], unet_dets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = list(sam_dets)
    for u in unet_dets:
        ubox = u["source_box"]
        overlap = False
        area_u = (ubox["x1"] - ubox["x0"]) * (ubox["y1"] - ubox["y0"])
        for s in sam_dets:
            sbox = s["source_box"]
            ix0 = max(ubox["x0"], sbox["x0"])
            iy0 = max(ubox["y0"], sbox["y0"])
            ix1 = min(ubox["x1"], sbox["x1"])
            iy1 = min(ubox["y1"], sbox["y1"])
            if ix1 > ix0 and iy1 > iy0:
                inter_area = (ix1 - ix0) * (iy1 - iy0)
                area_s = (sbox["x1"] - sbox["x0"]) * (sbox["y1"] - sbox["y0"])
                iou = inter_area / (area_u + area_s - inter_area) if (area_u + area_s - inter_area) > 0 else 0
                if iou > 0.3 or (area_u > 0 and inter_area / area_u > 0.6):
                    overlap = True
                    break
        if not overlap:
            kept.append(u)
    return kept


class LiveNotConfigured(Exception):
    """Live inference was requested but no API key is configured."""


class BadPhrases(Exception):
    """The requested phrase list failed validation."""


class BadStrategy(Exception):
    """The requested execution strategy is unknown."""


def normalize_phrases(phrases: list[str] | None) -> list[str]:
    """Default set when None; otherwise validated, de-duplicated phrases."""
    if phrases is None:
        return default_prompts()
    if not isinstance(phrases, list) or not phrases:
        raise BadPhrases("phrases must be a non-empty list")
    if len(phrases) > MAX_PHRASES_PER_RUN:
        raise BadPhrases(
            f"at most {MAX_PHRASES_PER_RUN} phrases per run "
            f"(got {len(phrases)})"
        )
    cleaned: list[str] = []
    for phrase in phrases:
        if not isinstance(phrase, str) or not phrase.strip():
            raise BadPhrases("each phrase must be a non-empty string")
        if len(phrase) > 200:
            raise BadPhrases(f"phrase too long (200 chars max): {phrase[:40]!r}")
        if phrase.strip() not in cleaned:
            cleaned.append(phrase.strip())
    return cleaned


def execute_run(
    inspections: InspectionStore,
    runs: RunStore,
    inspection_id: str,
    config: Config,
    *,
    confirm_live: bool = False,
    phrases: list[str] | None = None,
    transport: Transport | None = None,
    timeout: float = 180.0,
    strategy: str | None = None,
    allow_reinspect: bool = True,
    architecture: str = "baseline",
    allow_offline_fallback: bool = False,
    wonder_mode: str | None = None,
) -> dict[str, Any]:
    """Run cached-first inference for one inspection. Returns the record.

    ``strategy`` is None/``"single"`` (default, whole-image cells only)
    or ``"multiview"`` (whole + selective tiles + reinspect crops with
    reconciled merge records). Anything else raises BadStrategy.
    Safety rules (confirm_live gate, no retries, per-cell failures)
    apply identically to every view of every strategy.
    """
    if strategy is None or strategy == prov.STRATEGY_SINGLE:
        return _execute_single(inspections, runs, inspection_id, config,
                               confirm_live=confirm_live, phrases=phrases,
                               transport=transport, timeout=timeout,
                               architecture=architecture,
                               allow_offline_fallback=allow_offline_fallback,
                               wonder_mode=wonder_mode)
    if strategy == prov.STRATEGY_MULTIVIEW:
        return _execute_multiview(inspections, runs, inspection_id, config,
                                  confirm_live=confirm_live, phrases=phrases,
                                  transport=transport, timeout=timeout,
                                  allow_reinspect=allow_reinspect,
                                  architecture=architecture,
                                  allow_offline_fallback=allow_offline_fallback,
                                  wonder_mode=wonder_mode)
    raise BadStrategy(f"unknown strategy: {strategy!r} "
                      f"(want {prov.STRATEGY_SINGLE!r} "
                      f"or {prov.STRATEGY_MULTIVIEW!r})")


def _execute_single(
    inspections: InspectionStore,
    runs: RunStore,
    inspection_id: str,
    config: Config,
    *,
    confirm_live: bool = False,
    phrases: list[str] | None = None,
    transport: Transport | None = None,
    timeout: float = 180.0,
    architecture: str = "baseline",
    allow_offline_fallback: bool = False,
    wonder_mode: str | None = None,
) -> dict[str, Any]:
    """Single-view execution (default). Record shape is frozen.

    Per phrase: an exact cache hit (same model/phrase/image-sha/mime,
    i.e. the identical sanitized trace) is reused with zero transport
    calls. Without a hit and without ``confirm_live``, the cell stays
    ``uncached``. With confirmation, one live attempt is made; failures
    are recorded per cell and never corrupt cached results.
    """
    wanted = normalize_phrases(phrases)
    record = inspections.get(inspection_id)
    image_path = inspections.dir_for(inspection_id) / record["stored_image"]
    raw = image_path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != record["sha256"]:
        raise RuntimeError("stored image hash mismatch; refusing to run")

    if architecture == "unet_standalone":
        kept = _run_unet_detections(raw)
        status = "complete"
        cells = [{"phrase": "crack", "status": "cached", "run_id": "crack_unet", "detections": len(kept), "empty": not kept}]
        record["status"] = status
        record["run"] = {
            "created_utc": utcnow(),
            "confirm_live": False,
            "phrases": ["crack"],
            "cells": cells,
            "cache_hits": 1,
            "live_calls": 0,
            "failures": 0,
            "model": "crack_unet",
            "merged": {
                "crack": {
                    "kept": kept,
                    "suppressed": [],
                    "views_seen": [{"view": "whole", "view_kind": "whole", "status": "cached", "run_id": "crack_unet"}],
                }
            },
        }
        record["audit"] = {
            "selected_architecture": "unet_standalone",
            "actual_architecture_executed": "unet_standalone",
            "model_id": "crack_unet",
            "model_version": "1.0",
            "inference_status": status,
            "fallback_reason": None,
            "cache_usage": "offline_local",
            "wonder_mode": wonder_mode or "disabled",
        }
        with io_lock:
            assets = inspections.dir_for(inspection_id) / "assets"
            report_html = inspections.dir_for(inspection_id) / "report.html"
            if assets.is_dir():
                for child in assets.iterdir():
                    if child.is_file():
                        child.unlink()
            if report_html.is_file():
                report_html.unlink()
            return inspections.update(inspection_id, record)

    cells: list[dict[str, Any]] = []
    for phrase in wanted:
        request = build_image_request(
            phrase, base64.b64encode(raw).decode("ascii"), record["mime"]
        )
        trace = sanitized_request(request, digest)
        run_id = runs.find_cached(trace)
        if run_id is not None:
            cells.append({
                "phrase": phrase, "status": "cached", "run_id": run_id,
            })
        else:
            cells.append({
                "phrase": phrase, "status": "uncached", "run_id": None,
                "_trace": trace,
            })

    uncached = [c for c in cells if c["status"] == "uncached"]
    live_calls = 0
    if uncached and confirm_live:
        if not config.model_api_key:
            raise LiveNotConfigured(
                "MODEL_API_KEY is not set (env or .env); "
                "nothing was sent and no cached results changed"
            )
        auth_headers(config)  # raises before any billing if key vanished
        adapter = SAM31Adapter(config, transport or UrllibTransport())
        for cell in uncached:
            try:
                result = adapter.segment_image(
                    raw, cell["phrase"], record["mime"], timeout=timeout
                )
            except Exception as exc:  # per-cell failure, others continue
                cell["status"] = "failed"
                cell["error"] = f"{type(exc).__name__}: {exc}"
                continue
            run_id = runs.save(
                cell["_trace"],
                result.raw_response,
                RunMeta(
                    purpose="app-inspection",
                    models={"segment": config.sam_model},
                    input_hashes={"image": digest},
                    notes={
                        "phrase": cell["phrase"],
                        "image": str(image_path),
                        "mime": record["mime"],
                        "detections": len(result.detections),
                        "inspection_id": inspection_id,
                    },
                ),
            )
            live_calls += 1
            cell["status"] = "live"
            cell["run_id"] = run_id
            cell["detections"] = len(result.detections)
            cell["empty"] = not result.detections

    for cell in cells:
        cell.pop("_trace", None)

    n_ok = sum(1 for c in cells if c["status"] in ("cached", "live"))
    n_failed = sum(1 for c in cells if c["status"] == "failed")
    if n_ok == len(cells):
        status = "complete"
    elif n_ok or n_failed:
        status = "partial"
    elif n_failed == len(cells) and cells:
        status = "failed"
    else:
        status = "uploaded"  # nothing ran: cached-only with zero hits

    actual_arch = architecture or "baseline"
    fallback_reason = None
    merged = None

    if actual_arch == "hybrid":
        unet_kept = _run_unet_detections(raw)
        sam_crack_dets = []
        for cell in cells:
            if cell.get("phrase") in ("crack", "crack in wall") and cell.get("status") in ("cached", "live"):
                dets, _ = _parsed_cells({}, cell, runs)
                for d in dets:
                    sam_crack_dets.append({
                        "source_box": dict(d["box"]),
                        "run_id": d.get("run_id"),
                        "view": d.get("view", "whole"),
                        "view_kind": d.get("view_kind", "whole"),
                        "provenance": {"model": "sam-3.1", "confidence": "uncalibrated_zero_shot"},
                        "mask_payload": d.get("mask_payload"),
                        "mask_width": d.get("mask_width"),
                        "mask_height": d.get("mask_height"),
                        "mask_encoding": d.get("mask_encoding"),
                    })
        fused = _fuse_hybrid_detections(sam_crack_dets, unet_kept)
        merged = {
            "crack": {
                "kept": fused,
                "suppressed": [],
                "views_seen": [{"view": "whole", "view_kind": "whole", "status": "hybrid", "run_id": "hybrid_fusion"}],
            }
        }
        status = "complete"
    elif actual_arch == "baseline" and status == "failed" and allow_offline_fallback:
        fallback_kept = _run_unet_detections(raw)
        status = "complete"
        actual_arch = "unet_standalone"
        fallback_reason = "Meta SAM 3.1 unavailable; offline fallback active"
        merged = {
            "crack": {
                "kept": fallback_kept,
                "suppressed": [],
                "views_seen": [{"view": "whole", "view_kind": "whole", "status": "offline_local", "run_id": "crack_unet"}],
            }
        }

    record["status"] = status
    record["run"] = {
        "created_utc": utcnow(),
        "confirm_live": confirm_live,
        "phrases": wanted,
        "cells": cells,
        "cache_hits": sum(1 for c in cells if c["status"] == "cached"),
        "live_calls": live_calls,
        "failures": n_failed,
        "model": config.sam_model,
        "merged": merged,
    }
    record["audit"] = {
        "selected_architecture": architecture or "baseline",
        "actual_architecture_executed": actual_arch,
        "model_id": config.sam_model if ("sam" in actual_arch or actual_arch == "baseline") else "crack_unet",
        "model_version": "3.1" if ("sam" in actual_arch or actual_arch == "baseline") else "1.0",
        "inference_status": status,
        "fallback_reason": fallback_reason,
        "cache_usage": "offline_local" if actual_arch == "unet_standalone" else ("cached" if sum(1 for c in cells if c["status"] == "cached") > 0 else ("live_api" if live_calls > 0 else "offline_local")),
        "wonder_mode": wonder_mode or "shadow",
    }
    # Evidence below was rendered from older results, if any. Locked:
    # on-demand rendering may be writing these files right now.
    with io_lock:
        assets = inspections.dir_for(inspection_id) / "assets"
        report_html = inspections.dir_for(inspection_id) / "report.html"
        if assets.is_dir():
            for child in assets.iterdir():
                if child.is_file():
                    child.unlink()
        if report_html.is_file():
            report_html.unlink()
        return inspections.update(inspection_id, record)


def _view_cell(runs: RunStore, phrase: str, view: Any, raw: bytes,
               mime: str, digest: str) -> dict[str, Any]:
    """Cache-checked cell for one (phrase, view) pair (no transport)."""
    request = build_image_request(
        phrase, base64.b64encode(raw).decode("ascii"), mime)
    trace = sanitized_request(request, digest)
    run_id = runs.find_cached(trace)
    cell: dict[str, Any] = {
        "phrase": phrase, "view": view.view_id, "view_kind": view.kind,
        "tile_box": dict(view.box), "run_id": run_id,
        "status": "cached" if run_id is not None else "uncached",
    }
    if run_id is None:
        cell["_trace"] = trace
        cell["_raw"] = raw
        cell["_mime"] = mime
        cell["_digest"] = digest
    return cell


def _execute_uncached(cells: list[dict[str, Any]], *, runs: RunStore,
                      config: Config, transport: Transport | None,
                      timeout: float, image_path: Path,
                      inspection_id: str, source_sha: str) -> int:
    """One live attempt per uncached view cell. Returns live-call count.

    Views have incompatible bytes by construction, so each cell checks
    its own cache key first; unconfirmed cells stay unexecuted.
    """
    uncached = [c for c in cells if c["status"] == "uncached"]
    if not uncached:
        return 0
    if not config.model_api_key:
        raise LiveNotConfigured(
            "MODEL_API_KEY is not set (env or .env); "
            "nothing was sent and no cached results changed"
        )
    auth_headers(config)  # raises before any billing if key vanished
    adapter = SAM31Adapter(config, transport or UrllibTransport())
    live_calls = 0
    for cell in uncached:
        try:
            result = adapter.segment_image(
                cell["_raw"], cell["phrase"], cell["_mime"], timeout=timeout)
        except Exception as exc:  # per-cell failure, others continue
            cell["status"] = "failed"
            cell["error"] = f"{type(exc).__name__}: {exc}"
            continue
        label = (str(image_path) if cell["view_kind"] == "whole"
                 else f"{image_path} :: view {cell['view']}")
        notes = {
            "phrase": cell["phrase"], "image": label,
            "mime": cell["_mime"], "detections": len(result.detections),
            "inspection_id": inspection_id,
        }
        notes.update(prov.view_notes(
            cell["view"], cell["view_kind"], cell["tile_box"],
            source_sha, cell["phrase"]))
        run_id = runs.save(
            cell["_trace"],
            result.raw_response,
            RunMeta(
                purpose="app-inspection",
                models={"segment": config.sam_model},
                input_hashes={"image": cell["_digest"]},
                notes=notes,
            ),
        )
        live_calls += 1
        cell["status"] = "live"
        cell["run_id"] = run_id
        cell["detections"] = len(result.detections)
        cell["empty"] = not result.detections
        cell["_live"] = result.detections
    return live_calls


def _parsed_cells(parsed: dict[str, Any], cell: dict[str, Any],
                  runs: RunStore) -> tuple[list[dict], list[dict]]:
    """Detection dicts for one executed cell + any parse errors."""
    if cell.get("status") not in ("cached", "live") or not cell.get("run_id"):
        return [], []
    run_id = cell["run_id"]
    if run_id in parsed:
        return parsed[run_id]
    try:
        if cell.get("_live") is not None:
            dets = [{
                "phrase": cell["phrase"], "box": d.box.to_dict(),
                "view": cell["view"], "view_kind": cell["view_kind"],
                "tile_box": dict(cell["tile_box"]),
                "frame_width": d.frame_width,
                "frame_height": d.frame_height,
                "run_id": run_id,
                "mask_payload": d.mask_payload,
                "mask_width": d.mask_width,
                "mask_height": d.mask_height,
                "mask_encoding": d.mask_encoding,
            } for d in cell["_live"]]
        else:
            dets = mv_reconcile.detections_from_run(
                runs, run_id, cell["phrase"], cell["view"],
                cell["view_kind"], cell["tile_box"])
        parsed[run_id] = (dets, [])
    except Exception as exc:
        parsed[run_id] = ([], [{
            "view": cell["view"], "run_id": run_id,
            "error": f"{type(exc).__name__}: {exc}"}])
    return parsed[run_id]


def _execute_multiview(
    inspections: InspectionStore,
    runs: RunStore,
    inspection_id: str,
    config: Config,
    *,
    confirm_live: bool = False,
    phrases: list[str] | None = None,
    transport: Transport | None = None,
    timeout: float = 180.0,
    allow_reinspect: bool = True,
    architecture: str = "baseline",
    allow_offline_fallback: bool = False,
    wonder_mode: str | None = None,
) -> dict[str, Any]:
    """Multi-view execution: whole + selective tiles + reinspect + merge.

    Each view is an independent cache-checked cell through the same
    adapter and trace form as single-view (whole cells share cache keys
    with single-view/CLI runs; tile bytes can never collide with them).
    Merged detections are derived artifacts recorded on the inspection
    record only -- never written to the model-response cache.
    """
    wanted = normalize_phrases(phrases)
    record = inspections.get(inspection_id)
    image_path = inspections.dir_for(inspection_id) / record["stored_image"]
    raw = image_path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != record["sha256"]:
        raise RuntimeError("stored image hash mismatch; refusing to run")

    if architecture == "unet_standalone":
        kept = _run_unet_detections(raw)
        status = "complete"
        cells = [{"phrase": "crack", "status": "cached", "run_id": "crack_unet", "detections": len(kept), "empty": not kept}]
        record["status"] = status
        record["run"] = {
            "created_utc": utcnow(),
            "confirm_live": False,
            "phrases": ["crack"],
            "cells": cells,
            "cache_hits": 1,
            "live_calls": 0,
            "failures": 0,
            "model": "crack_unet",
            "merged": {
                "crack": {
                    "kept": kept,
                    "suppressed": [],
                    "views_seen": [{"view": "whole", "view_kind": "whole", "status": "cached", "run_id": "crack_unet"}],
                }
            },
        }
        record["audit"] = {
            "selected_architecture": "unet_standalone",
            "actual_architecture_executed": "unet_standalone",
            "model_id": "crack_unet",
            "model_version": "1.0",
            "inference_status": status,
            "fallback_reason": None,
            "cache_usage": "offline_local",
            "wonder_mode": wonder_mode or "disabled",
        }
        with io_lock:
            assets = inspections.dir_for(inspection_id) / "assets"
            report_html = inspections.dir_for(inspection_id) / "report.html"
            if assets.is_dir():
                for child in assets.iterdir():
                    if child.is_file():
                        child.unlink()
            if report_html.is_file():
                report_html.unlink()
            return inspections.update(inspection_id, record)
    width, height = record["width"], record["height"]

    def go(cells: list[dict[str, Any]]) -> int:
        if not confirm_live:
            return 0
        return _execute_uncached(
            cells, runs=runs, config=config, transport=transport,
            timeout=timeout, image_path=image_path,
            inspection_id=inspection_id, source_sha=digest)

    whole_view = mv_views.whole_view(width, height)
    cells: list[dict[str, Any]] = []
    live_calls = 0
    parsed: dict[str, Any] = {}

    # Phase A: whole-image cells (identical traces to single-view).
    for phrase in wanted:
        cells.append(_view_cell(runs, phrase, whole_view, raw,
                                record["mime"], digest))
    live_calls += go(cells)

    # Phase B: selective tiles per crack-family phrase.
    tile_views = mv_views.tile_views(width, height)
    tile_bytes = mv_raster.view_bytes_map(raw, tile_views)
    for phrase in wanted:
        whole = next(c for c in cells
                     if c["phrase"] == phrase and c["view_kind"] == "whole")
        count = whole.get("detections")
        if count is None and whole.get("status") == "cached":
            dets, _ = _parsed_cells(parsed, whole, runs)
            count = len(dets)
        if count is None:
            count = 0 if whole.get("status") != "failed" else -1
        if count < 0 or not mv_views.should_tile(phrase, width, height,
                                                 count):
            continue
        for view in tile_views:
            view_raw = tile_bytes[view.view_id]
            cells.append(_view_cell(runs, phrase, view, view_raw,
                                    "image/png", sha256_bytes(view_raw)))
    live_calls += go([c for c in cells if c["status"] == "uncached"])

    # Phase C: reinspect crops around kept tiny detections (max 2/image).
    if allow_reinspect:
        tiny: list[dict[str, int]] = []
        for phrase in wanted:
            phrase_dets: list[dict[str, Any]] = []
            for cell in cells:
                if cell["phrase"] != phrase:
                    continue
                dets, _ = _parsed_cells(parsed, cell, runs)
                phrase_dets.extend(dets)
            if not phrase_dets:
                continue
            kept, _, _ = mv_reconcile.reconcile(phrase_dets, width, height,
                                                digest)
            for entry in kept:
                box = entry["source_box"]
                area_frac = ((box["x1"] - box["x0"]) * (box["y1"] - box["y0"])
                             / (width * height))
                if area_frac < TINY_AREA_FRAC:
                    tiny.append(box)
        for view in mv_views.reinspect_views(tiny, width, height):
            view_raw = mv_raster.cut_view(raw, view.box)
            view_digest = sha256_bytes(view_raw)
            # Reinspect views screen the crack phrases that produced them;
            # keep it to phrases already in this run.
            for phrase in wanted:
                if phrase not in mv_views.CRACK_FAMILY:
                    continue
                cells.append(_view_cell(runs, phrase, view, view_raw,
                                        "image/png", view_digest))
        live_calls += go([c for c in cells if c["status"] == "uncached"])

    # Merge executed cells per phrase (derived artifact, never cached).
    merged: dict[str, Any] = {}
    for phrase in wanted:
        phrase_dets = []
        parse_errors: list[dict[str, Any]] = []
        views_seen: list[dict[str, Any]] = []
        for cell in cells:
            if cell["phrase"] != phrase:
                continue
            views_seen.append({
                "view": cell["view"], "view_kind": cell["view_kind"],
                "status": cell["status"], "run_id": cell.get("run_id")})
            dets, errors = _parsed_cells(parsed, cell, runs)
            phrase_dets.extend(dets)
            parse_errors.extend(errors)
        kept, suppressed, rejected = mv_reconcile.reconcile(
            phrase_dets, width, height, digest)
        manifest = prov.merge_manifest(digest, phrase, kept, suppressed,
                                       views_seen)
        manifest["rejected"] = rejected
        manifest["parse_errors"] = parse_errors
        manifest["counts"]["rejected"] = len(rejected)
        merged[phrase] = manifest

    for cell in cells:
        for key in ("_trace", "_raw", "_mime", "_digest", "_live"):
            cell.pop(key, None)

    n_ok = sum(1 for c in cells if c["status"] in ("cached", "live"))
    n_failed = sum(1 for c in cells if c["status"] == "failed")
    if n_ok == len(cells):
        status = "complete"
    elif n_ok or n_failed:
        status = "partial"
    elif n_failed == len(cells) and cells:
        status = "failed"
    else:
        status = "uploaded"

    actual_arch = architecture or "baseline"
    fallback_reason = None

    if actual_arch == "hybrid":
        unet_kept = _run_unet_detections(raw)
        crack_manifest = merged.get("crack")
        sam_crack_dets = list(crack_manifest.get("kept", [])) if crack_manifest else []
        fused = _fuse_hybrid_detections(sam_crack_dets, unet_kept)
        if crack_manifest is None:
            merged["crack"] = {
                "source_image_sha256": digest,
                "phrase": "crack",
                "kept": fused,
                "suppressed": [],
                "rejected": [],
                "parse_errors": [],
                "views_seen": [{"view": "whole", "view_kind": "whole", "status": "hybrid", "run_id": "hybrid_fusion"}],
                "counts": {"kept": len(fused), "suppressed": 0, "rejected": 0, "views_seen": 1},
            }
        else:
            crack_manifest["kept"] = fused
            crack_manifest["counts"]["kept"] = len(fused)
        status = "complete"
    elif actual_arch == "baseline" and status == "failed" and allow_offline_fallback:
        fallback_kept = _run_unet_detections(raw)
        merged["crack"] = {
            "source_image_sha256": digest,
            "phrase": "crack",
            "kept": fallback_kept,
            "suppressed": [],
            "rejected": [],
            "parse_errors": [],
            "views_seen": [{"view": "whole", "view_kind": "whole", "status": "cached", "run_id": "crack_unet"}],
            "counts": {"kept": len(fallback_kept), "suppressed": 0, "rejected": 0, "views_seen": 1},
        }
        status = "complete"
        actual_arch = "unet_standalone"
        fallback_reason = "Meta SAM 3.1 unavailable; offline fallback active"

    def stats(kind: str) -> dict[str, int]:
        kind_cells = [c for c in cells if c["view_kind"] == kind]
        out = {"planned": len(kind_cells)}
        for state in ("cached", "live", "uncached", "failed"):
            out[state] = sum(1 for c in kind_cells if c["status"] == state)
        return out

    record["status"] = status
    record["run"] = {
        "created_utc": utcnow(),
        "confirm_live": confirm_live,
        "strategy": prov.STRATEGY_MULTIVIEW,
        "phrases": wanted,
        "cells": cells,
        "cache_hits": sum(1 for c in cells if c["status"] == "cached"),
        "live_calls": live_calls,
        "failures": n_failed,
        "model": config.sam_model if ("sam" in actual_arch or actual_arch == "baseline") else "crack_unet",
        "merged": merged,
        "view_stats": {kind: stats(kind)
                       for kind in ("whole", "tile", "reinspect")},
        "projected_cost_usd": round(live_calls * PRICE_PER_IMAGE_USD, 4),
    }
    record["audit"] = {
        "selected_architecture": architecture or "baseline",
        "actual_architecture_executed": actual_arch,
        "model_id": config.sam_model if ("sam" in actual_arch or actual_arch == "baseline") else "crack_unet",
        "model_version": "3.1" if ("sam" in actual_arch or actual_arch == "baseline") else "1.0",
        "inference_status": status,
        "fallback_reason": fallback_reason,
        "cache_usage": "offline_local" if actual_arch == "unet_standalone" else ("cached" if sum(1 for c in cells if c["status"] == "cached") > 0 else ("live_api" if live_calls > 0 else "offline_local")),
        "wonder_mode": wonder_mode or "shadow",
    }
    with io_lock:
        assets = inspections.dir_for(inspection_id) / "assets"
        report_html = inspections.dir_for(inspection_id) / "report.html"
        if assets.is_dir():
            for child in assets.iterdir():
                if child.is_file():
                    child.unlink()
        if report_html.is_file():
            report_html.unlink()
        return inspections.update(inspection_id, record)
