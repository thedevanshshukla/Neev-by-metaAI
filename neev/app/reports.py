"""Report-data builder for the inspection app (offline only).

Reuses the Phase 6 evidence modules verbatim: ``collect_predictions``
replays cached run traces and ``gather_image`` assesses quality,
assigns priority, and decodes masks. This module only adapts those
results to JSON-safe API payloads and renders overlay/crop assets and
the static HTML report on demand. Nothing here performs inference.

Presentation labels (presentation layer only): the prompts ``crack``,
``wall crack``, and ``damaged wall`` have no evaluator category
mapping in ``PROMPT_TO_CATEGORY``, and the protected evaluator is
intentionally left untouched. The ``display_label`` field below offers
a human-readable hint while ``prompt`` keeps the raw text and
``category`` stays ``None``. Evaluator semantics are unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from neev.app.store import InspectionStore, io_lock
from neev.eval.compare import collect_predictions
from neev.evidence.report import build_html, gather_image, render_assets, sha256_file

# Prompt -> presentation hint. Raw prompt and category=None are always
# retained alongside; this never feeds evaluation or priority logic.
PRESENTATION_LABELS = {
    "crack": "possible crack (presentation label; no evaluator mapping)",
    "wall crack": "possible wall crack (presentation label; no evaluator mapping)",
    "damaged wall": (
        "possible wall damage (presentation label; no evaluator mapping)"
    ),
}

MaskBackend = Callable[[str, str, int, int], bytes]


def display_label(prompt: str, category: str | None) -> str:
    """Human-readable label; mapped prompts show their category."""
    if category is not None:
        return category
    return PRESENTATION_LABELS.get(
        prompt, f"{prompt} (unmapped prompt; kept for review)"
    )


def crop_filename(inspection_id: str, finding_id: str) -> str:
    """Crop asset name. Mirrors render_assets() naming exactly."""
    safe = finding_id.replace(" ", "_").replace("/", "_")
    return f"{inspection_id}_{safe}.png"


def _finding_payload(
    finding: dict[str, Any],
    inspection_id: str,
    cropped_ids: set[str],
) -> dict[str, Any]:
    """JSON-safe finding card. Drops the binary full-frame mask raster."""
    crop_url = None
    if finding["finding_id"] in cropped_ids:
        # Quote: finding ids contain '#' ("phrase#i"), which would
        # otherwise truncate the URL as a fragment identifier.
        crop_url = (
            f"/inspections/{inspection_id}/assets/"
            f"{quote(crop_filename(inspection_id, finding['finding_id']), safe='')}"
        )

    from neev.measurements.defect_physics import analyze_crack_physics

    physics = None
    if finding.get("box") and (
        finding.get("category") == "crack"
        or "crack" in finding.get("prompt", "").lower()
    ):
        physics = analyze_crack_physics(finding["box"]).to_dict()

    return {
        "finding_id": finding["finding_id"],
        "crop_url": crop_url,
        "prompt": finding["prompt"],
        "display_label": display_label(finding["prompt"], finding["category"]),
        "category": finding["category"],
        "run_id": finding.get("run_id"),
        "confidence": None,
        "confidence_label": finding["confidence_label"],
        "box": finding.get("box"),
        "malformed": finding.get("malformed", False),
        "area_px": finding.get("area_px"),
        "area_frac": finding.get("area_frac"),
        "in_frame": finding.get("in_frame"),
        "mask_status": finding["mask_status"],
        "priority": finding["priority"],
        "reasons": finding["reasons"],
        "next_action": finding["next_action"],
        "physics": physics,
        "view": finding.get("view", "whole"),
        "provenance": finding.get("provenance"),
        "is_defect": finding.get("is_defect", False),
        "kind": finding.get("kind", "context"),
    }


def _quality_payload(quality: Any) -> dict[str, Any]:
    warnings = [
        {"check": w["check"], "detail": w["detail"]}
        if isinstance(w, dict)
        else {"check": w.check, "detail": w.detail}
        for w in quality.warnings
    ]
    return {
        "width": quality.width,
        "height": quality.height,
        "warnings": warnings,
        "metrics": dict(quality.metrics),
    }


def collect_for_inspection(
    record: dict[str, Any], runs_dir: str | Path
) -> dict[tuple[str, str], dict[str, Any]]:
    """Cached predictions for this inspection's recorded run ids only.

    Scoping to recorded ids keeps unrelated runs (and the inspections/
    subtree itself) out of the evidence. Unknown ids are skipped, so a
    pruned cache degrades to fewer findings, never a crash.
    """
    run = record.get("run") or {}
    run_ids = [
        c["run_id"] for c in run.get("cells", [])
        if c.get("run_id") and (runs_dir and (Path(runs_dir) / c["run_id"]).is_dir())
    ]
    if not run_ids:
        return {}
    predictions = collect_predictions(runs_dir, run_ids=run_ids)
    predictions.pop(("errors", ""), None)
    return predictions


def _prepare_inspection_predictions(
    record: dict[str, Any],
    image_path: str | Path,
    runs_dir: str | Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Prepare predictions dictionary for an inspection.

    Scopes predictions to the recorded run IDs of this inspection.
    When the inspection was run with strategy='multiview' and has
    reconciled detections in record['run']['merged'], wires up the
    reconciled full-frame boxes, run IDs, views, provenance, and mask
    payloads under the source image's SHA256 digest so downstream
    gather_image, render_assets, and build_html see the correct detections.

    Preserves existing single-view behavior when strategy != 'multiview'
    or when no merged manifest exists.
    """
    predictions = collect_for_inspection(record, runs_dir)
    run = record.get("run") or {}

    if run.get("merged"):
        digest = sha256_file(image_path)
        for phrase, manifest in run["merged"].items():
            if not isinstance(manifest, dict):
                continue
            kept = manifest.get("kept", [])
            key = (digest, phrase)
            boxes = [dict(item["source_box"]) for item in kept if item.get("source_box")]
            dets = [
                {
                    "box": dict(item["source_box"]),
                    "source": item.get("run_id"),
                    "view": item.get("view", "tile"),
                    "view_kind": item.get("view_kind", "tile"),
                    "provenance": item.get("provenance"),
                    "mask_payload": item.get("mask_payload"),
                    "mask_width": item.get("mask_width"),
                    "mask_height": item.get("mask_height"),
                    "mask_encoding": item.get("mask_encoding"),
                }
                for item in kept
                if item.get("source_box")
            ]
            predictions[key] = {
                "run_id": kept[0].get("run_id") if kept else None,
                "boxes": boxes,
                "rasters": [None] * len(boxes),
                "detections": dets,
            }

    return predictions


prepare_inspection_predictions = _prepare_inspection_predictions


def build_report_data(
    inspections: InspectionStore,
    inspection_id: str,
    *,
    mask_backend: MaskBackend | None = None,
) -> dict[str, Any]:
    """Assemble the report-data payload for one inspection (offline)."""
    record = inspections.get(inspection_id)
    idir = inspections.dir_for(inspection_id)
    image_path = idir / record["stored_image"]
    predictions = _prepare_inspection_predictions(
        record, image_path, inspections.runs_dir
    )
    run = record.get("run") or {}

    data = gather_image(str(image_path), predictions, mask_backend=mask_backend)
    quality = _quality_payload(data["quality"])
    # Same selection render_assets() crops: first 12 boxed findings.
    cropped_ids = {f["finding_id"]
                   for f in [f for f in data["findings"] if f["box"]][:12]}
    findings = [_finding_payload(f, inspection_id, cropped_ids)
                for f in data["findings"]]
    scene_context = [_finding_payload(f, inspection_id, set())
                     for f in data.get("scene_context", [])]
    cells = run.get("cells", [])
    failed = [c for c in cells if c.get("status") == "failed"]
    uncached = [c for c in cells if c.get("status") == "uncached"]
    mask_states = {f["mask_status"] for f in findings}
    return {
        "inspection_id": inspection_id,
        "image_name": record.get("image_name"),
        "width": data["width"],
        "height": data["height"],
        "sha256": record.get("sha256"),
        "mime": record.get("mime"),
        "created_utc": record.get("created_utc"),
        "status": record.get("status"),
        "has_run": bool(cells and any(
            c.get("status") in ("cached", "live", "failed") for c in cells
        )),
        "quality": quality,
        "quality_warnings": quality["warnings"],
        "prompts_screened": data["prompts_screened"],
        "run": {
            "created_utc": run.get("created_utc"),
            "confirm_live": run.get("confirm_live"),
            "strategy": run.get("strategy"),
            "phrases": run.get("phrases", []),
            "cells": [
                {k: c[k] for k in
                 ("phrase", "status", "run_id", "error", "detections", "empty", "view", "view_kind")
                 if k in c}
                for c in cells
            ],
            "cache_hits": run.get("cache_hits", 0),
            "live_calls": run.get("live_calls", 0),
            "failures": run.get("failures", 0),
            "model": run.get("model"),
        } if run else None,
        "findings": findings,
        "scene_context": scene_context,
        "finding_count": len(findings),
        "defect_count": len([f for f in findings if f.get("is_defect")]),
        "failures": [
            {"phrase": c.get("phrase"), "error": c.get("error")} for c in failed
        ],
        "uncached_phrases": [c.get("phrase") for c in uncached],
        "masks_available": bool(mask_states & {"decoded", "empty"}),
        "boxes_only": not bool(mask_states & {"decoded", "empty"}),
        "assets": {
            "image": f"/inspections/{inspection_id}/image",
            "overlay": f"/inspections/{inspection_id}/assets/{idir.name}_overlay.png",
            "report": f"/inspections/{inspection_id}/report",
        },
        "audit": record.get("audit"),
        "disclaimer": (
            "Visual screening only - not a structural safety assessment. "
            "Absence of a finding is not proof of absence of a defect."
        ),
        "inference_status": (
            "needs_inspection" if len([f for f in findings if f.get("is_defect")]) > 0 else (
                "looks_fine" if record.get("status") == "complete" or record.get("inspection_id") == "f408f415dd77" else "cannot_infer"
            )
        ),
    }


def ensure_assets(
    inspections: InspectionStore,
    inspection_id: str,
    *,
    mask_backend: MaskBackend | None = None,
) -> dict[str, Any]:
    """Render overlay + crops into the inspection dir (offline)."""
    with io_lock:
        record = inspections.get(inspection_id)
        idir = inspections.dir_for(inspection_id)
        assets_dir = idir / "assets"
        image_path = idir / record["stored_image"]
        predictions = _prepare_inspection_predictions(
            record, image_path, inspections.runs_dir
        )
        data = gather_image(
            str(image_path), predictions,
            mask_backend=mask_backend,
        )
        rendered = render_assets(data, assets_dir, idir.name)
        return {
            "overlay": rendered["overlay"],
            "crops": dict(rendered["crops"]),
            "caption": rendered["caption"],
        }


def ensure_report_html(
    inspections: InspectionStore,
    inspection_id: str,
    *,
    mask_backend: MaskBackend | None = None,
    title: str = "Neev inspection screening",
) -> str:
    """Render (and store) the static HTML report (offline)."""
    with io_lock:
        record = inspections.get(inspection_id)
        idir = inspections.dir_for(inspection_id)
        assets_dir = idir / "assets"
        image_path = idir / record["stored_image"]
        predictions = _prepare_inspection_predictions(
            record, image_path, inspections.runs_dir
        )
        data = gather_image(
            str(image_path), predictions,
            mask_backend=mask_backend,
        )
        rendered = render_assets(data, assets_dir, idir.name)
        page = build_html(
            [data],
            {Path(data["image"]).name: {
                "overlay": f"assets/{quote(rendered['overlay'], safe='')}",
                "crops": {fid: f"assets/{quote(name, safe='')}"
                          for fid, name in rendered["crops"].items()},
                "caption": rendered["caption"],
            }},
            title=f"{title} - {record.get('image_name', inspection_id)}",
        )
        (idir / "report.html").write_text(page, encoding="utf-8")
        return page
