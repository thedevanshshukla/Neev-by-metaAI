"""Multi-view provenance records (pure data builders, no I/O).

Every multi-view finding links back to its source image, view, tile
coordinates, prompt, and originating run id. Merge outputs are derived
artifacts: they are recorded in inspection run records and merge
manifests, never in the model-response cache.
"""

from __future__ import annotations

from typing import Any

STRATEGY_SINGLE = "single"
STRATEGY_MULTIVIEW = "multiview"


def view_notes(view_id: str, view_kind: str, box: dict[str, int],
               source_sha: str, phrase: str) -> dict[str, Any]:
    """RunMeta notes extension identifying a view cell (JSON-safe)."""
    return {
        "strategy": STRATEGY_MULTIVIEW,
        "view": view_id,
        "view_kind": view_kind,
        "source_sha256": source_sha,
        "tile_box": dict(box),
        "phrase": phrase,
    }


def finding_provenance(view_id: str, view_kind: str, box: dict[str, int],
                       source_sha: str, phrase: str,
                       run_id: str | None) -> dict[str, Any]:
    """Per-finding provenance attached to reconciled detections."""
    return {
        "source_sha256": source_sha,
        "view": view_id,
        "view_kind": view_kind,
        "tile_box": dict(box),
        "phrase": phrase,
        "run_id": run_id,
    }


def merge_manifest(source_sha: str, phrase: str, kept: list[dict[str, Any]],
                   suppressed: list[dict[str, Any]],
                   views: list[dict[str, Any]]) -> dict[str, Any]:
    """Derived-artifact record for one reconciled (image, prompt) pair."""
    return {
        "artifact": "multiview-merge",
        "source_sha256": source_sha,
        "phrase": phrase,
        "views": views,
        "kept": kept,
        "suppressed": suppressed,
        "counts": {"kept": len(kept), "suppressed": len(suppressed)},
    }
