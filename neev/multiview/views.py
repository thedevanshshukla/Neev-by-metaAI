"""Deterministic multi-view planning (pure geometry, no inference).

A view is one model input derived from a source image: the whole frame,
one overlapping tile, or one selective reinspection crop. Planning is a
pure function of image dimensions, prompt, and whole-view outcome, so
the same inspection always plans the same views.

Tile geometry reuses ``neev.eval.tiles.compute_tiles`` (2x2, overlap
0.25 by default -- the audited pilot grid). View ids are stable:
``whole``, ``r{row}c{col}``, ``re{index}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neev.eval.tiles import compute_tiles

# Prompts eligible for tiled views. Cracks are thin and localized, so
# only the crack family tiles; every other prompt stays whole-image.
CRACK_FAMILY = frozenset({
    "crack in wall",
    "crack",
    "wall crack",
    "damaged wall",
    "cracked wall",
    "plaster crack",
})

# Audited selective-execution policy (see Phase 8 plan):
# - tiles only at or above this smallest side (small images gain nothing)
# - tiles run when the whole view found nothing (recall rescue) or the
#   image is large enough that thin defects may hide at full-frame scale
MIN_TILE_SIDE_PX = 480
LARGE_IMAGE_AREA_PX = 1_000_000

# Reinspection crop limits: at most two crops per image, each an
# expansion of a kept tiny detection, clamped to the frame.
MAX_REINSPECT_CROPS = 2
REINSPECT_EXPAND = 1.5
REINSPECT_MIN_SIDE_PX = 256

TILE_ROWS = 2
TILE_COLS = 2
TILE_OVERLAP = 0.25


@dataclass(frozen=True)
class View:
    """One model input. ``box`` is source-image coords (whole = frame)."""

    view_id: str
    kind: str  # "whole" | "tile" | "reinspect"
    box: dict[str, int]
    source_width: int
    source_height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "kind": self.kind,
            "box": dict(self.box),
            "source_width": self.source_width,
            "source_height": self.source_height,
        }


def whole_view(width: int, height: int) -> View:
    """The full frame as a view."""
    if width <= 0 or height <= 0:
        raise ValueError(f"bad image dims: {width}x{height}")
    return View(view_id="whole", kind="whole",
                box={"x0": 0, "y0": 0, "x1": width, "y1": height},
                source_width=width, source_height=height)


def tile_views(width: int, height: int) -> list[View]:
    """Overlapping tiles covering the frame (audited pilot grid)."""
    whole_view(width, height)  # validates dims
    return [View(view_id=t.tile_id, kind="tile",
                 box={"x0": t.x0, "y0": t.y0, "x1": t.x1, "y1": t.y1},
                 source_width=width, source_height=height)
            for t in compute_tiles(width, height, rows=TILE_ROWS,
                                   cols=TILE_COLS, overlap=TILE_OVERLAP)]


def should_tile(prompt: str, width: int, height: int,
                whole_detections: int) -> bool:
    """Selective tile trigger (audited policy). Pure predicate."""
    if prompt not in CRACK_FAMILY:
        return False
    if min(width, height) < MIN_TILE_SIDE_PX:
        return False
    if whole_detections == 0:
        return True  # recall rescue: whole view saw nothing
    return width * height >= LARGE_IMAGE_AREA_PX


def expand_box(box: dict[str, int], factor: float,
               width: int, height: int) -> dict[str, int] | None:
    """Expand a box about its center, clamp to frame. None if degenerate."""
    try:
        x0, y0, x1, y1 = (box[k] for k in ("x0", "y0", "x1", "y1"))
    except (KeyError, TypeError):
        return None
    if not all(isinstance(v, int) and not isinstance(v, bool)
               for v in (x0, y0, x1, y1)) or not (x0 < x1 and y0 < y1):
        return None
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = (x1 - x0) * factor / 2.0, (y1 - y0) * factor / 2.0
    out = {"x0": max(0, int(cx - hw)), "y0": max(0, int(cy - hh)),
           "x1": min(width, int(cx + hw) + 1),
           "y1": min(height, int(cy + hh) + 1)}
    if not (out["x0"] < out["x1"] and out["y0"] < out["y1"]):
        return None
    return out


def _grow_to_min(box: dict[str, int], min_side: int,
                   width: int, height: int) -> dict[str, int] | None:
    """Grow a box to at least ``min_side`` per side, slid into the frame.

    A tiny detection becomes a minimum context window centered on it.
    None when the frame itself is smaller than the minimum.
    """
    if width < min_side or height < min_side:
        return None
    w, h = box["x1"] - box["x0"], box["y1"] - box["y0"]
    cx, cy = (box["x0"] + box["x1"]) / 2.0, (box["y0"] + box["y1"]) / 2.0
    hw, hh = max(w, min_side) / 2.0, max(h, min_side) / 2.0
    x0, y0 = int(cx - hw), int(cy - hh)
    x0 = max(0, min(width - int(hw * 2), x0))
    y0 = max(0, min(height - int(hh * 2), y0))
    return {"x0": x0, "y0": y0,
            "x1": min(width, x0 + int(hw * 2)),
            "y1": min(height, y0 + int(hh * 2))}


def reinspect_views(boxes: list[dict[str, int]], width: int, height: int,
                    max_crops: int = MAX_REINSPECT_CROPS) -> list[View]:
    """Plan at most ``max_crops`` reinspection crops from kept boxes.

    Callers pass only boxes worth a closer look (tiny detections).
    Each crop expands about the box, grows to a minimum context window,
    de-duplicates identical windows, and stays deterministic. Degenerate
    boxes drop out; frames smaller than the minimum yield no crops.
    """
    whole_view(width, height)  # validates dims
    views: list[View] = []
    seen: set[tuple[int, int, int, int]] = set()
    for box in boxes:
        if len(views) >= max_crops:
            break
        grown = expand_box(box, REINSPECT_EXPAND, width, height)
        if grown is None:
            continue
        grown = _grow_to_min(grown, REINSPECT_MIN_SIDE_PX, width, height)
        if grown is None:
            continue
        key = (grown["x0"], grown["y0"], grown["x1"], grown["y1"])
        if key in seen:
            continue
        seen.add(key)
        views.append(View(view_id=f"re{len(views)}", kind="reinspect",
                          box=grown, source_width=width,
                          source_height=height))
    return views
