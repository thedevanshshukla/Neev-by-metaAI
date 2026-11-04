"""Overlapping-tile helpers for the crack tiling experiment.

Pure geometry: no network, no filesystem, no model calls. Tiles are
derived from native image dimensions; predictions map back by offset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Tile:
    tile_id: str
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def _axis_starts(length: int, count: int, overlap: float) -> list[tuple[int, int]]:
    """Cover [0, length) with `count` spans at ~`overlap` fraction.

    Span size t solves t + (count-1)*t*(1-overlap) = length, rounded up;
    first span starts at 0 and the last ends exactly at `length`, so all
    borders are covered and nothing runs out of bounds.
    """
    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    size = math.ceil(length / (count - overlap * (count - 1)))
    size = min(size, length)
    if count == 1:
        return [(0, length)]
    step = (length - size) / (count - 1)
    spans = []
    for i in range(count):
        start = round(i * step) if i else 0
        end = start + size
        if i == count - 1:
            end, start = length, length - size
        spans.append((start, end))
    return spans


def compute_tiles(width: int, height: int, rows: int = 2, cols: int = 2,
                  overlap: float = 0.25) -> list[Tile]:
    """2D grid of overlapping tiles covering the whole image."""
    xs = _axis_starts(width, cols, overlap)
    ys = _axis_starts(height, rows, overlap)
    return [Tile(tile_id=f"r{r}c{c}", x0=x0, y0=y0, x1=x1, y1=y1)
            for r, (y0, y1) in enumerate(ys)
            for c, (x0, x1) in enumerate(xs)]


def translate_box(box: dict[str, Any], dx: float, dy: float) -> dict[str, Any]:
    """Shift a box by a tile offset, preserving extra keys."""
    out = dict(box)
    out["x0"] = box["x0"] + dx
    out["x1"] = box["x1"] + dx
    out["y0"] = box["y0"] + dy
    out["y1"] = box["y1"] + dy
    return out


def clip_box(box: dict[str, Any], width: int, height: int
             ) -> dict[str, Any] | None:
    """Clip a box to the native frame; None when nothing remains."""
    out = dict(box)
    out["x0"] = max(0, min(width, box["x0"]))
    out["x1"] = max(0, min(width, box["x1"]))
    out["y0"] = max(0, min(height, box["y0"]))
    out["y1"] = max(0, min(height, box["y1"]))
    if not (out["x0"] < out["x1"] and out["y0"] < out["y1"]):
        return None
    return out


def aabb_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Axis-aligned box IoU (tile predictions are axis-aligned)."""
    ix0, iy0 = max(a["x0"], b["x0"]), max(a["y0"], b["y0"])
    ix1, iy1 = min(a["x1"], b["x1"]), min(a["y1"], b["y1"])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    aa = (a["x1"] - a["x0"]) * (a["y1"] - a["y0"])
    bb = (b["x1"] - b["x0"]) * (b["y1"] - b["y0"])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def aabb_iomin(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Intersection over minimum area (containment score)."""
    ix0, iy0 = max(a["x0"], b["x0"]), max(a["y0"], b["y0"])
    ix1, iy1 = min(a["x1"], b["x1"]), min(a["y1"], b["y1"])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    aa = (a["x1"] - a["x0"]) * (a["y1"] - a["y0"])
    bb = (b["x1"] - b["x0"]) * (b["y1"] - b["y0"])
    min_area = min(aa, bb)
    return inter / min_area if min_area > 0 else 0.0


def _area(box: dict[str, Any]) -> float:
    return (box["x1"] - box["x0"]) * (box["y1"] - box["y0"])


def merge_cross_tile(boxes: list[dict[str, Any]], iou_threshold: float = 0.5,
                     iomin_threshold: float | None = None
                     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Same-class cross-tile NMS with a deterministic survivor rule.

    No confidence scores exist, so the largest-area box survives each
    duplicate cluster (ties broken by coordinates). Returns
    (kept, suppressed); inputs are never mutated.
    If ``iomin_threshold`` is provided, smaller boxes whose area is
    heavily contained inside a kept larger box (IoMin >= threshold)
    are also suppressed as duplicate fragments.
    """
    order = sorted(range(len(boxes)),
                   key=lambda i: (-_area(boxes[i]), boxes[i]["x0"],
                                   boxes[i]["y0"], boxes[i]["x1"],
                                   boxes[i]["y1"], i))
    kept_idx: list[int] = []
    suppressed: list[dict[str, Any]] = []
    for i in order:
        is_dup = False
        for k in kept_idx:
            if aabb_iou(boxes[i], boxes[k]) >= iou_threshold:
                is_dup = True
                break
            if (iomin_threshold is not None
                    and aabb_iomin(boxes[i], boxes[k]) >= iomin_threshold):
                is_dup = True
                break
        if is_dup:
            suppressed.append(boxes[i])
        else:
            kept_idx.append(i)
    return [boxes[i] for i in kept_idx], suppressed
