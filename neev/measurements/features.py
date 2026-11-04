"""Derived building measurements from SAM boxes/masks (pure functions).

Threshold defaults come from docs.md section 10 and are parameters, not
constants buried in code, so experiments can sweep them. Every function
returns its inputs alongside the verdict for traceability.
"""

from __future__ import annotations

from dataclasses import dataclass

from neev.adapters.base import BoundingBox


def cluster_rows(values: list[float], tolerance: float) -> list[list[float]]:
    """Group 1-D positions into rows; a gap over tolerance starts a row."""
    rows: list[list[float]] = []
    for value in sorted(values):
        if rows and value - rows[-1][-1] <= tolerance:
            rows[-1].append(value)
        else:
            rows.append([value])
    return rows


@dataclass(frozen=True)
class StoreyCount:
    rows: int
    row_centers_y: list[float]
    windows_used: int


def count_storeys(
    window_boxes: list[BoundingBox],
    image_height_px: int,
    row_tolerance_frac: float = 0.03,
) -> StoreyCount:
    """Each row of window centres is one storey (docs.md section 10)."""
    if not window_boxes or image_height_px <= 0:
        return StoreyCount(rows=0, row_centers_y=[], windows_used=0)
    centers = [(b.y0 + b.y1) / 2 for b in window_boxes]
    rows = cluster_rows(centers, row_tolerance_frac * image_height_px)
    return StoreyCount(
        rows=len(rows),
        row_centers_y=[sum(r) / len(r) for r in rows],
        windows_used=len(window_boxes),
    )


@dataclass(frozen=True)
class Openness:
    """Share of the ground band with no wall, versus the floors above."""

    ground_openness: float
    upper_openness: float
    ratio_vs_upper: float | None


def ground_openness(
    ground_open_px: int, ground_band_px: int, upper_open_px: int, upper_band_px: int
) -> Openness:
    """Openness fractions from wall-mask pixel counts. Zero bands -> 0.0."""
    ground = ground_open_px / ground_band_px if ground_band_px > 0 else 0.0
    upper = upper_open_px / upper_band_px if upper_band_px > 0 else 0.0
    return Openness(
        ground_openness=ground,
        upper_openness=upper,
        ratio_vs_upper=(ground / upper) if upper > 0 else None,
    )


@dataclass(frozen=True)
class Overhang:
    ratio: float
    flag: bool


def overhang_ratio(
    upper_width_px: float, ground_width_px: float, flag_at: float = 1.1
) -> Overhang:
    """Widest upper width / ground width. Above 1.1 is a flag (docs.md)."""
    ratio = upper_width_px / ground_width_px if ground_width_px > 0 else 0.0
    return Overhang(ratio=ratio, flag=ratio >= flag_at)


@dataclass(frozen=True)
class Pounding:
    gap_frac: float
    slabs_aligned: bool
    flag: bool


def pounding_risk(
    gap_px: float,
    frame_width_px: float,
    slabs_aligned: bool,
    gap_frac_threshold: float = 0.01,
) -> Pounding:
    """Flag when masks nearly touch AND slab rows do not line up (docs.md)."""
    gap_frac = gap_px / frame_width_px if frame_width_px > 0 else 1.0
    return Pounding(
        gap_frac=gap_frac,
        slabs_aligned=slabs_aligned,
        flag=(gap_frac < gap_frac_threshold) and not slabs_aligned,
    )
