"""Defect Physics & Structural Mechanics Analyzer (DSP & Morphological Engine).

Extracts structural engineering properties from surface defects:
- Crack Centerline & Arc Length: 1-pixel medial skeleton and path length
- Aperture Profile: Minimum, mean, p95 (peak serviceability opening), and maximum crack width
- Morphological Topology: Crack tip endpoints, branching junctions, and tortuosity
- Structural Mechanics Classification:
    * DIAGONAL_SHEAR (30° - 60°): Seismic in-plane shear failure in columns / masonry panels
    * HORIZONTAL_FLEXURAL (0° - 30°): Out-of-plane bending tension at floor lines
    * VERTICAL_SETTLEMENT (60° - 90°): Differential foundation settlement
    * MAP_SHRINKAGE_ASR: High tortuosity / multi-branching non-structural surface cracking
- Severity Rating (IS 456 / ACI 224 standard): Cosmetic (<0.3mm), Minor (0.3-1.0mm), Severe (>1.0mm)
- Actionable Engineering Mitigations (IS 13935 / FEMA P-154 / ACI 224)

Pure, deterministic math. Works with NumPy/SciPy when available, with pure-Python fallbacks.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

# Orientation classes for structural seismic screening
ORIENTATION_DIAGONAL_SHEAR = "diagonal_shear"      # ~45 deg: critical earthquake shear
ORIENTATION_HORIZONTAL_FLEXURE = "horizontal_flexure"  # ~0 deg: slab/beam bending tension
ORIENTATION_VERTICAL_SETTLEMENT = "vertical_settlement"  # ~90 deg: foundation differential movement
ORIENTATION_MAP_CRACKING = "map_shrinkage_asr"       # High tortuosity / multi-branch

# Severity classes based on crack aperture (width)
SEVERITY_COSMETIC = "cosmetic"    # < 0.3 mm (or < 3 px default)
SEVERITY_MODERATE = "moderate"    # 0.3 - 1.0 mm (or 3 - 8 px)
SEVERITY_STRUCTURAL = "structural_severe"  # > 1.0 mm (or > 8 px)


@dataclass(frozen=True)
class CrackPhysicsReport:
    length_px: float
    max_width_px: float
    mean_width_px: float
    orientation_deg: float
    orientation_type: str
    severity: str
    aspect_ratio: float
    pixel_area: int
    skeleton_pixel_count: int
    estimated_width_mm: float | None = None
    estimated_length_mm: float | None = None
    seismic_risk_factor: float = 1.0  # multiplier: shear cracks get 1.5x risk penalty
    p95_width_px: float = 0.0
    tortuosity: float = 1.0
    branch_points: int = 0
    crack_tips: list[list[int]] = field(default_factory=list)
    failure_mechanism: str = "Unspecified fracture"
    engineering_action: str = "Standard visual monitoring"
    code_reference: str = "IS 456 / ACI 224R-01"

    screening_disclaimer: str = "Visual screening only; ordinary photographs cannot determine crack depth, structural integrity, or building safety."
    is_calibrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classify_orientation(angle_deg: float) -> str:
    """Classify crack angle relative to horizontal [0..90]."""
    abs_angle = abs(angle_deg) % 180.0
    if abs_angle > 90.0:
        abs_angle = 180.0 - abs_angle

    if 30.0 <= abs_angle <= 60.0:
        return ORIENTATION_DIAGONAL_SHEAR
    elif abs_angle < 30.0:
        return ORIENTATION_HORIZONTAL_FLEXURE
    else:
        return ORIENTATION_VERTICAL_SETTLEMENT


def _classify_severity(width_px: float, width_mm: float | None) -> str:
    """Aperture-based visual prominence classification (IS 456 / ACI 224 reference)."""
    if width_mm is not None:
        if width_mm >= 1.0:
            return SEVERITY_STRUCTURAL
        elif width_mm >= 0.3:
            return SEVERITY_MODERATE
        return SEVERITY_COSMETIC

    # Pixel heuristic if GSD not calibrated:
    if width_px >= 8.0:
        return SEVERITY_STRUCTURAL
    elif width_px >= 3.0:
        return SEVERITY_MODERATE
    return SEVERITY_COSMETIC


def _structural_diagnostics(
    orient_type: str,
    tortuosity: float,
    branch_count: int,
) -> tuple[str, str, str, float]:
    """Derive observable pattern description, visual screening guidance, and reference."""
    if orient_type == ORIENTATION_MAP_CRACKING:
        return (
            "Observed Shrinkage Craze / Surface Crazing Pattern",
            "Visual screening observation: surface craze cracking. Recommend on-site visual monitoring for moisture penetration.",
            "Visual screening benchmark (IS 456 Annex F / ACI 221.1R)",
            1.0,
        )

    if orient_type == ORIENTATION_DIAGONAL_SHEAR:
        return (
            "Observed Diagonal Linear Fracture (~45° in-plane angle, Seismic In-Plane Shear Pattern)",
            "Visual screening flag: diagonal crack pattern observed. Recommend physical examination by a qualified structural engineer to evaluate facade condition.",
            "Visual screening benchmark (IS 13935 / FEMA P-154 screening guide)",
            1.5,
        )
    elif orient_type == ORIENTATION_HORIZONTAL_FLEXURE:
        return (
            "Observed Horizontal Linear Crack (~0° flexure pattern)",
            "Visual screening flag: horizontal joint/boundary crack observed. Recommend visual inspection of wall-slab boundary by a qualified engineer.",
            "Visual screening benchmark (IS 456 / ACI 224R)",
            1.0,
        )
    else:  # VERTICAL_SETTLEMENT
        return (
            "Observed Vertical Surface Fissure (~90° orientation, Differential Movement Pattern)",
            "Visual screening flag: vertical fissure observed. Recommend baseline displacement monitoring and physical foundation inspection by a qualified engineer.",
            "Visual screening benchmark (IS 1904 / ACI 224.1R)",
            1.2,
        )


def _box_physics(box: dict[str, Any], gsd_mm_per_px: float | None = None) -> CrackPhysicsReport:
    """Fallback physics estimates from bounding box geometry alone."""
    w = float(box["x1"] - box["x0"])
    h = float(box["y1"] - box["y0"])
    diag = math.hypot(w, h)
    short_s = min(w, h)
    long_s = max(w, h)
    aspect = long_s / short_s if short_s > 0 else 1.0

    angle = math.degrees(math.atan2(h, w))
    orient_type = _classify_orientation(angle)

    mechanism, action, code_ref, risk_factor = _structural_diagnostics(
        orient_type, 1.0, 0
    )

    w_mm = short_s * gsd_mm_per_px if gsd_mm_per_px else None
    l_mm = diag * gsd_mm_per_px if gsd_mm_per_px else None
    severity = _classify_severity(short_s, w_mm)

    x0, y0, x1, y1 = int(box["x0"]), int(box["y0"]), int(box["x1"]), int(box["y1"])
    crack_tips = [[x0, y0], [x1, y1]]

    return CrackPhysicsReport(
        length_px=round(diag, 1),
        max_width_px=round(short_s, 1),
        mean_width_px=round(short_s * 0.8, 1),
        orientation_deg=round(angle, 1),
        orientation_type=orient_type,
        severity=severity,
        aspect_ratio=round(aspect, 2),
        pixel_area=int(w * h),
        skeleton_pixel_count=int(diag),
        estimated_width_mm=round(w_mm, 2) if w_mm else None,
        estimated_length_mm=round(l_mm, 1) if l_mm else None,
        seismic_risk_factor=risk_factor,
        p95_width_px=round(short_s * 0.95, 1),
        tortuosity=1.0,
        branch_points=0,
        crack_tips=crack_tips,
        failure_mechanism=mechanism,
        engineering_action=action,
        code_reference=code_ref,
    )


def analyze_crack_physics(
    box: dict[str, Any],
    raster: bytes | None = None,
    raster_width: int = 0,
    raster_height: int = 0,
    gsd_mm_per_px: float | None = None,
) -> CrackPhysicsReport:
    """Compute structural crack physics, DSP topology, and seismic failure mechanics.

    If binary raster is available, extracts medial skeleton, distance transform aperture,
    endpoints/junctions, and second-moment eigen-orientation.
    Otherwise falls back cleanly to bounding box geometry.
    """
    if raster is None or raster_width <= 0 or raster_height <= 0 or len(raster) != raster_width * raster_height:
        return _box_physics(box, gsd_mm_per_px)

    try:
        import numpy as np
        from neev.measurements.crack_dsp import extract_aperture_profile, extract_skeleton_topology
    except ImportError:
        return _box_physics(box, gsd_mm_per_px)

    arr = np.frombuffer(raster, dtype=np.uint8).reshape((raster_height, raster_width))
    pixel_area = int(arr.sum())
    if pixel_area == 0:
        return _box_physics(box, gsd_mm_per_px)

    # 1. DSP Topology & Skeleton Analysis
    skel_metrics = extract_skeleton_topology(arr > 0)
    aperture_dist = extract_aperture_profile(arr > 0, gsd_mm_per_px)

    # 2. Spatial second-moments for principal orientation
    ys, xs = np.nonzero(arr)
    if len(xs) >= 3:
        cov = np.cov(xs, ys)
        evals, evecs = np.linalg.eigh(cov)
        principal_vec = evecs[:, -1]
        angle_rad = math.atan2(principal_vec[1], principal_vec[0])
        angle_deg = math.degrees(angle_rad)
        eigen_ratio = float(evals[0] / evals[1]) if evals[1] > 0 else 0.0
    else:
        angle_deg = 0.0
        eigen_ratio = 0.0

    orient_type = _classify_orientation(angle_deg)
    # Map/craze cracking requires isotropic multi-directionality (high eigen_ratio) and multiple branch junctions
    if eigen_ratio >= 0.40 and len(skel_metrics.junctions) >= 3 and skel_metrics.tortuosity >= 1.35:
        orient_type = ORIENTATION_MAP_CRACKING

    mechanism, action, code_ref, risk_factor = _structural_diagnostics(
        orient_type, skel_metrics.tortuosity, len(skel_metrics.junctions)
    )

    max_w = aperture_dist.max_width_px
    mean_w = aperture_dist.mean_width_px
    p95_w = aperture_dist.p95_width_px
    length_px = skel_metrics.length_px if skel_metrics.length_px > 0 else math.hypot(raster_width, raster_height)

    w_mm = aperture_dist.estimated_max_mm
    l_mm = length_px * gsd_mm_per_px if gsd_mm_per_px else None
    severity = _classify_severity(max_w, w_mm)

    aspect = length_px / max_w if max_w > 0 else 1.0

    # Translate local tips to full-frame coordinates if box provided
    tips_in_box = []
    x_offset = int(box.get("x0", 0))
    y_offset = int(box.get("y0", 0))
    for pt in skel_metrics.endpoints[:2]:
        tips_in_box.append([pt[0] + x_offset, pt[1] + y_offset])
    if not tips_in_box:
        tips_in_box = [[x_offset, y_offset], [int(box.get("x1", raster_width)), int(box.get("y1", raster_height))]]

    return CrackPhysicsReport(
        length_px=round(length_px, 1),
        max_width_px=round(max_w, 1),
        mean_width_px=round(mean_w, 1),
        orientation_deg=round(angle_deg, 1),
        orientation_type=orient_type,
        severity=severity,
        aspect_ratio=round(aspect, 2),
        pixel_area=pixel_area,
        skeleton_pixel_count=skel_metrics.pixel_count,
        estimated_width_mm=w_mm,
        estimated_length_mm=round(l_mm, 1) if l_mm else None,
        seismic_risk_factor=risk_factor,
        p95_width_px=p95_w,
        tortuosity=skel_metrics.tortuosity,
        branch_points=len(skel_metrics.junctions),
        crack_tips=tips_in_box,
        failure_mechanism=mechanism,
        engineering_action=action,
        code_reference=code_ref,
    )
