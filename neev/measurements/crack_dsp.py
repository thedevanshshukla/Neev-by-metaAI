"""Digital Signal Processing (DSP) Structural Defect & Ridge Analysis Engine.

Implements high-frequency structural feature extraction:
- Multi-Scale Hessian Ridge Detector (Frangi Vesselness): Invariant to surface illumination gradients
- Centerline Skeleton Graph: Traces 1-pixel medial trajectory, extracts crack endpoints (tips) and junctions
- Tortuosity & Path Roughness: Distinguishes planar shear fractures (T ~ 1.0) from stepped masonry or shrinkage map cracking (T > 1.3)
- Aperture Profile: Min, mean, p95 (peak serviceability opening), and max aperture via Euclidean Distance Transform
- Candidate Salience Proposer: Autonomous detection of high-contrast linear discontinuities

Pure, deterministic math implemented using NumPy and SciPy.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class SkeletonMetrics:
    length_px: float
    chord_length_px: float
    tortuosity: float
    endpoints: list[tuple[int, int]]
    junctions: list[tuple[int, int]]
    pixel_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "length_px": self.length_px,
            "chord_length_px": self.chord_length_px,
            "tortuosity": self.tortuosity,
            "endpoints": [list(pt) for pt in self.endpoints],
            "junctions": [list(pt) for pt in self.junctions],
            "pixel_count": self.pixel_count,
        }


@dataclass(frozen=True)
class ApertureDistribution:
    min_width_px: float
    mean_width_px: float
    p95_width_px: float
    max_width_px: float
    estimated_mean_mm: float | None = None
    estimated_p95_mm: float | None = None
    estimated_max_mm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_frangi_vesselness(
    image_gray: np.ndarray,
    scales: tuple[float, ...] = (1.2, 2.5, 4.0),
    beta: float = 0.5,
    c_ratio: float = 0.08,
) -> np.ndarray:
    """Compute multi-scale Hessian ridge detector (Frangi vesselness).

    Highlights curvilinear dark features (cracks) against lighter textured walls.
    Returns normalized salience map in [0.0, 1.0].
    """
    if image_gray.ndim != 2:
        raise ValueError("image_gray must be a 2D array")

    arr = image_gray.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    vesselness_max = np.zeros_like(arr)

    for sigma in scales:
        # Second-order Gaussian derivative convolutions
        Ixx = ndimage.gaussian_filter(arr, sigma, order=(0, 2))
        Iyy = ndimage.gaussian_filter(arr, sigma, order=(2, 0))
        Ixy = ndimage.gaussian_filter(arr, sigma, order=(1, 1))

        # Analytical 2D eigensystem solution
        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy**2
        disc = np.maximum(0.0, trace**2 - 4.0 * det)
        sqrt_disc = np.sqrt(disc)

        l1 = 0.5 * (trace - sqrt_disc)
        l2 = 0.5 * (trace + sqrt_disc)

        # Sort eigenvalues: |e1| <= |e2|
        swap = np.abs(l1) > np.abs(l2)
        e1 = np.where(swap, l2, l1)
        e2 = np.where(swap, l1, l2)

        # Dark ridges on bright background have e2 > 0
        Rb = np.abs(e1) / (np.abs(e2) + 1e-6)
        S = np.sqrt(e1**2 + e2**2)
        c = c_ratio * float(S.max()) if S.max() > 0 else 1.0

        vessel = np.exp(-(Rb**2) / (2.0 * beta**2)) * (1.0 - np.exp(-(S**2) / (2.0 * c**2)))
        vessel = np.where(e2 > 0.0, vessel, 0.0)

        vesselness_max = np.maximum(vesselness_max, vessel)

    v_max = float(vesselness_max.max())
    if v_max > 0:
        vesselness_max = vesselness_max / v_max

    return vesselness_max


def extract_skeleton_topology(
    binary_mask: np.ndarray,
) -> SkeletonMetrics:
    """Extract 1-pixel medial skeleton, endpoints, junctions, and tortuosity."""
    if binary_mask.ndim != 2 or binary_mask.sum() == 0:
        return SkeletonMetrics(
            length_px=0.0,
            chord_length_px=0.0,
            tortuosity=1.0,
            endpoints=[],
            junctions=[],
            pixel_count=0,
        )

    # 1. Morphological thinning / distance transform skeleton
    dist = ndimage.distance_transform_edt(binary_mask)
    kernel_8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)

    # Local ridge maxima of distance transform
    skel = (dist > 0.5) & (dist >= ndimage.maximum_filter(dist, footprint=kernel_8) - 0.25)
    pixel_count = int(skel.sum())
    if pixel_count == 0:
        skel = binary_mask > 0
        pixel_count = int(skel.sum())

    # 2. Convolve with 8-neighbor kernel to classify topology
    neighbor_count = ndimage.convolve(skel.astype(int), kernel_8, mode="constant") * skel

    # Endpoints: 1 neighbor (or isolated 0)
    endpoints_arr = np.argwhere((neighbor_count == 1) | (neighbor_count == 0))
    # Junctions: 3 or more neighbors
    junctions_arr = np.argwhere(neighbor_count >= 3)

    endpoints = [(int(pt[1]), int(pt[0])) for pt in endpoints_arr]  # (x, y)
    junctions = [(int(pt[1]), int(pt[0])) for pt in junctions_arr]  # (x, y)

    # 3. Path arc length (approximate diagonal steps as sqrt(2))
    arc_length = float(pixel_count)

    # 4. Chord distance between dominant extremities
    if len(endpoints) >= 2:
        pts = np.array(endpoints[:30], dtype=float)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_matrix = np.hypot(diff[:, :, 0], diff[:, :, 1])
        chord_len = float(dist_matrix.max()) if dist_matrix.size > 0 else 1.0
        chord_len = max(1.0, chord_len)
    elif len(endpoints) == 1:
        # Distance from endpoint to furthest skeleton pixel
        ep = endpoints[0]
        skel_pts = np.argwhere(skel)
        dists = np.hypot(skel_pts[:, 1] - ep[0], skel_pts[:, 0] - ep[1])
        chord_len = float(dists.max()) if len(dists) > 0 else 1.0
    else:
        skel_pts = np.argwhere(skel)
        if len(skel_pts) >= 2:
            chord_len = float(math.hypot(
                skel_pts[:, 1].max() - skel_pts[:, 1].min(),
                skel_pts[:, 0].max() - skel_pts[:, 0].min(),
            ))
        else:
            chord_len = 1.0

    tortuosity = round(arc_length / chord_len, 2) if chord_len > 0 else 1.0
    tortuosity = max(1.0, tortuosity)

    return SkeletonMetrics(
        length_px=round(arc_length, 1),
        chord_length_px=round(chord_len, 1),
        tortuosity=tortuosity,
        endpoints=endpoints[:6],  # limit to top endpoints
        junctions=junctions[:10],
        pixel_count=pixel_count,
    )


def extract_aperture_profile(
    binary_mask: np.ndarray,
    gsd_mm_per_px: float | None = None,
) -> ApertureDistribution:
    """Compute sub-pixel Euclidean distance transform aperture profile."""
    if binary_mask.ndim != 2 or binary_mask.sum() == 0:
        return ApertureDistribution(0.0, 0.0, 0.0, 0.0)

    dist = ndimage.distance_transform_edt(binary_mask)
    radii = dist[dist > 0.5]
    if len(radii) == 0:
        return ApertureDistribution(1.0, 1.0, 1.0, 1.0)

    # Full aperture is diameter (2 * distance to boundary)
    apertures = radii * 2.0

    min_w = float(np.percentile(apertures, 5))
    mean_w = float(apertures.mean())
    p95_w = float(np.percentile(apertures, 95))
    max_w = float(apertures.max())

    return ApertureDistribution(
        min_width_px=round(min_w, 1),
        mean_width_px=round(mean_w, 1),
        p95_width_px=round(p95_w, 1),
        max_width_px=round(max_w, 1),
        estimated_mean_mm=round(mean_w * gsd_mm_per_px, 2) if gsd_mm_per_px else None,
        estimated_p95_mm=round(p95_w * gsd_mm_per_px, 2) if gsd_mm_per_px else None,
        estimated_max_mm=round(max_w * gsd_mm_per_px, 2) if gsd_mm_per_px else None,
    )


@dataclass(frozen=True)
class CrackCandidateProposal:
    proposal_id: int
    box: dict[str, int]
    polygon: list[tuple[float, float]]
    salience_score: float
    energy: float
    pixel_count: int
    tortuosity: float
    aspect_ratio: float
    dominant_angle_deg: float
    is_rectilinear: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "box": dict(self.box),
            "polygon": [list(pt) for pt in self.polygon],
            "salience_score": round(self.salience_score, 4),
            "energy": round(self.energy, 2),
            "pixel_count": self.pixel_count,
            "tortuosity": round(self.tortuosity, 2),
            "aspect_ratio": round(self.aspect_ratio, 2),
            "dominant_angle_deg": round(self.dominant_angle_deg, 1),
            "is_rectilinear": self.is_rectilinear,
        }


def propose_crack_candidate_boxes(
    image_gray: np.ndarray,
    scales: tuple[float, ...] = (1.2, 2.5, 4.0),
    threshold: float = 0.15,
    max_proposals: int = 15,
    min_pixels: int = 25,
    min_energy: float = 0.0,
    pad_px: int = 8,
) -> list[CrackCandidateProposal]:
    """Generate offline crack candidate bounding proposals using Frangi Hessian ridge salience.

    Pure, deterministic classical image processing function for Phase 2 candidate generation.
    """
    if image_gray.ndim != 2:
        raise ValueError("image_gray must be a 2D array")

    h, w = image_gray.shape
    if h == 0 or w == 0:
        return []

    v = compute_frangi_vesselness(image_gray, scales=scales, beta=0.5, c_ratio=0.08)
    binary = v > threshold

    lbl, num = ndimage.label(binary)
    if num == 0:
        return []

    objects = ndimage.find_objects(lbl)
    candidates: list[CrackCandidateProposal] = []

    for idx, sl in enumerate(objects):
        comp = lbl[sl] == (idx + 1)
        px = int(comp.sum())
        if px < min_pixels:
            continue

        energy = float(v[sl][comp].sum())
        if min_energy > 0.0 and energy < min_energy:
            continue
        bw = sl[1].stop - sl[1].start
        bh = sl[0].stop - sl[0].start
        aspect = max(bw, bh) / (min(bw, bh) + 1.0)

        # Skeleton topology & tortuosity
        skel = extract_skeleton_topology(comp)
        tortuosity = skel.tortuosity
        length = max(1.0, skel.length_px)

        # Orientation estimation
        ys, xs = np.where(comp)
        if len(xs) >= 5:
            cov = np.cov(xs, ys)
            angle_rad = 0.5 * math.atan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1])
            angle_deg = math.degrees(angle_rad) % 180
        else:
            angle_deg = 0.0

        is_horiz = (angle_deg <= 5.0) or (angle_deg >= 175.0)
        is_vert = abs(angle_deg - 90.0) <= 5.0
        is_rectilinear = bool((is_horiz or is_vert) and (tortuosity < 1.08))

        # Soft ranking score (rectilinear receives soft down-weighting without unconditional deletion)
        rect_mult = 0.70 if is_rectilinear else 1.0
        score = energy * (length**0.5) * rect_mult

        y0, y1 = max(0, sl[0].start - pad_px), min(h, sl[0].stop + pad_px)
        x0, x1 = max(0, sl[1].start - pad_px), min(w, sl[1].stop + pad_px)
        box = {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)}
        poly = [
            (float(x0), float(y0)),
            (float(x1), float(y0)),
            (float(x1), float(y1)),
            (float(x0), float(y1)),
        ]

        candidates.append(
            CrackCandidateProposal(
                proposal_id=idx,
                box=box,
                polygon=poly,
                salience_score=score,
                energy=energy,
                pixel_count=px,
                tortuosity=tortuosity,
                aspect_ratio=aspect,
                dominant_angle_deg=angle_deg,
                is_rectilinear=is_rectilinear,
            )
        )

    candidates.sort(key=lambda c: -c.salience_score)
    return candidates[:max_proposals]
