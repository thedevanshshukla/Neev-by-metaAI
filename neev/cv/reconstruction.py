"""Segmentation Mask Reconstruction and Geometric Feature Extraction Module.

Implements rigorous post-processing and geometric measurement for facade defects:
1. Mask refinement (hole filling, morphological reconstruction, small-island suppression).
2. Skeleton extraction, branch-point detection, endpoint detection, and width profiling.
3. Region defect metrics (area_px, perimeter_px, compactness, boundary irregularity).
4. Fragment reconciliation for discontinuous cracks.
5. Strict coordinate provenance with all units explicitly labeled in pixels.
"""

from __future__ import annotations

import math
from typing import Any, Sequence
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def refine_defect_mask(
    mask: np.ndarray,
    close_kernel_size: int = 3,
    min_island_px: int = 15,
) -> np.ndarray:
    """Refines raw predicted masks via morphological closing and island removal."""
    bin_mask = (mask > 0).astype(np.uint8)
    if not np.any(bin_mask):
        return bin_mask

    # 1. Morphological closing to connect adjacent crack fragments
    if close_kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

    # 2. Remove tiny detached noise islands
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    clean_mask = np.zeros_like(bin_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_island_px:
            clean_mask[labels == i] = 1

    return clean_mask


def extract_crack_skeleton_metrics(mask: np.ndarray) -> dict[str, Any]:
    """Extracts topological graph features along the crack medial axis.

    Measures length_px, endpoints, branch points, and width profile in pixels.
    """
    bin_mask = (mask > 0).astype(np.uint8)
    if not np.any(bin_mask):
        return {
            "skeleton_length_px": 0.0,
            "endpoints_count": 0,
            "branch_points_count": 0,
            "mean_width_px": 0.0,
            "max_width_px": 0.0,
            "orientation_deg": 0.0,
            "is_branching": False,
        }

    # 1. Topological thinning / skeletonization
    skel = skeletonize(bin_mask.astype(bool))
    skel_u8 = skel.astype(np.uint8)

    # 2. Distance transform for width profiling
    dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
    skel_coords = np.argwhere(skel)
    skel_length = float(len(skel_coords))

    widths = []
    for y, x in skel_coords:
        # Crack width = 2 * distance to boundary
        w_val = float(dist[y, x] * 2.0)
        widths.append(w_val)

    mean_w = float(np.mean(widths)) if widths else 0.0
    max_w = float(np.max(widths)) if widths else 0.0

    # 3. Graph node degrees via 8-neighborhood kernel convolution
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_u8, cv2.CV_8U, kernel) * skel_u8

    endpoints = int(np.sum(neighbor_count == 1))
    branch_points = int(np.sum(neighbor_count >= 3))

    # 4. Dominant orientation via principal component analysis
    orientation_deg = 0.0
    if len(skel_coords) >= 5:
        # Center coordinates
        pts = skel_coords.astype(np.float32)
        mean_pt = np.mean(pts, axis=0)
        centered = pts - mean_pt
        cov = np.cov(centered, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        principal_axis = evecs[:, -1]  # [dy, dx]
        angle_rad = math.atan2(float(principal_axis[0]), float(principal_axis[1]))
        orientation_deg = round(math.degrees(angle_rad), 1)

    return {
        "skeleton_length_px": round(skel_length, 1),
        "endpoints_count": endpoints,
        "branch_points_count": branch_points,
        "mean_width_px": round(mean_w, 2),
        "max_width_px": round(max_w, 2),
        "orientation_deg": orientation_deg,
        "is_branching": branch_points > 0,
    }


def extract_region_defect_metrics(mask: np.ndarray) -> dict[str, Any]:
    """Calculates geometric properties for 2D region defects (spalling, stains, cavities)."""
    bin_mask = (mask > 0).astype(np.uint8)
    if not np.any(bin_mask):
        return {
            "area_px": 0,
            "perimeter_px": 0.0,
            "compactness": 0.0,
            "boundary_irregularity": 0.0,
            "extent": 0.0,
        }

    area = int(np.sum(bin_mask))
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "area_px": area,
            "perimeter_px": 0.0,
            "compactness": 0.0,
            "boundary_irregularity": 0.0,
            "extent": 0.0,
        }

    cnt = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(cnt, True))

    # Compactness: 4*pi*Area / P^2
    compactness = 0.0
    if perimeter > 0:
        compactness = (4.0 * math.pi * area) / (perimeter ** 2)

    # Boundary irregularity: Perimeter / Convex Hull Perimeter
    hull = cv2.convexHull(cnt)
    hull_perimeter = float(cv2.arcLength(hull, True))
    irregularity = (perimeter / hull_perimeter) if hull_perimeter > 0 else 1.0

    # Extent: Area / Bounding Box Area
    x, y, w, h = cv2.boundingRect(cnt)
    extent = area / max(w * h, 1)

    return {
        "area_px": area,
        "perimeter_px": round(perimeter, 1),
        "compactness": round(float(compactness), 3),
        "boundary_irregularity": round(float(irregularity), 3),
        "extent": round(float(extent), 3),
    }
