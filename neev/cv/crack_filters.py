"""Crack-Specific Signal Processing and Multi-Scale Ridge Filtering.

Implements physics and differential geometry based ridge detectors:
1. Multi-scale Frangi vesselness filter (eigenvalues of Hessian matrix).
2. Sato continuous ridge/tubularity filter.
3. Multi-scale Black Top-Hat (Bottom-Hat) morphological transformation for dark fissures.
4. Directional Gabor filter bank for multi-angle crack orientation discovery.
5. Signal-to-candidate proposal extractor for faint defect recovery.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import frangi, sato


def compute_frangi_vesselness(
    img_gray: np.ndarray,
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
    black_ridges: bool = True,
) -> np.ndarray:
    """Computes multi-scale Frangi vesselness filter using Hessian eigenvalues.

    Cracks manifest as dark (or light) linear ridges with high principal curvature:
    lambda_2 >> lambda_1 ~ 0.
    """
    img_norm = img_gray.astype(np.float32) / 255.0
    vesselness = frangi(
        img_norm,
        sigmas=sigmas,
        black_ridges=black_ridges,
        beta=0.5,
    )
    # Scale to 0-1 range
    max_val = np.max(vesselness)
    if max_val > 0:
        vesselness /= max_val
    return vesselness.astype(np.float32)


def compute_sato_tubularity(
    img_gray: np.ndarray,
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0),
    black_ridges: bool = True,
) -> np.ndarray:
    """Computes continuous tubularity/ridge response via Sato filter."""
    img_norm = img_gray.astype(np.float32) / 255.0
    tubularity = sato(
        img_norm,
        sigmas=sigmas,
        black_ridges=black_ridges,
    )
    max_val = np.max(tubularity)
    if max_val > 0:
        tubularity /= max_val
    return tubularity.astype(np.float32)


def compute_black_tophat(
    img_gray: np.ndarray,
    kernel_sizes: tuple[int, ...] = (5, 9, 15),
) -> np.ndarray:
    """Multi-scale Black Top-Hat (Bottom-Hat) morphology.

    BottomHat(I) = Closing(I) - I.
    Isolates dark linear fissures that are narrower than the structuring element.
    """
    accum = np.zeros_like(img_gray, dtype=np.float32)
    for k in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        tophat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
        accum += tophat.astype(np.float32) / len(kernel_sizes)

    max_val = np.max(accum)
    if max_val > 0:
        accum /= max_val
    return accum.astype(np.float32)


def compute_directional_gabor(
    img_gray: np.ndarray,
    num_angles: int = 8,
    wavelengths: tuple[float, ...] = (4.0, 8.0),
) -> np.ndarray:
    """Computes maximum energy across a directional Gabor filter bank."""
    accum = np.zeros_like(img_gray, dtype=np.float32)
    img_float = img_gray.astype(np.float32)

    for theta_idx in range(num_angles):
        theta = (theta_idx * np.pi) / num_angles
        for lambd in wavelengths:
            kernel = cv2.getGaborKernel(
                ksize=(15, 15),
                sigma=2.5,
                theta=theta,
                lambd=lambd,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F,
            )
            filtered = cv2.filter2D(img_float, cv2.CV_32F, kernel)
            accum = np.maximum(accum, np.abs(filtered))

    max_val = np.max(accum)
    if max_val > 0:
        accum /= max_val
    return accum.astype(np.float32)


def extract_signal_crack_candidates(
    img_rgb: np.ndarray,
    percentile_threshold: float = 96.0,
    min_area_px: int = 40,
    max_area_ratio: float = 0.25,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generates localized crack candidate regions from fused physical ridge signals.

    Combines Frangi, Sato, and Bottom-Hat transforms to recover faint hairline structures.
    Returns composite energy map and candidate bounding boxes with structural features.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    scale = 1.0
    if max(h, w) > 800:
        scale = 800.0 / max(h, w)
        sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
        proc_gray = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        proc_gray = gray
        sw, sh = w, h

    # 1. Compute multi-scale filter responses on scale-normalized view
    frangi_resp = compute_frangi_vesselness(proc_gray, sigmas=(1.0, 2.0, 3.0))
    sato_resp = compute_sato_tubularity(proc_gray, sigmas=(1.0, 2.0))
    tophat_resp = compute_black_tophat(proc_gray, kernel_sizes=(5, 11))

    # 2. Weighted multi-signal fusion
    fused_energy = 0.45 * frangi_resp + 0.35 * sato_resp + 0.20 * tophat_resp

    # Energy peak floor: reject completely homogeneous views without any ridge structure
    if np.max(fused_energy) < 0.15:
        return fused_energy, []

    # 3. Dynamic percentile thresholding
    t_val = np.percentile(fused_energy, percentile_threshold)
    binary = (fused_energy >= t_val).astype(np.uint8)

    # Morphological closing to bridge micro-discontinuities along fracture paths
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    # 4. Connected component candidate extraction
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_closed, connectivity=8)

    candidates = []
    max_total_area = sw * sh * max_area_ratio
    scaled_min_area = max(5, int(min_area_px * (scale ** 2)))

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < scaled_min_area or area > max_total_area:
            continue

        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Filter out purely isotropic compact blobs (cracks are elongated fissures)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        fill_factor = area / max(bw * bh, 1)

        if aspect < 2.0 or max(bw, bh) < 20:
            continue

        # Region mask for signal intensity measurement
        comp_mask = (labels[by:by + bh, bx:bx + bw] == i)
        comp_energy = fused_energy[by:by + bh, bx:bx + bw][comp_mask]
        mean_sig = float(np.mean(comp_energy)) if len(comp_energy) > 0 else 0.0

        # Scale coordinates back to original native resolution
        pad = 10
        x0 = max(0, int((bx - 5) / scale) - pad)
        y0 = max(0, int((by - 5) / scale) - pad)
        x1 = min(w, int((bx + bw + 5) / scale) + pad)
        y1 = min(h, int((by + bh + 5) / scale) + pad)

        orig_area = int(area / (scale ** 2))

        candidates.append({
            "box": {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)},
            "source": "dsp_ridge_filter",
            "signal_strength": round(mean_sig, 3),
            "area_px": orig_area,
            "aspect_ratio": round(float(aspect), 2),
            "fill_factor": round(float(fill_factor), 3),
            "defect_type": "crack",
        })

    return fused_energy, candidates
