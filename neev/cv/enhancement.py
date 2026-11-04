"""Facade Image Enhancement Module for Defect Screening.

Implements non-destructive, configurable computer-vision enhancements using OpenCV:
1. LAB-space Contrast Limited Adaptive Histogram Equalization (CLAHE).
2. Multi-Scale Retinex (MSR) illumination normalization for deep shadows.
3. Edge-preserving bilateral denoising to eliminate grain while keeping crack edges sharp.
4. Shadow-aware background illumination field normalization.
5. Complementary multi-view generation preserving original coordinates.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def apply_clahe_lab(
    img_bgr: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Applies CLAHE on the Lightness (L) channel in LAB color space.

    Preserves color channels (A and B) while enhancing localized crack contrast.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def apply_multiscale_retinex(
    img_bgr: np.ndarray,
    sigmas: tuple[int, ...] = (15, 45, 90),
    dynamic_range_clip: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Multi-Scale Retinex (MSR) illumination normalization.

    Decomposes the image into reflectance and illumination components:
    R_i = sum_k w_k * [log(I_i + 1) - log(I_i * G_k + 1)]
    Normalizes steep shadows under overhangs, balconies, and soffits.
    Scale-adaptive to avoid memory exhaustion on high-resolution facade images.
    """
    h, w = img_bgr.shape[:2]
    scale = 1.0
    if max(h, w) > 1024:
        scale = 1024.0 / max(h, w)
        small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))
        proc_bgr = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    else:
        proc_bgr = img_bgr

    img_float = proc_bgr.astype(np.float32) + 1.0
    weight = 1.0 / len(sigmas)
    retinex = np.zeros_like(img_float)

    for sigma in sigmas:
        try:
            blur = cv2.GaussianBlur(img_float, (0, 0), float(sigma))
        except (cv2.error, Exception):
            k = int(2 * round(float(sigma) * 2) + 1)
            blur = cv2.blur(img_float, (k, k))
        retinex += weight * (np.log(img_float) - np.log(blur + 1e-4))
        del blur

    # Normalize each channel back to 0-255 using percentile clipping
    out_small = np.zeros_like(proc_bgr)
    for c in range(3):
        ch = retinex[:, :, c]
        p_low, p_high = np.percentile(ch, dynamic_range_clip)
        if p_high > p_low:
            ch_norm = np.clip((ch - p_low) / (p_high - p_low) * 255.0, 0, 255)
        else:
            ch_norm = np.full_like(ch, 128.0)
        out_small[:, :, c] = ch_norm.astype(np.uint8)

    if scale != 1.0:
        return cv2.resize(out_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return out_small


def apply_edge_preserving_denoise(
    img_bgr: np.ndarray,
    d: int = 5,
    sigma_color: float = 40.0,
    sigma_space: float = 40.0,
) -> np.ndarray:
    """Applies bilateral filtering to suppress plaster texture grain while keeping cracks sharp."""
    h, w = img_bgr.shape[:2]
    if max(h, w) > 1200:
        scale = 1200.0 / max(h, w)
        sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
        small = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        denoised = cv2.bilateralFilter(small, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        return cv2.resize(denoised, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.bilateralFilter(img_bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def apply_shadow_illumination_equalization(
    img_bgr: np.ndarray,
    blur_kernel_size: int = 61,
) -> np.ndarray:
    """Estimates low-frequency background illumination and normalizes illumination gradients."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg_illum = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    bg_float = np.maximum(bg_illum.astype(np.float32), 1.0)

    # Divide lightness channel by background field
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32)
    l_equalized = (l_float / bg_float) * 128.0
    l_equalized = np.clip(l_equalized, 0, 255).astype(np.uint8)

    lab_out = cv2.merge((l_equalized, a, b))
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)


def compute_shadow_mask(
    img_bgr: np.ndarray,
    threshold_ratio: float = 0.35,
    blur_kernel_size: int = 61,
) -> np.ndarray:
    """Computes a binary mask where True represents deep shadow or underexposed zones.

    Used by Shadow-Aware Dual-Path Inference to accept enhanced-view detections
    strictly within shadow regions without creating false positives in sunlit areas.
    """
    h, w = img_bgr.shape[:2]
    max_dim = max(h, w)
    if max_dim > 1024:
        scale = 1024.0 / max_dim
        sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        proc_bgr = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        proc_bgr = img_bgr

    gray = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2GRAY)
    k = min(blur_kernel_size, min(gray.shape[:2]) // 2 * 2 + 1)
    if k % 2 == 0:
        k += 1
    bg_illum = cv2.GaussianBlur(gray, (k, k), 0)
    max_val = float(np.max(bg_illum)) if bg_illum.size > 0 else 1.0
    cutoff = max_val * threshold_ratio
    mask_small = (bg_illum <= cutoff).astype(np.uint8)

    if scale != 1.0:
        mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask_full.astype(bool)
    return mask_small.astype(bool)


def is_candidate_in_shadow(
    box: dict[str, int],
    shadow_mask: np.ndarray,
    min_shadow_overlap: float = 0.35,
) -> bool:
    """Determines whether a candidate box falls inside a deep shadow zone."""
    h, w = shadow_mask.shape[:2]
    x0 = max(0, min(w - 1, int(box["x0"])))
    y0 = max(0, min(h - 1, int(box["y0"])))
    x1 = max(x0 + 1, min(w, int(box["x1"])))
    y1 = max(y0 + 1, min(h, int(box["y1"])))

    crop_mask = shadow_mask[y0:y1, x0:x1]
    if crop_mask.size == 0:
        return False
    return (float(np.mean(crop_mask)) >= min_shadow_overlap)


def generate_enhanced_views(
    image: Image.Image | np.ndarray,
) -> dict[str, Image.Image]:
    """Generates complementary enhanced views alongside the original image."""
    if isinstance(image, Image.Image):
        w, h = image.size
        max_dim = max(w, h)
        if max_dim > 1536:
            scale = 1536.0 / max_dim
            sw, sh = int(round(w * scale)), int(round(h * scale))
            proc_im = image.resize((sw, sh), Image.Resampling.BILINEAR)
            im_rgb = np.array(proc_im.convert("RGB"))
        else:
            im_rgb = np.array(image.convert("RGB"))
    else:
        h, w = image.shape[:2]
        max_dim = max(w, h)
        if max_dim > 1536:
            scale = 1536.0 / max_dim
            sw, sh = int(round(w * scale)), int(round(h * scale))
            im_rgb = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            im_rgb = image

    im_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)

    try:
        clahe_bgr = apply_clahe_lab(im_bgr, clip_limit=2.5)
        retinex_bgr = apply_multiscale_retinex(im_bgr)
        denoised_bgr = apply_edge_preserving_denoise(im_bgr)
        shadow_eq_bgr = apply_shadow_illumination_equalization(im_bgr)
    except Exception:
        clahe_bgr = im_bgr
        retinex_bgr = im_bgr
        denoised_bgr = im_bgr
        shadow_eq_bgr = im_bgr

    return {
        "original": Image.fromarray(im_rgb),
        "clahe": Image.fromarray(cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB)),
        "retinex": Image.fromarray(cv2.cvtColor(retinex_bgr, cv2.COLOR_BGR2RGB)),
        "denoised": Image.fromarray(cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)),
        "shadow_equalized": Image.fromarray(cv2.cvtColor(shadow_eq_bgr, cv2.COLOR_BGR2RGB)),
    }
