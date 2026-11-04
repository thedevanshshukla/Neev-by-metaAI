"""Lightweight deterministic image-quality checks (numpy + Pillow only).

Heuristics with conservative, configurable thresholds. A warning means
"this factor may hide defects - look closer", never a calibrated
inspection-suitability verdict. Every warning names the check that fired.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QualityThresholds:
    min_side_px: int = 480
    blur_laplacian_var: float = 60.0
    dark_mean_luma: float = 40.0
    bright_sat_frac: float = 0.15


@dataclass
class QualityReport:
    width: int
    height: int
    warnings: list[dict[str, str]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def _laplacian_variance(grey: Any) -> float:
    import numpy as np

    arr = np.asarray(grey, dtype=np.float64)
    lap = (arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2]
           + arr[1:-1, 2:] - 4.0 * arr[1:-1, 1:-1])
    return float(lap.var())


def assess_image(image: Any,
                 thresholds: QualityThresholds | None = None) -> QualityReport:
    """Assess a Pillow image (or numpy HxWx3 array). Never raises on pixels."""
    import numpy as np
    from PIL import Image

    thr = thresholds or QualityThresholds()
    if isinstance(image, np.ndarray):
        pil = Image.fromarray(image)
    else:
        pil = image
    width, height = pil.size
    report = QualityReport(width=width, height=height)
    grey = np.asarray(pil.convert("L"), dtype=np.float64)
    report.metrics["min_side_px"] = min(width, height)
    if min(width, height) < thr.min_side_px:
        report.warnings.append({
            "check": "low_resolution",
            "detail": (f"smallest side is {min(width, height)}px "
                       f"(threshold {thr.min_side_px}px); thin defects "
                       f"may be unresolvable"),
        })
    if width >= 3 and height >= 3:
        blur = _laplacian_variance(grey)
        report.metrics["laplacian_var"] = round(blur, 2)
        if blur < thr.blur_laplacian_var:
            report.warnings.append({
                "check": "possible_blur",
                "detail": (f"Laplacian variance {blur:.1f} "
                           f"(threshold {thr.blur_laplacian_var}); image may "
                           f"be blurred - re-photograph if defects matter"),
            })
    else:
        report.metrics["laplacian_var"] = None
    mean_luma = float(grey.mean())
    report.metrics["mean_luma"] = round(mean_luma, 2)
    if mean_luma < thr.dark_mean_luma:
        report.warnings.append({
            "check": "too_dark",
            "detail": (f"mean luminance {mean_luma:.1f} "
                       f"(threshold {thr.dark_mean_luma}); defects may hide "
                       f"in shadows"),
        })
    sat_frac = float((grey >= 250).mean())
    report.metrics["saturated_frac"] = round(sat_frac, 4)
    if sat_frac > thr.bright_sat_frac:
        report.warnings.append({
            "check": "overexposed",
            "detail": (f"{sat_frac * 100:.1f}% pixels near-white "
                       f"(threshold {thr.bright_sat_frac * 100:.0f}%); "
                       f"washed-out areas hide surface defects"),
        })
    return report
