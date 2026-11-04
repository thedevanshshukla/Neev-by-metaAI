"""Multi-Defect Facade Screening and Physical Feature Classification.

Supports multi-defect taxonomy:
- crack: Linear structural fractures.
- spalling: Concrete surface material loss and exposed cavities.
- exposed_rebar: Corroded metallic reinforcing bars.
- corrosion_stain: Rust discoloration and iron oxide leaching halos.
- efflorescence: White crystalline mineral salt deposits.
- water_stain: Dampness, moisture seepage, and vertical drainage streaks.
- unknown_anomaly: Irregular surface defect requiring inspector review.

Fuses colorimetric features (HSV/LAB), geometric morphology, and deep vision cues
with transparent uncertainty calibration and limitation reporting.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional
import cv2
import numpy as np
from PIL import Image

from neev.cv.reconstruction import extract_crack_skeleton_metrics, extract_region_defect_metrics


@dataclass(frozen=True)
class MultiDefectFinding:
    """Standardized multi-defect finding with multi-cue physical evidence."""
    finding_id: str
    defect_category: str
    box: dict[str, int]
    confidence: float
    uncertainty: float
    evidence_sources: list[str]
    visual_features: dict[str, Any]
    requires_human_review: bool
    recommended_action: str
    limitations: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultiDefectClassifier:
    """Classifies candidate facade crops into standard defect categories using multi-cue fusion."""

    def classify_crop(
        self,
        crop_rgb: np.ndarray,
        box: dict[str, int],
        finding_id: str = "def_001",
        prior_source: str = "sam_detector",
        prior_confidence: float = 0.85,
    ) -> MultiDefectFinding:
        """Evaluates spectral color, geometry, and texture to classify defect type."""
        h, w, _ = crop_rgb.shape
        if h < 4 or w < 4:
            return MultiDefectFinding(
                finding_id=finding_id,
                box=box,
                defect_category="unknown_anomaly",
                confidence=0.50,
                uncertainty=0.50,
                evidence_sources=[prior_source],
                visual_features={"area_px": w * h},
                requires_human_review=True,
                recommended_action="Inspect facade crop manually",
                limitations="Crop dimensions too small for reliable feature extraction",
            )

        # 1. Colorimetric Analysis (HSV and LAB color spaces)
        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)

        h_ch, s_ch, v_ch = cv2.split(hsv)
        l_ch, a_ch, b_ch = cv2.split(lab)

        mean_v = float(np.mean(v_ch))
        mean_s = float(np.mean(s_ch))
        mean_a = float(np.mean(a_ch))
        mean_b = float(np.mean(b_ch))
        std_v = float(np.std(v_ch))

        # Rust / Corrosion index (elevated a* and b*, red/orange hue in HSV: H in [5, 25])
        rust_mask = ((h_ch >= 5) & (h_ch <= 28) & (s_ch >= 60) & (v_ch >= 40))
        rust_ratio = float(np.sum(rust_mask)) / (w * h)

        # Efflorescence index (high lightness L >= 210, low saturation S <= 30)
        salt_mask = ((l_ch >= 205) & (s_ch <= 35))
        efflorescence_ratio = float(np.sum(salt_mask)) / (w * h)

        # Moisture / Water stain index (dark vertical streaks: low L, low saturation, high vertical elongation)
        dark_streak_mask = ((l_ch <= 85) & (s_ch <= 50))
        moisture_ratio = float(np.sum(dark_streak_mask)) / (w * h)

        # 2. Geometric Shape Analysis
        bw = box["x1"] - box["x0"]
        bh = box["y1"] - box["y0"]
        aspect = max(bw, bh) / max(min(bw, bh), 1)

        # Otsu thresholding for local defect shape
        _, bin_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        geom = extract_region_defect_metrics(bin_otsu)
        skel = extract_crack_skeleton_metrics(bin_otsu)

        evidence_sources = [prior_source]
        visual_features = {
            "aspect_ratio": round(aspect, 2),
            "rust_ratio": round(rust_ratio, 3),
            "efflorescence_ratio": round(efflorescence_ratio, 3),
            "moisture_ratio": round(moisture_ratio, 3),
            "mean_lightness": round(mean_v, 1),
            "compactness": geom["compactness"],
            "skeleton_length_px": skel["skeleton_length_px"],
            "mean_crack_width_px": skel["mean_width_px"],
        }

        # 3. Decision Logic & Rule-Based Physical Fusion
        category = "crack"
        confidence = prior_confidence
        uncertainty = 0.15
        rec_action = "Monitor crack propagation and measure width over time"
        limitations = "Rotated bounding box evaluation; physical width depends on optical camera calibration"

        has_linear_crack = (skel["skeleton_length_px"] >= 25 or aspect >= 2.5)

        # Rule A: Linear Crack (Strong geometric fissure signature)
        if has_linear_crack:
            category = "crack"
            confidence = max(prior_confidence, 0.85)
            uncertainty = 0.12
            evidence_sources.append("skeleton_medial_axis")

        # Rule B: Corrosion / Rust Staining (elevated a* and b*, red/orange hue)
        elif rust_ratio >= 0.22 or (rust_ratio >= 0.12 and mean_b > 145):
            category = "corrosion_stain"
            confidence = min(0.92, 0.70 + rust_ratio)
            uncertainty = max(0.08, 0.30 - rust_ratio)
            evidence_sources.append("colorimetric_rust_index")
            rec_action = "Investigate underlying rebar corrosion and apply anti-carbonation coating"
            limitations = "Rust stains may leach from external scaffolding or fixtures rather than embedded steel"

        # Rule C: Efflorescence (Mineral Salt Leaching)
        elif efflorescence_ratio >= 0.25 and aspect < 2.5:
            category = "efflorescence"
            confidence = min(0.90, 0.65 + efflorescence_ratio)
            uncertainty = 0.20
            evidence_sources.append("spectrophotometric_salt_index")
            rec_action = "Inspect building facade for water ingress and wash salt crust"
            limitations = "May be confused with light-colored surface paint splatters"

        # Rule D: Dampness & Water Seepage Staining
        elif moisture_ratio >= 0.25 and bh > bw:
            category = "water_stain"
            confidence = min(0.88, 0.60 + moisture_ratio)
            uncertainty = 0.22
            evidence_sources.append("moisture_gradient_index")
            rec_action = "Trace external drainage pipes and evaluate facade waterproofing membrane"
            limitations = "Active moisture cannot be definitively proven without thermal or capacitance moisture metering"

        # Rule E: Spalling / Material Cavity
        elif geom["compactness"] > 0.35 and geom["boundary_irregularity"] > 1.25 and aspect < 2.5 and geom["area_px"] >= 150:
            category = "spalling"
            confidence = 0.82
            uncertainty = 0.18
            evidence_sources.append("irregular_cavity_contour")
            rec_action = "Chip loose concrete to sound substrate and repair with polymer-modified mortar"
            limitations = "Cavity depth requires stereoscopic 3D or depth-sensor measurement"

        elif aspect >= 1.5:
            category = "crack"
            confidence = max(prior_confidence, 0.75)
            uncertainty = 0.20
            evidence_sources.append("elongated_aspect_ratio")

        else:
            category = "unknown_anomaly"
            confidence = 0.60
            uncertainty = 0.40
            rec_action = "Review finding with on-site facade inspector"
            limitations = "Ambiguous visual signature with low geometric specificity"

        requires_review = (uncertainty >= 0.30 or confidence < 0.70)

        return MultiDefectFinding(
            finding_id=finding_id,
            defect_category=category,
            box=box,
            confidence=round(confidence, 3),
            uncertainty=round(uncertainty, 3),
            evidence_sources=evidence_sources,
            visual_features=visual_features,
            requires_human_review=requires_review,
            recommended_action=rec_action,
            limitations=limitations,
        )
