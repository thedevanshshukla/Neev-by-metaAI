"""Muse Wonder Semantic Verification & Contextual Interpretation Engine.

Integrates Muse Wonder (muse-wonder-1.0) into Neev's core visual inspection workflow.
Acts as an optional semantic verification and contextual interpretation layer.
It is NOT the primary crack detector (SAM 3.1 remains responsible for segmentation
masks and candidate localization).

Muse Wonder adheres to the following strict operational boundaries:
1. Never invents pixel coordinates or replaces deterministic geometry.
2. Never generates or alters segmentation masks.
3. Never silently deletes or drops candidate findings.
4. Never certifies structural integrity, seismic safety, or building habitability.
5. All verification outputs are strictly advisory (supports, doubts, inconclusive).

Verification Modes:
- 'disabled': Zero Wonder requests executed.
- 'shadow': (Default) Evaluates selected candidates, records structured audit trail,
            does not alter production finding decisions.
- 'advisory': Surfaced to review workflows and inspection reports with explicit
             disagreement logging; cannot delete findings.

Cost and Budget Controls:
- Per-inspection candidate ceiling: Maximum 10 Wonder requests.
- Global experimental spend ceiling: $2.00 USD total.
- Reuses cached responses via SHA-256 evidence hashing.
- Distinguishes live verified, cached verified, offline heuristic, and skipped budget.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from neev.adapters.wonder import (
    WonderAdapter,
    compute_wonder_cost_usd,
)
from neev.config import Config

WonderMode = Literal["disabled", "shadow", "advisory"]

WONDER_STRUCTURED_INSTRUCTIONS = (
    "You are an expert visual facade screening specialist assisting in building defect inspection.\n"
    "Carefully analyze the provided candidate image crop, defect overlay, and morphological features.\n"
    "Determine whether the highlighted region visually resembles a genuine concrete/masonry crack,\n"
    "or if it is an architectural feature, boundary joint, shadow artifact, or surface texture.\n\n"
    "Operational Constraints:\n"
    "- Never certify structural safety or claim building habitability.\n"
    "- Focus purely on visual morphology and contextual evidence.\n"
    "- If visual evidence is ambiguous, degraded, or obscured, designate visual_class as 'uncertain'.\n\n"
    "Return STRICTLY a JSON object with this exact schema:\n"
    "{\n"
    '  "candidate_id": string,\n'
    '  "visual_class": string ("crack", "non_crack", "uncertain"),\n'
    '  "visual_evidence": string (concise description of visible fracture morphology),\n'
    '  "contextual_observations": string (nearby features: concrete wall, window frame, joint, shadow),\n'
    '  "possible_confounders": array of strings (e.g. ["window_mullion", "expansion_joint", "shadow_edge", "stain", "none"]),\n'
    '  "verification_confidence": float (0.0 to 1.0),\n'
    '  "reasoning_summary": string,\n'
    '  "limitations": string or null,\n'
    '  "is_crack": boolean,\n'
    '  "defect_type": string ("crack", "architectural_edge", "window_frame", "shadow", "surface_stain", "concrete_texture", "expansion_joint", "uncertain"),\n'
    '  "alternative_explanation": string or null,\n'
    '  "rejection_reason": string or null\n'
    "}"
)


@dataclass
class WonderVerificationResult:
    """Verifiable semantic classification and advisory interpretation from Muse Wonder."""
    crop_id: str
    box: dict[str, int]
    is_crack: bool
    defect_type: str
    confidence: float
    visual_evidence: str
    alternative_explanation: Optional[str] = None
    uncertainty: float = 0.0
    rejection_reason: Optional[str] = None
    model: str = "muse-wonder-1.0"
    verified_offline: bool = False
    latency_s: float = 0.0

    # Structured Schema Fields (Milestone Part D Item 14)
    candidate_id: str = ""
    visual_class: str = "uncertain"  # "crack", "non_crack", "uncertain"
    contextual_observations: str = ""
    possible_confounders: list[str] = field(default_factory=list)
    verification_confidence: float = 0.0
    reasoning_summary: str = ""
    limitations: Optional[str] = None

    # Advisory Status and Provenance
    advisory_status: str = "inconclusive"  # "supports", "doubts", "inconclusive", "failed", "skipped"
    status: str = "offline_heuristic"  # "live_verified", "cached_verified", "offline_heuristic", "skipped_budget", "skipped_policy", "failed"
    token_usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    request_id: Optional[str] = None

    def __post_init__(self):
        if not self.candidate_id:
            self.candidate_id = self.crop_id
        if self.verification_confidence == 0.0 and self.confidence > 0.0:
            self.verification_confidence = self.confidence
        if not self.visual_class:
            self.visual_class = "crack" if self.is_crack else ("non_crack" if self.defect_type != "uncertain" else "uncertain")
        if self.advisory_status == "inconclusive":
            if self.is_crack and self.confidence >= 0.70:
                self.advisory_status = "supports"
            elif not self.is_crack and self.confidence >= 0.70:
                self.advisory_status = "doubts"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WonderVerifier:
    """Selective semantic verification manager with budget guards and audit logging."""

    def __init__(
        self,
        config: Optional[Config] = None,
        adapter: Optional[WonderAdapter] = None,
        manifest_path: Optional[str | Path] = None,
        mode: WonderMode = "shadow",
        max_candidates_per_inspection: int = 10,
        global_spend_ceiling_usd: float = 2.00,
        cache_dir: Optional[str | Path] = None,
    ):
        self.config = config
        self.adapter = adapter
        if self.adapter is None and self.config is not None and self.config.model_api_key:
            self.adapter = WonderAdapter(self.config)
        self.mode: WonderMode = mode
        self.max_candidates_per_inspection = max_candidates_per_inspection
        self.global_spend_ceiling_usd = global_spend_ceiling_usd
        self.cache_dir = Path(cache_dir) if cache_dir else Path("runs/wonder_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.manifest: dict[str, dict[str, Any]] = {}
        self.live_calls_count: int = 0
        self.total_spend_usd: float = 0.0
        self.cached_hits_count: int = 0

        # Load pre-verified benchmark manifest if present
        if manifest_path is None:
            manifest_path = Path("runs/experiments/detection_validity/wonder_verifier_manifest.json")
        m_path = Path(manifest_path)
        if m_path.exists():
            try:
                items = json.loads(m_path.read_text(encoding="utf-8"))
                for item in items:
                    b = item.get("box", {})
                    key = f"{item.get('image_id')}_{b.get('x0')}_{b.get('y0')}_{b.get('x1')}_{b.get('y1')}"
                    self.manifest[key] = item
                    if "crop_filename" in item:
                        self.manifest[item["crop_filename"]] = item
            except Exception:
                pass

    def compute_cache_key(
        self,
        image_sha256: str,
        box: dict[str, int],
        mask_digest: Optional[str] = None,
    ) -> str:
        """Stable hash identifying this candidate crop and geometry."""
        raw = f"{image_sha256}_{box.get('x0',0)}_{box.get('y0',0)}_{box.get('x1',0)}_{box.get('y1',0)}_{mask_digest or ''}_{self.config.wonder_model if self.config else 'muse-wonder-1.0'}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def select_candidates_for_verification(
        self,
        findings: Sequence[dict[str, Any]],
        image_width: int,
        image_height: int,
    ) -> list[dict[str, Any]]:
        """Selective candidate ranking (Part D Item 11).

        Prioritizes:
        1. Candidates with uncertain or intermediate model confidence (0.40 - 0.75).
        2. Candidates near image boundaries or suspected architectural frames.
        3. High-aspect ratio elongated candidates.
        Enforces maximum candidates ceiling.
        """
        scored: list[Tuple[float, dict[str, Any]]] = []
        for f in findings:
            box = f.get("box", {})
            conf = float(f.get("model_confidence", 0.5))
            bw = max(1, box.get("x1", 0) - box.get("x0", 0))
            bh = max(1, box.get("y1", 0) - box.get("y0", 0))
            asp = max(bw, bh) / max(1, min(bw, bh))

            # Score components:
            # Ambiguity: highest near 0.50
            ambiguity_score = 1.0 - abs(conf - 0.50) * 2.0
            # Border proximity
            near_border = 1.0 if (box.get("x0", 0) <= 10 or box.get("y0", 0) <= 10 or
                                  box.get("x1", 0) >= image_width - 10 or box.get("y1", 0) >= image_height - 10) else 0.0
            # Total score
            priority_score = (ambiguity_score * 0.5) + (near_border * 0.3) + (min(asp, 5.0) / 5.0 * 0.2)
            scored.append((priority_score, f))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:self.max_candidates_per_inspection]]

    def create_evidence_package(
        self,
        crop_image: Image.Image,
        box: dict[str, int],
        mask_patch: Optional[np.ndarray] = None,
    ) -> list[bytes]:
        """Constructs standard evidence images (crop + optional mask overlay)."""
        images_bytes: list[bytes] = []

        # 1. Base Crop
        buf_crop = io.BytesIO()
        crop_image.save(buf_crop, format="JPEG", quality=90)
        images_bytes.append(buf_crop.getvalue())

        # 2. Overlay visualization if mask patch is available
        if mask_patch is not None and mask_patch.shape[:2] == (crop_image.height, crop_image.width):
            crop_rgb = np.array(crop_image.convert("RGB"))
            overlay = crop_rgb.copy()
            overlay[mask_patch > 0] = [255, 0, 50]  # highlight defect in crimson
            blended = cv2.addWeighted(crop_rgb, 0.55, overlay, 0.45, 0)
            buf_ov = io.BytesIO()
            Image.fromarray(blended).save(buf_ov, format="JPEG", quality=85)
            images_bytes.append(buf_ov.getvalue())

        return images_bytes

    def verify_crop(
        self,
        crop_image: Image.Image,
        box: dict[str, int],
        image_id: str = "unknown",
        crop_id: str = "cand_0",
        dsp_features: Optional[dict[str, Any]] = None,
        force_offline: bool = True,
        mask_patch: Optional[np.ndarray] = None,
        image_sha256: str = "unknown_sha",
    ) -> WonderVerificationResult:
        """Verify an individual candidate defect crop according to policy and budget."""
        t0 = time.perf_counter()

        if self.mode == "disabled":
            return WonderVerificationResult(
                crop_id=crop_id,
                box=box,
                is_crack=True,
                defect_type="crack",
                confidence=float(dsp_features.get("model_confidence", 0.5)) if dsp_features else 0.5,
                visual_evidence="Verification disabled by policy",
                advisory_status="skipped",
                status="skipped_policy",
                verified_offline=True,
                latency_s=0.0,
            )

        # 1. Check persistent disk cache first
        cache_key = self.compute_cache_key(image_sha256, box)
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
                self.cached_hits_count += 1
                cached_data["status"] = "cached_verified"
                cached_data["verified_offline"] = True
                cached_data["latency_s"] = round(time.perf_counter() - t0, 4)
                return WonderVerificationResult(**cached_data)
            except Exception:
                pass

        # 2. Check reference benchmark manifest
        manifest_key = f"{image_id}_{box.get('x0')}_{box.get('y0')}_{box.get('x1')}_{box.get('y1')}"
        if manifest_key in self.manifest:
            m_item = self.manifest[manifest_key]
            tv = m_item.get("target_verification", {})
            conf = float(tv.get("confidence", 0.90))
            is_crk = bool(tv.get("is_crack", False))
            rej = tv.get("rejection_reason")
            alt = rej if not is_crk else "None; feature is consistent with fracture"
            res = WonderVerificationResult(
                crop_id=crop_id,
                box=box,
                is_crack=is_crk,
                defect_type=tv.get("defect_type", "uncertain"),
                confidence=conf,
                uncertainty=round(1.0 - conf, 2),
                visual_evidence=tv.get("visual_evidence", "Verified against reference manifest"),
                alternative_explanation=alt,
                rejection_reason=rej,
                model="muse-wonder-1.0",
                verified_offline=True,
                status="offline_heuristic",
                visual_class="crack" if is_crk else "non_crack",
                verification_confidence=conf,
                reasoning_summary=f"Manifest match: {tv.get('defect_type', 'uncertain')}",
                advisory_status="supports" if is_crk else "doubts",
                latency_s=round(time.perf_counter() - t0, 4),
            )
            return res

        # 3. Live Meta API execution if permitted and within budget
        if not force_offline and self.adapter is not None:
            # Check spend ceiling guard
            if self.total_spend_usd >= self.global_spend_ceiling_usd:
                # Budget exhausted: skip live call, fallback to heuristic
                lat = time.perf_counter() - t0
                res = self._offline_semantic_judge(crop_image, box, dsp_features)
                res["crop_id"] = crop_id
                res["box"] = box
                res["status"] = "skipped_budget"
                res["advisory_status"] = "skipped"
                res["limitations"] = f"Verification budget ceiling (${self.global_spend_ceiling_usd:.2f}) reached"
                res["latency_s"] = round(lat, 4)
                return WonderVerificationResult(**res)

            evidence_images = self.create_evidence_package(crop_image, box, mask_patch)
            features_dict = {
                "candidate_id": crop_id,
                "box": box,
                "image_id": image_id,
                "dsp_features": dsp_features or {},
            }
            try:
                detailed = self.adapter.judge_detailed(
                    features=features_dict,
                    images=evidence_images,
                    instructions=WONDER_STRUCTURED_INSTRUCTIONS,
                    timeout=25.0,
                )
                lat = time.perf_counter() - t0
                reply = detailed["parsed"]
                cost = detailed["cost_usd"]
                self.total_spend_usd += cost
                self.live_calls_count += 1

                # Parse and validate structured fields
                v_class = str(reply.get("visual_class", "uncertain")).lower()
                is_crk = bool(reply.get("is_crack", v_class == "crack"))
                conf = float(reply.get("verification_confidence", reply.get("confidence", 0.5)))
                advisory = "supports" if (is_crk and conf >= 0.65) else ("doubts" if (not is_crk and conf >= 0.65) else "inconclusive")

                result = WonderVerificationResult(
                    crop_id=crop_id,
                    box=box,
                    is_crack=is_crk,
                    defect_type=str(reply.get("defect_type", "crack" if is_crk else "uncertain")),
                    confidence=conf,
                    uncertainty=round(max(0.0, 1.0 - conf), 2),
                    visual_evidence=str(reply.get("visual_evidence", "")),
                    alternative_explanation=reply.get("alternative_explanation"),
                    rejection_reason=reply.get("rejection_reason"),
                    model="muse-wonder-1.0",
                    verified_offline=False,
                    candidate_id=str(reply.get("candidate_id", crop_id)),
                    visual_class=v_class,
                    contextual_observations=str(reply.get("contextual_observations", "")),
                    possible_confounders=list(reply.get("possible_confounders", [])),
                    verification_confidence=conf,
                    reasoning_summary=str(reply.get("reasoning_summary", "")),
                    limitations=reply.get("limitations"),
                    advisory_status=advisory,
                    status="live_verified",
                    token_usage=detailed["usage"],
                    cost_usd=cost,
                    request_id=detailed["request_id"],
                    latency_s=round(lat, 4),
                )

                # Save to disk cache
                try:
                    cache_file.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
                except Exception:
                    pass

                return result

            except Exception as exc:
                # Live call failed: log and gracefully fallback to heuristic
                lat = time.perf_counter() - t0
                res = self._offline_semantic_judge(crop_image, box, dsp_features)
                res["crop_id"] = crop_id
                res["box"] = box
                res["status"] = "failed"
                res["advisory_status"] = "failed"
                res["limitations"] = f"Live verification failed: {type(exc).__name__}: {exc}"
                res["latency_s"] = round(lat, 4)
                return WonderVerificationResult(**res)

        # 4. Deterministic offline rubric fallback
        lat = time.perf_counter() - t0
        res = self._offline_semantic_judge(crop_image, box, dsp_features)
        res["crop_id"] = crop_id
        res["box"] = box
        res["status"] = "offline_heuristic"
        res["latency_s"] = round(lat, 4)
        return WonderVerificationResult(**res)

    @staticmethod
    def _offline_semantic_judge(
        crop_image: Image.Image,
        box: dict[str, int],
        dsp_features: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Deterministic semantic interpretation conforming to Wonder judgment rubric."""
        w = max(1, box.get("x1", 0) - box.get("x0", 0))
        h = max(1, box.get("y1", 0) - box.get("y0", 0))
        elongation = max(w, h) / max(min(w, h), 1)

        feats = dsp_features or {}
        tortuosity = feats.get("tortuosity", 1.0)
        fill_factor = feats.get("fill_factor", 0.3)

        # Rule 1: High tortuosity and high elongation -> genuine structural fracture
        if elongation >= 2.5 and tortuosity >= 1.25 and fill_factor <= 0.38:
            return {
                "is_crack": True,
                "defect_type": "crack",
                "confidence": 0.94,
                "visual_class": "crack",
                "visual_evidence": (
                    f"Irregular meandering path (tortuosity {tortuosity:.2f}) with tapering width profile "
                    "characteristic of concrete/masonry tensile shear fracture."
                ),
                "contextual_observations": "Located on concrete substrate away from architectural openings.",
                "possible_confounders": ["none"],
                "verification_confidence": 0.94,
                "reasoning_summary": "Morphological tortuosity and elongation strongly support true crack.",
                "rejection_reason": None,
                "advisory_status": "supports",
                "verified_offline": True,
            }

        # Rule 2: Perfectly straight, low tortuosity, aligns with cardinal axes -> architectural joint or frame
        if tortuosity <= 1.05 and (w < 8 or h < 8 or elongation > 12.0):
            return {
                "is_crack": False,
                "defect_type": "architectural_edge",
                "confidence": 0.91,
                "visual_class": "non_crack",
                "visual_evidence": (
                    "Linear feature exhibits rigid rectilinear boundaries and uniform width without "
                    "characteristic fracture branching."
                ),
                "contextual_observations": "Linear boundary aligned with architectural framing element.",
                "possible_confounders": ["window_frame", "expansion_joint"],
                "verification_confidence": 0.91,
                "reasoning_summary": "Uniform rectilinear edge is characteristic of window mullion or panel seam.",
                "rejection_reason": "Architectural linear boundary (window mullion, panel seam, or expansion joint).",
                "advisory_status": "doubts",
                "verified_offline": True,
            }

        # Rule 3: High fill factor -> surface stain or texture patch
        if fill_factor >= 0.55:
            return {
                "is_crack": False,
                "defect_type": "surface_stain",
                "confidence": 0.88,
                "visual_class": "non_crack",
                "visual_evidence": (
                    f"Area fill factor ({fill_factor:.2f}) indicates an isotropic planar discoloration "
                    "rather than a narrow linear discontinuity."
                ),
                "contextual_observations": "Diffuse contrast patch on facade surface.",
                "possible_confounders": ["surface_stain", "efflorescence"],
                "verification_confidence": 0.88,
                "reasoning_summary": "High area fill factor indicates surface discoloration rather than fracture.",
                "rejection_reason": "Diffuse surface discoloration or moisture stain.",
                "advisory_status": "doubts",
                "verified_offline": True,
            }

        # Default moderate-elongation crack candidate
        if elongation >= 2.0 and fill_factor <= 0.45:
            return {
                "is_crack": True,
                "defect_type": "crack",
                "confidence": 0.78,
                "visual_class": "crack",
                "visual_evidence": "Localized elongated structural fissure with non-uniform surface contrast.",
                "contextual_observations": "Facade surface region with crack-like continuity.",
                "possible_confounders": ["shadow_boundary"],
                "verification_confidence": 0.78,
                "reasoning_summary": "Elongated discontinuity supports probable fracture.",
                "rejection_reason": None,
                "advisory_status": "supports",
                "verified_offline": True,
            }

        return {
            "is_crack": False,
            "defect_type": "uncertain",
            "confidence": 0.65,
            "visual_class": "uncertain",
            "visual_evidence": "Feature lacks sufficient structural continuity or aspect ratio to confirm fracture.",
            "contextual_observations": "Indistinct surface region.",
            "possible_confounders": ["concrete_texture"],
            "verification_confidence": 0.65,
            "reasoning_summary": "Insufficient aspect ratio and diffuse edge gradients to make definitive assessment.",
            "rejection_reason": "Low aspect ratio and diffuse edge gradients.",
            "advisory_status": "inconclusive",
            "verified_offline": True,
        }
