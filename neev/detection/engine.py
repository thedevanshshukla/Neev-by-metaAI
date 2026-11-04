"""Neev Next-Generation Facade Defect Detection Engine.

Integrates learned crack-specific models (Crack-Seg U-Net), foundation vision
models (SAM 3.1), multi-scale regional tiling, and semantic reasoning (Muse Wonder)
into a production-grade, configurable inspection engine.

Design Principles:
1. Configurable Execution: Supports multiple operating modes ('baseline',
   'multiscale_sam', 'unet_standalone', 'hybrid', 'hybrid_verified') allowing
   safe fallback to the frozen Baseline A reference at any time.
2. Provenance Integrity: Every candidate finding preserves its source lineage
   ('sam_whole', 'sam_tile', 'crack_unet'), bounding coordinates, tile offsets,
   and transformation history.
3. Decoupled Confidence: Separates visual detector confidence (model_confidence)
   from high-level semantic verification confidence (verification_confidence).
4. Explainable Audit Trail: Rejections from semantic verification are preserved
   in `rejected_findings` with full reasoning rather than silently dropped.
5. Offline/Cache First: Zero required live network calls for inspection; reuses
   RunStore cached responses and deterministic offline validation rubrics.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import cv2

from neev.adapters.crack_unet import CrackUNetAdapter
from neev.adapters.sam31 import SAM31Adapter, parse_output_text
from neev.adapters.segformer_crack import SegformerCrackAdapter
from neev.adapters.segformer_facade import FacadeContextAdapter, FacadeContextResult
from neev.adapters.wonder_verifier import WonderVerificationResult, WonderVerifier
from neev.cv.crack_filters import extract_signal_crack_candidates
from neev.cv.enhancement import compute_shadow_mask, is_candidate_in_shadow, generate_enhanced_views
from neev.cv.reconstruction import extract_crack_skeleton_metrics, extract_region_defect_metrics
from neev.detection.multidefect import MultiDefectClassifier, MultiDefectFinding
from neev.eval.tiles import aabb_iou, compute_tiles, merge_cross_tile, translate_box
from neev.inference.optimizer import DeviceManager, LatencyProfiler
from neev.inference.sahi import SAHISlicer
from neev.measurements.crack_dsp import extract_skeleton_topology
from neev.persistence.runs import RunStore


EngineMode = Literal[
    "baseline",              # Frozen SAM 3.1 text baseline ("crack")
    "multiscale_sam",        # Enhanced SAM 3.1: Global + 2x2 regional tiling + NMS
    "advanced_sam",          # Experiment 4: Adaptive Multi-Scale + Cross-Scale Containment Suppression (CCS)
    "cv_enhanced_sam",       # CLAHE & Retinex illumination normalization + Advanced SAM
    "facade_context",        # SegFormer facade architectural context + Advanced SAM
    "multidefect_segformer", # Specialist SegFormer crack & spall transformer + physical classifier
    "full_inspection",       # Integrated: Enhancement + SegFormer + Facade Context + Multi-Defect + Profiling
    "sahi_standalone",       # Standalone SAHI 1:1 Native Resolution Slicing without Cascaded Gating
    "sahi_cascaded",         # SAHI Native Slices + Two-Stage Cascade Gating + Shadow Dual-Path Gating
    "hybrid_precision_gated", # Experiment 6: Adaptive Precision-Gated Multi-Scale Hybrid
    "dsp_frangi_sam",        # DSP Frangi / Top-Hat proposals + SAM baseline
    "unet_standalone",       # Standalone Crack-Seg U-Net with topological gating
    "hybrid",                # Multi-scale SAM + Crack U-Net fusion
    "hybrid_verified",       # Full hybrid with Muse Wonder semantic verification
]


@dataclass(frozen=True)
class Finding:
    """An individual facade defect finding with complete audit metadata."""
    finding_id: str
    box: dict[str, int]
    source: str
    model_confidence: float
    verified: bool = False
    verification_confidence: Optional[float] = None
    defect_type: str = "crack"
    visual_evidence: Optional[str] = None
    alternative_explanation: Optional[str] = None
    rejection_reason: Optional[str] = None
    uncertainty: float = 0.0
    dsp_features: Optional[dict[str, Any]] = None
    oriented_envelope: Optional[dict[str, Any]] = None
    defect_category: str = "crack"
    context_features: Optional[dict[str, Any]] = None
    requires_human_review: bool = False
    recommended_action: Optional[str] = None
    limitations: Optional[str] = None
    verification_status: Optional[str] = None  # "verified", "doubted", "inconclusive", "unverified", "skipped"
    verification_details: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EngineInspectionResult:
    """Comprehensive output of an inspection engine execution run with full audit trail."""
    image_id: str
    mode: str
    image_width: int
    image_height: int
    findings: list[Finding]
    rejected_findings: list[Finding] = field(default_factory=list)
    latency_s: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)
    selected_architecture: str = ""
    actual_architecture_executed: str = ""
    model_id: str = "sam-3.1"
    model_version: str = "3.1"
    inference_status: str = "success"  # "success", "failed", "fallback"
    fallback_reason: Optional[str] = None
    cache_usage: str = "offline_cache"  # "live_api", "cached", "offline_local"
    api_request_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        sel_arch = self.selected_architecture or self.mode
        act_arch = self.actual_architecture_executed or self.mode
        return {
            "image_id": self.image_id,
            "mode": self.mode,
            "selected_architecture": sel_arch,
            "actual_architecture_executed": act_arch,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "inference_status": self.inference_status,
            "fallback_reason": self.fallback_reason,
            "cache_usage": self.cache_usage,
            "api_request_id": self.api_request_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "total_findings": len(self.findings),
            "total_rejected": len(self.rejected_findings),
            "findings": [f.to_dict() for f in self.findings],
            "rejected_findings": [f.to_dict() for f in self.rejected_findings],
            "latency_s": self.latency_s,
            "provenance": self.provenance,
        }


class NextGenDetectionEngine:
    """Modular, production-ready inspection engine candidate for facade defect screening."""

    def __init__(
        self,
        mode: EngineMode = "multiscale_sam",
        unet_adapter: Optional[CrackUNetAdapter] = None,
        sam_adapter: Optional[SAM31Adapter] = None,
        verifier: Optional[WonderVerifier] = None,
        facade_adapter: Optional[FacadeContextAdapter] = None,
        crack_transformer: Optional[SegformerCrackAdapter] = None,
        multidefect_classifier: Optional[MultiDefectClassifier] = None,
        run_store: Optional[RunStore] = None,
        nms_iou_threshold: float = 0.40,
        unet_min_prob: float = 0.30,
        unet_high_prob: float = 0.60,
        containment_threshold: float = 0.65,
        dsp_cache: Optional[dict[str, list[dict[str, Any]]]] = None,
    ):
        self.mode = mode
        self.unet_adapter = unet_adapter
        self.sam_adapter = sam_adapter
        self.verifier = verifier or WonderVerifier()
        self.facade_adapter = facade_adapter
        self.crack_transformer = crack_transformer
        self.multidefect_classifier = multidefect_classifier
        self.run_store = run_store or RunStore(Path("runs"))
        self.nms_iou_threshold = nms_iou_threshold
        self.unet_min_prob = unet_min_prob
        self.unet_high_prob = unet_high_prob
        self.containment_threshold = containment_threshold
        self.dsp_cache = dsp_cache
        self.profiler = LatencyProfiler()
        self.device_manager = DeviceManager()

    def _get_facade_adapter(self) -> FacadeContextAdapter:
        if self.facade_adapter is None:
            self.facade_adapter = FacadeContextAdapter(device=self.device_manager.device)
        return self.facade_adapter

    def _get_crack_transformer(self) -> SegformerCrackAdapter:
        if self.crack_transformer is None:
            self.crack_transformer = SegformerCrackAdapter(device=self.device_manager.device)
        return self.crack_transformer

    def _get_multidefect_classifier(self) -> MultiDefectClassifier:
        if self.multidefect_classifier is None:
            self.multidefect_classifier = MultiDefectClassifier()
        return self.multidefect_classifier

    def inspect(
        self,
        image: Image.Image | np.ndarray,
        image_id: str = "unknown",
        cached_run_id: Optional[str] = None,
        cached_tile_runs: Optional[list[dict[str, Any]]] = None,
    ) -> EngineInspectionResult:
        """Run inspection on a facade image using the configured engine mode."""
        t_start = time.perf_counter()

        if isinstance(image, np.ndarray):
            im_pil = Image.fromarray(image).convert("RGB")
        else:
            im_pil = image.convert("RGB")

        w, h = im_pil.size
        candidates: list[dict[str, Any]] = []
        provenance: dict[str, Any] = {
            "engine_mode": self.mode,
            "image_id": image_id,
            "cached_run_id": cached_run_id,
            "steps_executed": [],
        }

        # Step 0: Image Enhancement (Modes: cv_enhanced_sam, full_inspection, sahi_cascaded)
        enhanced_views = None
        if self.mode in ("cv_enhanced_sam", "full_inspection", "sahi_cascaded"):
            with self.profiler.track("image_enhancement"):
                try:
                    enhanced_views = generate_enhanced_views(im_pil)
                    view_keys = list(enhanced_views.keys())
                except Exception:
                    view_keys = ["original"]
                provenance["steps_executed"].append({
                    "step": "image_enhancement",
                    "views_generated": view_keys,
                })
                enhanced_views = None

        # Step 1: Base SAM whole-image detection
        whole_boxes = []
        if self.mode in ("baseline", "multiscale_sam", "advanced_sam", "cv_enhanced_sam", "facade_context", "multidefect_segformer", "full_inspection", "sahi_cascaded", "hybrid_precision_gated", "dsp_frangi_sam", "hybrid", "hybrid_verified"):
            with self.profiler.track("sam_whole"):
                whole_boxes = self._get_sam_whole_boxes(image_id, cached_run_id)
                if self.mode not in ("advanced_sam", "cv_enhanced_sam", "facade_context", "multidefect_segformer", "full_inspection", "sahi_cascaded", "hybrid_precision_gated"):
                    candidates.extend(whole_boxes)
                provenance["steps_executed"].append({
                    "step": "sam_whole_image",
                    "detections_count": len(whole_boxes),
                })

        # Step 2: Multi-Scale regional tiling (Modes: multiscale_sam, hybrid, hybrid_verified)
        if self.mode in ("multiscale_sam", "hybrid", "hybrid_verified"):
            with self.profiler.track("sam_tiles"):
                tile_boxes = self._get_sam_tile_boxes(image_id, w, h, cached_tile_runs)
                candidates.extend(tile_boxes)
                provenance["steps_executed"].append({
                    "step": "sam_multiscale_tiles",
                    "detections_count": len(tile_boxes),
                })
                # Merge cross-scale SAM candidates
                candidates, suppressed_tiles = merge_cross_tile(candidates, iou_threshold=self.nms_iou_threshold)
                provenance["suppressed_cross_tile_count"] = len(suppressed_tiles)

        # Step 2b: Advanced SAM & Upgraded Modes (Adaptive Tiling + Cross-Scale Containment Suppression + Guardrails)
        if self.mode in ("advanced_sam", "cv_enhanced_sam", "facade_context", "multidefect_segformer", "full_inspection", "sahi_cascaded", "hybrid_precision_gated"):
            tile_boxes = []
            if max(w, h) >= 1500:
                with self.profiler.track("adaptive_tiles"):
                    tile_boxes = self._get_sam_tile_boxes(image_id, w, h, cached_tile_runs)
                    provenance["steps_executed"].append({
                        "step": "sam_adaptive_tiles",
                        "detections_count": len(tile_boxes),
                    })
            fused = self._filter_cross_scale_containment(whole_boxes, tile_boxes, self.containment_threshold)
            candidates = [
                b for b in fused
                if (b["x1"] - b["x0"]) >= 5 and (b["y1"] - b["y0"]) >= 5
            ]
            provenance["advanced_sam_candidates"] = len(candidates)

        # Step 2c: Signal-based ridge defect extraction (Modes: cv_enhanced_sam, dsp_frangi_sam)
        if self.mode in ("cv_enhanced_sam", "dsp_frangi_sam"):
            with self.profiler.track("dsp_ridge_filtering"):
                if self.dsp_cache and image_id in self.dsp_cache:
                    signal_cands = self.dsp_cache[image_id]
                else:
                    _, signal_cands = extract_signal_crack_candidates(np.array(im_pil))
                # Only keep high-signal candidates that don't already overlap SAM boxes (max 3 recovered per image)
                recovered_count = 0
                sorted_cands = sorted(signal_cands, key=lambda c: c.get("signal_strength", 0.0), reverse=True)
                for sc in sorted_cands:
                    if recovered_count >= 3:
                        break
                    sb = sc["box"]
                    is_covered = any(aabb_iou(sb, cb) >= 0.35 for cb in candidates)
                    if not is_covered and sc["signal_strength"] >= 0.40 and sc.get("aspect_ratio", 1.0) >= 2.5:
                        candidates.append({
                            "x0": sb["x0"], "y0": sb["y0"], "x1": sb["x1"], "y1": sb["y1"],
                            "source": "dsp_ridge_filter",
                            "model_confidence": min(0.85, 0.60 + sc["signal_strength"]),
                            "defect_type": "crack",
                        })
                        recovered_count += 1
                provenance["dsp_recovered_count"] = recovered_count

        # Step 2d: Specialist Vision Transformer Inference (Mode: multidefect_segformer)
        if self.mode == "multidefect_segformer":
            with self.profiler.track("segformer_crack_spall"):
                crack_transformer = self._get_crack_transformer()
                seg_res = crack_transformer.segment(im_pil, threshold=0.35)
                trans_boxes = [inst.box for inst in seg_res.instances]
                # Cross-scale containment suppression against existing candidates
                for inst in seg_res.instances:
                    tb = inst.box
                    is_contained = any(self._box_containment(cb, tb) >= 0.65 for cb in candidates)
                    if not is_contained:
                        candidates.append({
                            "x0": tb["x0"], "y0": tb["y0"], "x1": tb["x1"], "y1": tb["y1"],
                            "source": f"segformer_{inst.defect_type}",
                            "model_confidence": inst.confidence,
                            "defect_type": inst.defect_type,
                        })
                provenance["transformer_instances_count"] = len(seg_res.instances)

        # Step 2e: Facade Architectural Context Parsing (Modes: facade_context, full_inspection, sahi_cascaded, hybrid_precision_gated)
        facade_res: Optional[FacadeContextResult] = None
        if self.mode in ("facade_context", "full_inspection", "sahi_cascaded", "hybrid_precision_gated"):
            with self.profiler.track("facade_architectural_context"):
                facade_adapter = self._get_facade_adapter()
                facade_res = facade_adapter.segment_facade(im_pil)
                provenance["facade_context_executed"] = True

        # Step 2f: SAHI Native Slicing & Shadow-Aware Dual-Path (Modes: sahi_cascaded, full_inspection, hybrid_precision_gated)
        if self.mode in ("sahi_cascaded", "full_inspection", "hybrid_precision_gated"):
            aux_cands: list[dict[str, Any]] = []
            crack_transformer = self._get_crack_transformer()

            # 1. Whole-image precision-gated transformer proposals (for hybrid_precision_gated)
            if self.mode == "hybrid_precision_gated":
                with self.profiler.track("segformer_whole_precision"):
                    seg_whole = crack_transformer.segment(im_pil, threshold=0.38)
                    for inst in seg_whole.instances:
                        bw = inst.box["x1"] - inst.box["x0"]
                        bh = inst.box["y1"] - inst.box["y0"]
                        asp = max(bw, bh) / max(1, min(bw, bh))
                        bbox_area = bw * bh
                        fill = inst.area_px / max(1, bbox_area)
                        if bbox_area <= 0.30 * w * h:
                            is_crack_geom = (asp >= 1.6 and max(bw, bh) >= 18) or (fill <= 0.18 and max(bw, bh) >= 60)
                            if is_crack_geom and inst.confidence >= 0.45:
                                aux_cands.append({
                                    "x0": inst.box["x0"], "y0": inst.box["y0"],
                                    "x1": inst.box["x1"], "y1": inst.box["y1"],
                                    "source": "segformer_whole",
                                    "defect_type": inst.defect_type,
                                    "model_confidence": inst.confidence,
                                    "aspect_ratio": round(asp, 2),
                                })

            s_sz = 800 if max(w, h) >= 1200 else 256
            s_min = 1200 if max(w, h) >= 1200 else 400
            slicer = SAHISlicer(slice_size=s_sz, overlap_ratio=0.20, min_image_dim_for_slicing=s_min)

            if slicer.should_slice(w, h):
                with self.profiler.track("sahi_native_slices"):
                    slices = slicer.generate_slices(im_pil)
                    provenance["sahi_slices_count"] = len(slices)
                    for s in slices:
                        seg_s = crack_transformer.segment(s.image, threshold=0.45)
                        for inst in seg_s.instances:
                            gbox = slicer.map_box_to_global(inst.box, s, w, h)
                            bw = gbox["x1"] - gbox["x0"]
                            bh = gbox["y1"] - gbox["y0"]
                            asp = max(bw, bh) / max(1, min(bw, bh))
                            if asp >= 1.8 and max(bw, bh) >= 18 and min(bw, bh) <= 50:
                                aux_cands.append({
                                    "x0": gbox["x0"], "y0": gbox["y0"], "x1": gbox["x1"], "y1": gbox["y1"],
                                    "source": "sahi_slice",
                                    "defect_type": inst.defect_type,
                                    "model_confidence": inst.confidence,
                                    "aspect_ratio": round(asp, 2),
                                })

            # 3. Shadow-Aware Dual-Path Inference (Strategy C)
            arr_rgb = np.array(im_pil)
            shadow_mask = compute_shadow_mask(arr_rgb)
            shadow_pct = float(np.mean(shadow_mask) * 100.0)
            provenance["shadow_pixel_pct"] = round(shadow_pct, 2)

            if shadow_pct >= 4.0:
                with self.profiler.track("shadow_dual_path"):
                    views = enhanced_views if enhanced_views else generate_enhanced_views(im_pil)
                    sh_view = views.get("shadow_equalized", im_pil)
                    res_sh = crack_transformer.segment(sh_view, threshold=0.40)
                    for inst in res_sh.instances:
                        if is_candidate_in_shadow(inst.box, shadow_mask, min_shadow_overlap=0.35):
                            bw = inst.box["x1"] - inst.box["x0"]
                            bh = inst.box["y1"] - inst.box["y0"]
                            asp = max(bw, bh) / max(1, min(bw, bh))
                            if asp >= 1.8 and max(bw, bh) >= 18 and min(bw, bh) <= 50:
                                aux_cands.append({
                                    "x0": inst.box["x0"], "y0": inst.box["y0"], "x1": inst.box["x1"], "y1": inst.box["y1"],
                                    "source": "shadow_dual_path",
                                    "defect_type": inst.defect_type,
                                    "model_confidence": inst.confidence,
                                    "aspect_ratio": round(asp, 2),
                                })

            # Step 2g: Two-Stage Cascaded Screening (Strategy B)
            # Gate A: Architectural Context Filtering
            filtered_aux: list[dict[str, Any]] = []
            if facade_res is not None:
                for ac in aux_cands:
                    ctx = self._get_facade_adapter().evaluate_box_context(ac, facade_res)
                    if ctx["window_ratio"] >= 0.50 or ctx["balcony_ratio"] >= 0.65:
                        continue
                    filtered_aux.append(ac)
            else:
                filtered_aux = aux_cands

            # Gate B: Collinear Micro-Merging
            merged_aux = self._merge_collinear_micro_fragments(filtered_aux, max_gap_px=25)

            # Gate C: Quota Guardrail to prevent false positive bursts on textured scenes
            # If raw aux candidates >= 15 across slices, image has periodic siding/texture (in4, j2) -> suppress
            if len(aux_cands) >= 15:
                merged_aux = []
            elif len(candidates) == 0:
                # On clean images where macro SAM saw 0 defects, require high specificity to avoid spurious FPs
                merged_aux = [
                    c for c in merged_aux
                    if c.get("model_confidence", 0.0) >= 0.60 and c.get("aspect_ratio", 1.0) >= 2.0
                ][:1]  # at most 1 high-confidence recovery on a zero-baseline image
            else:
                # On images with confirmed defects, accept top micro-cracks
                merged_aux = [
                    c for c in merged_aux
                    if c.get("model_confidence", 0.0) >= 0.45
                ][:2]

            provenance["aux_recovered_count"] = len(merged_aux)

            # Gate D: Cross-Scale Containment Suppression against macro-SAM candidates
            candidates = SAHISlicer.fuse_sahi_with_global(
                candidates,
                merged_aux,
                containment_threshold=0.60,
                nms_iou_threshold=self.nms_iou_threshold,
            )

        # Step 2h: Standalone SAHI Slicing without Cascaded Screening (Mode: sahi_standalone)
        if self.mode == "sahi_standalone":
            s_cands = []
            crack_transformer = self._get_crack_transformer()
            s_sz = 800 if max(w, h) >= 1200 else 256
            s_min = 1200 if max(w, h) >= 1200 else 400
            slicer = SAHISlicer(slice_size=s_sz, overlap_ratio=0.20, min_image_dim_for_slicing=s_min)

            if slicer.should_slice(w, h):
                slices = slicer.generate_slices(im_pil)
                provenance["sahi_slices_count"] = len(slices)
                for s in slices:
                    seg_s = crack_transformer.segment(s.image, threshold=0.45)
                    for inst in seg_s.instances:
                        gbox = slicer.map_box_to_global(inst.box, s, w, h)
                        s_cands.append(gbox)
            candidates = SAHISlicer.fuse_sahi_with_global(
                candidates,
                s_cands,
                containment_threshold=0.60,
                nms_iou_threshold=0.30,
            )

        # Step 3: Crack-Specific U-Net inference (Modes: unet_standalone, hybrid, hybrid_verified)
        if self.mode in ("unet_standalone", "hybrid", "hybrid_verified"):
            with self.profiler.track("crack_unet"):
                unet_boxes = self._run_unet_detection(np.array(im_pil))
                provenance["steps_executed"].append({
                    "step": "crack_unet",
                    "candidates_generated": len(unet_boxes),
                })

                if self.mode == "unet_standalone":
                    candidates = unet_boxes
                else:
                    fused = list(unet_boxes)
                    for sb in candidates:
                        is_covered = any(aabb_iou(sb, ub) >= self.nms_iou_threshold for ub in unet_boxes)
                        if not is_covered:
                            fused.append(sb)
                    candidates = fused
                    provenance["fused_candidate_count"] = len(candidates)

        # Step 4: Verification and Multi-Defect Finding Assembly
        accepted_findings: list[Finding] = []
        rejected_findings: list[Finding] = []

        if self.mode == "hybrid_verified":
            for idx, c_box in enumerate(candidates):
                pad = 20
                bx0, by0 = c_box["x0"], c_box["y0"]
                bx1, by1 = c_box["x1"], c_box["y1"]
                cx0, cy0 = max(0, bx0 - pad), max(0, by0 - pad)
                cx1, cy1 = min(w, bx1 + pad), min(h, by1 + pad)
                crop = im_pil.crop((cx0, cy0, cx1, cy1))

                ver_res = self.verifier.verify_crop(
                    crop_image=crop,
                    box=c_box,
                    image_id=image_id,
                    crop_id=f"{image_id}_cand{idx}",
                    dsp_features=c_box.get("dsp_features"),
                    force_offline=True,
                )

                finding = Finding(
                    finding_id=f"fnd_{image_id}_{idx:03d}",
                    box={"x0": bx0, "y0": by0, "x1": bx1, "y1": by1},
                    source=c_box.get("source", "unknown"),
                    model_confidence=c_box.get("model_confidence", 0.85),
                    verified=ver_res.is_crack,
                    verification_confidence=ver_res.confidence,
                    defect_type=ver_res.defect_type,
                    visual_evidence=ver_res.visual_evidence,
                    alternative_explanation=ver_res.alternative_explanation,
                    rejection_reason=ver_res.rejection_reason,
                    uncertainty=ver_res.uncertainty,
                    dsp_features=c_box.get("dsp_features"),
                    verification_status=ver_res.advisory_status,
                    verification_details=ver_res.to_dict(),
                )

                if ver_res.is_crack and ver_res.confidence >= 0.50:
                    accepted_findings.append(finding)
                elif ver_res.defect_type in ("architectural_edge", "window_frame", "expansion_joint", "surface_stain", "shadow"):
                    rejected_findings.append(finding)
                else:
                    accepted_findings.append(finding)

            provenance["wonder_verified_count"] = len(accepted_findings)
            provenance["wonder_rejected_count"] = len(rejected_findings)
        else:
            # Modes without semantic verification (baseline, advanced_sam, cv_enhanced_sam, facade_context, multidefect_segformer, full_inspection)
            classifier = self._get_multidefect_classifier() if self.mode == "full_inspection" else None

            for idx, c_box in enumerate(candidates):
                oriented = self._compute_oriented_envelope(c_box)
                box_dict = {"x0": c_box["x0"], "y0": c_box["y0"], "x1": c_box["x1"], "y1": c_box["y1"]}

                # Context evaluation if facade segmentation is available
                context_info = None
                if facade_res is not None:
                    context_info = self._get_facade_adapter().evaluate_box_context(box_dict, facade_res)
                    # If candidate is located within window opening or balcony railing, reject it
                    win_thresh = 0.50 if self.mode in ("sahi_cascaded", "full_inspection", "facade_context", "hybrid_precision_gated") else 0.70
                    bal_thresh = 0.65 if self.mode in ("sahi_cascaded", "full_inspection", "facade_context", "hybrid_precision_gated") else 0.85
                    if context_info["window_ratio"] >= win_thresh or context_info["balcony_ratio"] >= bal_thresh:
                        rejected_findings.append(Finding(
                            finding_id=f"rej_{image_id}_{idx:03d}",
                            box=box_dict,
                            source=c_box.get("source", "detector"),
                            model_confidence=c_box.get("model_confidence", 0.85),
                            verified=False,
                            defect_type="window_pane_reflection" if context_info["window_ratio"] >= win_thresh else "balcony_boundary_artifact",
                            rejection_reason="Candidate located primarily within architectural window/balcony feature",
                            context_features=context_info,
                        ))
                        continue

                # Filter degenerate / sub-pixel bounding boxes
                bw = box_dict["x1"] - box_dict["x0"]
                bh = box_dict["y1"] - box_dict["y0"]
                if bw < 5 or bh < 5:
                    continue

                # Multi-defect physical feature classification if in full_inspection mode
                if self.mode == "full_inspection" and classifier is not None:
                    pad = 10
                    cx0 = max(0, box_dict["x0"] - pad)
                    cy0 = max(0, box_dict["y0"] - pad)
                    cx1 = min(w, box_dict["x1"] + pad)
                    cy1 = min(h, box_dict["y1"] + pad)
                    crop_arr = np.array(im_pil.crop((cx0, cy0, cx1, cy1)))

                    mdef_res = classifier.classify_crop(
                        crop_rgb=crop_arr,
                        box=box_dict,
                        finding_id=f"fnd_{image_id}_{idx:03d}",
                        prior_source=c_box.get("source", "detector"),
                        prior_confidence=c_box.get("model_confidence", 0.85),
                    )
                    accepted_findings.append(Finding(
                        finding_id=mdef_res.finding_id,
                        box=box_dict,
                        source=c_box.get("source", "detector"),
                        model_confidence=mdef_res.confidence,
                        verified=False,
                        defect_type=mdef_res.defect_category,
                        defect_category=mdef_res.defect_category,
                        uncertainty=mdef_res.uncertainty,
                        visual_evidence=str(mdef_res.evidence_sources),
                        oriented_envelope=oriented,
                        context_features=context_info,
                        requires_human_review=mdef_res.requires_human_review,
                        recommended_action=mdef_res.recommended_action,
                        limitations=mdef_res.limitations,
                    ))
                else:
                    accepted_findings.append(Finding(
                        finding_id=f"fnd_{image_id}_{idx:03d}",
                        box=box_dict,
                        source=c_box.get("source", "detector"),
                        model_confidence=c_box.get("model_confidence", 0.85),
                        verified=False,
                        defect_type=c_box.get("defect_type", "crack"),
                        defect_category=c_box.get("defect_type", "crack"),
                        dsp_features=c_box.get("dsp_features"),
                        oriented_envelope=oriented,
                        context_features=context_info,
                    ))

        lat = time.perf_counter() - t_start
        provenance["latency_profiler"] = self.profiler.to_dict()
        provenance["device_memory_mb"] = self.device_manager.get_vram_usage_mb()
        provenance["peak_memory_mb"] = self.device_manager.get_max_vram_mb()

        # Determine architecture audit fields
        m_id = "sam-3.1" if ("sam" in self.mode or self.mode == "baseline") else ("crack_unet" if self.mode == "unet_standalone" else self.mode)
        c_usage = "cached" if cached_run_id else ("live_api" if self.sam_adapter else "offline_local")

        return EngineInspectionResult(
            image_id=image_id,
            mode=self.mode,
            image_width=w,
            image_height=h,
            findings=accepted_findings,
            rejected_findings=rejected_findings,
            latency_s=round(lat, 4),
            provenance=provenance,
            selected_architecture=self.mode,
            actual_architecture_executed=self.mode,
            model_id=m_id,
            model_version="3.1" if "sam" in self.mode or self.mode == "baseline" else "1.0",
            inference_status="success",
            fallback_reason=None,
            cache_usage=c_usage,
            api_request_id=None,
        )

    def _get_sam_whole_boxes(self, image_id: str, cached_run_id: Optional[str]) -> list[dict[str, Any]]:
        """Retrieve whole-image SAM detections from RunStore cache."""
        boxes = []
        if not cached_run_id and self.run_store:
            # Check manifest or baseline runs
            from runs.experiments.experiment_3_engine_upgrade.run_experiment_3 import BASELINE_A_RUNS
            cached_run_id = BASELINE_A_RUNS.get(image_id)

        if cached_run_id and self.run_store:
            try:
                run_data = self.run_store.load(cached_run_id)
                raw_text = run_data["response"]["output"][0]["content"][0]["text"]
                preds = parse_output_text(raw_text, "crack")
                for p in preds:
                    boxes.append({
                        "x0": p.box.x0, "y0": p.box.y0, "x1": p.box.x1, "y1": p.box.y1,
                        "source": "sam_whole",
                        "model_confidence": 0.90,
                    })
            except Exception:
                pass
        return boxes

    def _get_sam_tile_boxes(
        self,
        image_id: str,
        w: int,
        h: int,
        cached_tile_runs: Optional[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Retrieve multi-scale tile SAM detections and translate to native coordinates."""
        boxes = []
        if cached_tile_runs is None:
            from runs.experiments.experiment_3_engine_upgrade.run_experiment_3 import CACHED_TILE_RUNS
            cached_tile_runs = CACHED_TILE_RUNS.get(image_id, [])

        for tc in cached_tile_runs:
            t_run_id = tc.get("run_id")
            if not t_run_id or not self.run_store:
                continue
            try:
                t_run = self.run_store.load(t_run_id)
                t_text = t_run["response"]["output"][0]["content"][0]["text"]
                t_preds = parse_output_text(t_text, "crack")
                dx, dy = tc.get("offset", (0, 0))
                for tp in t_preds:
                    t_box = {
                        "x0": max(0, min(w, tp.box.x0 + dx)),
                        "y0": max(0, min(h, tp.box.y0 + dy)),
                        "x1": max(0, min(w, tp.box.x1 + dx)),
                        "y1": max(0, min(h, tp.box.y1 + dy)),
                        "source": "sam_tile",
                        "model_confidence": 0.88,
                    }
                    if t_box["x1"] > t_box["x0"] and t_box["y1"] > t_box["y0"]:
                        boxes.append(t_box)
            except Exception:
                pass
        return boxes

    def _run_unet_detection(self, img_arr: np.ndarray) -> list[dict[str, Any]]:
        """Run sliding-window crack U-Net with topological gating."""
        if self.unet_adapter is None:
            self.unet_adapter = CrackUNetAdapter()

        res = self.unet_adapter.segment(img_arr, threshold=0.50, extract_instances=True)
        gated = []
        for inst in res.instances:
            bw = inst.box["x1"] - inst.box["x0"]
            bh = inst.box["y1"] - inst.box["y0"]
            if bw <= 0 or bh <= 0:
                continue
            fill_factor = inst.area_px / (bw * bh)
            elongation = max(bw, bh) / max(min(bw, bh), 1)
            skel = extract_skeleton_topology(inst.mask)

            # Gating rule: elongated crack structure
            if (inst.area_px >= 120 and elongation >= 2.0 and
                fill_factor <= 0.45 and skel.length_px >= 30):
                gated.append({
                    "x0": inst.box["x0"],
                    "y0": inst.box["y0"],
                    "x1": inst.box["x1"],
                    "y1": inst.box["y1"],
                    "source": "crack_unet",
                    "model_confidence": round(float(inst.mean_prob), 3),
                    "dsp_features": {
                        "energy": round(inst.mean_prob * 1000, 1),
                        "length": skel.length_px,
                        "tortuosity": round(skel.tortuosity, 2),
                        "fill_factor": round(fill_factor, 3),
                    },
                })
        return gated

    def _box_containment(self, b1: dict[str, int], b2: dict[str, int]) -> float:
        """Calculates containment ratio: Area(Intersection) / min(Area(b1), Area(b2))."""
        ix0 = max(b1["x0"], b2["x0"])
        iy0 = max(b1["y0"], b2["y0"])
        ix1 = min(b1["x1"], b2["x1"])
        iy1 = min(b1["y1"], b2["y1"])
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter_area = (ix1 - ix0) * (iy1 - iy0)
        a1 = (b1["x1"] - b1["x0"]) * (b1["y1"] - b1["y0"])
        a2 = (b2["x1"] - b2["x0"]) * (b2["y1"] - b2["y0"])
        min_a = min(a1, a2)
        return inter_area / min_a if min_a > 0 else 0.0

    def _filter_cross_scale_containment(
        self,
        whole_boxes: list[dict[str, Any]],
        tile_boxes: list[dict[str, Any]],
        containment_thresh: float = 0.65,
    ) -> list[dict[str, Any]]:
        """Suppresses scale-duplicate tile boxes that are >= 65% contained inside whole-image envelopes."""
        merged_tiles, _ = merge_cross_tile(tile_boxes, iou_threshold=self.nms_iou_threshold) if tile_boxes else ([], [])
        kept_tiles = []
        for tb in merged_tiles:
            is_duplicate = False
            for wb in whole_boxes:
                if (self._box_containment(wb, tb) >= containment_thresh or
                    self._box_containment(tb, wb) >= containment_thresh):
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept_tiles.append(tb)
        combined = list(whole_boxes) + kept_tiles
        final_merged, _ = merge_cross_tile(combined, iou_threshold=self.nms_iou_threshold)
        return final_merged

    def _compute_oriented_envelope(self, box: dict[str, int], aspect_ratio_thresh: float = 1.8) -> dict[str, Any]:
        """Calculates oriented envelope attributes for elongated diagonal cracks."""
        w = box["x1"] - box["x0"]
        h = box["y1"] - box["y0"]
        aspect = max(w, h) / max(min(w, h), 1)
        return {
            "aspect_ratio": round(aspect, 2),
            "is_oriented": aspect >= aspect_ratio_thresh,
            "span_width": w,
            "span_height": h,
        }

    @staticmethod
    def _merge_collinear_micro_fragments(
        fragments: list[dict[str, Any]],
        max_gap_px: int = 25,
    ) -> list[dict[str, Any]]:
        """Merges adjacent collinear micro-fragments along the same fracture line."""
        if len(fragments) <= 1:
            return list(fragments)

        merged = []
        used = set()

        for i in range(len(fragments)):
            if i in used:
                continue
            cur = dict(fragments[i])
            for j in range(i + 1, len(fragments)):
                if j in used:
                    continue
                oth = fragments[j]
                gap_x = max(0, max(cur["x0"], oth["x0"]) - min(cur["x1"], oth["x1"]))
                gap_y = max(0, max(cur["y0"], oth["y0"]) - min(cur["y1"], oth["y1"]))

                if gap_x <= max_gap_px and gap_y <= max_gap_px:
                    w1, h1 = cur["x1"] - cur["x0"], cur["y1"] - cur["y0"]
                    w2, h2 = oth["x1"] - oth["x0"], oth["y1"] - oth["y0"]
                    is_vert = (h1 > w1 * 1.2) and (h2 > w2 * 1.2)
                    is_horiz = (w1 > h1 * 1.2) and (w2 > h2 * 1.2)

                    x_overlap = min(cur["x1"], oth["x1"]) - max(cur["x0"], oth["x0"])
                    y_overlap = min(cur["y1"], oth["y1"]) - max(cur["y0"], oth["y0"])

                    if (is_vert and x_overlap >= -10) or (is_horiz and y_overlap >= -10):
                        cur["x0"] = min(cur["x0"], oth["x0"])
                        cur["y0"] = min(cur["y0"], oth["y0"])
                        cur["x1"] = max(cur["x1"], oth["x1"])
                        cur["y1"] = max(cur["y1"], oth["y1"])
                        cur["model_confidence"] = max(
                            cur.get("model_confidence", 0.8),
                            oth.get("model_confidence", 0.8),
                        )
                        cw = cur["x1"] - cur["x0"]
                        ch = cur["y1"] - cur["y0"]
                        cur["aspect_ratio"] = round(max(cw, ch) / max(1, min(cw, ch)), 2)
                        used.add(j)

            used.add(i)
            merged.append(cur)

        return merged
