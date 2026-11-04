"""Unit and integration tests for Neev Capability Upgrade.

Verifies:
1. Image enhancement module (CLAHE, Retinex, bilateral filtering, multi-view generation).
2. Physics-based crack filtering (Frangi vesselness, Sato tubularity, Bottom-Hat morphology).
3. Geometric mask reconstruction (skeleton length, endpoints, branch points, width profiling, compactness).
4. Facade architectural context adapter (SegFormer CMP Facade wall/window/balcony parsing).
5. Specialist crack & spalling transformer adapter (SegFormer defect segmentation).
6. Multi-defect physical & spectral classification (cracks, spalling, corrosion, efflorescence, water stains).
7. Inference latency profiling and VRAM management.
8. Execution of all upgraded engine modes ('cv_enhanced_sam', 'facade_context', 'multidefect_segformer', 'full_inspection').
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from neev.adapters.segformer_crack import SegformerCrackAdapter
from neev.adapters.segformer_facade import FacadeContextAdapter
from neev.cv.crack_filters import (
    compute_black_tophat,
    compute_frangi_vesselness,
    compute_sato_tubularity,
    extract_signal_crack_candidates,
)
from neev.cv.enhancement import (
    apply_clahe_lab,
    apply_edge_preserving_denoise,
    apply_multiscale_retinex,
    generate_enhanced_views,
)
from neev.cv.reconstruction import (
    extract_crack_skeleton_metrics,
    extract_region_defect_metrics,
    refine_defect_mask,
)
from neev.detection.engine import EngineInspectionResult, Finding, NextGenDetectionEngine
from neev.detection.multidefect import MultiDefectClassifier, MultiDefectFinding
from neev.inference.optimizer import DeviceManager, LatencyProfiler, TileBatcher


def test_image_enhancement_suite():
    """Verify CLAHE, Retinex, and multi-view generation maintain dimensions and range."""
    img = np.random.randint(40, 220, (150, 200, 3), dtype=np.uint8)
    views = generate_enhanced_views(img)
    assert "original" in views
    assert "clahe" in views
    assert "retinex" in views
    assert "denoised" in views
    assert "shadow_equalized" in views

    for k, v in views.items():
        assert isinstance(v, Image.Image)
        assert v.size == (200, 150)


def test_physics_crack_filters():
    """Verify Frangi, Sato, and Bottom-Hat transforms register linear ridge responses."""
    gray = np.full((120, 120), 200, dtype=np.uint8)
    # Add dark crack line
    gray[30:90, 60] = 20

    frangi_out = compute_frangi_vesselness(gray)
    sato_out = compute_sato_tubularity(gray)
    tophat_out = compute_black_tophat(gray)

    assert frangi_out.shape == (120, 120)
    assert sato_out.shape == (120, 120)
    assert tophat_out.shape == (120, 120)
    # Peak response along the crack coordinate
    assert frangi_out[50, 60] > frangi_out[50, 10]
    assert sato_out[50, 60] > sato_out[50, 10]


def test_signal_candidate_extraction():
    """Verify faint crack candidate recovery from fused ridge signal."""
    rgb = np.full((100, 100, 3), 180, dtype=np.uint8)
    rgb[20:80, 45:47] = 25  # thin vertical crack
    energy, cands = extract_signal_crack_candidates(rgb, percentile_threshold=95.0, min_area_px=15)
    assert energy.shape == (100, 100)
    assert len(cands) >= 1
    c0 = cands[0]
    assert c0["source"] == "dsp_ridge_filter"
    assert "x0" in c0["box"]
    assert c0["aspect_ratio"] >= 1.0


def test_geometric_reconstruction_metrics():
    """Verify skeleton extraction and region metrics."""
    # Diagonal crack line
    mask = np.zeros((80, 80), dtype=np.uint8)
    for i in range(20, 60):
        mask[i, i] = 1

    skel_info = extract_crack_skeleton_metrics(mask)
    assert skel_info["skeleton_length_px"] >= 35.0
    assert skel_info["endpoints_count"] >= 2
    assert skel_info["mean_width_px"] > 0

    # Spall circular cavity
    cavity = np.zeros((80, 80), dtype=np.uint8)
    cavity[25:55, 25:55] = 1
    reg_info = extract_region_defect_metrics(cavity)
    assert reg_info["area_px"] == 900
    assert reg_info["compactness"] > 0.70
    assert reg_info["extent"] == pytest.approx(1.0, abs=0.05)


def test_multidefect_classification_color_and_geometry():
    """Verify multi-defect classifier distinguishes rust stains from cracks."""
    classifier = MultiDefectClassifier()

    # 1. Rust stain crop (orange/brown pixels)
    rust_crop = np.full((60, 60, 3), (180, 75, 25), dtype=np.uint8)
    res_rust = classifier.classify_crop(rust_crop, {"x0": 10, "y0": 10, "x1": 70, "y1": 70})
    assert isinstance(res_rust, MultiDefectFinding)
    assert res_rust.defect_category == "corrosion_stain"
    assert "colorimetric_rust_index" in res_rust.evidence_sources

    # 2. Linear crack crop
    crack_crop = np.full((60, 60, 3), 200, dtype=np.uint8)
    crack_crop[:, 28:32] = 20
    res_crack = classifier.classify_crop(crack_crop, {"x0": 20, "y0": 10, "x1": 80, "y1": 70})
    assert res_crack.defect_category in ("crack", "unknown_anomaly")


def test_device_manager_and_profiler():
    """Verify latency profiling and memory tracking."""
    dev_mgr = DeviceManager()
    assert dev_mgr.device in ("cuda", "cpu")

    profiler = LatencyProfiler()
    with profiler.track("stage_a"):
        _ = [i ** 2 for i in range(10000)]

    res = profiler.to_dict()
    assert "stage_a" in res["stages_ms"]
    assert res["total_latency_ms"] >= 0.0


def test_upgraded_engine_modes():
    """Verify execution of upgraded engine modes without errors."""
    im = Image.new("RGB", (400, 400), (200, 200, 200))
    modes = ["cv_enhanced_sam", "facade_context", "multidefect_segformer", "sahi_cascaded", "full_inspection"]

    for m in modes:
        eng = NextGenDetectionEngine(mode=m)
        res = eng.inspect(im, image_id="cr3")
        assert isinstance(res, EngineInspectionResult)
        assert res.mode == m
        assert res.image_width == 400
        assert res.image_height == 400
        assert "provenance" in res.to_dict()
        assert "latency_profiler" in res.provenance


def test_sahi_slicing_and_coordinate_mapping():
    """Verify SAHI sliding window generation and global coordinate translation."""
    from neev.inference.sahi import SAHISlicer

    im = Image.new("RGB", (1600, 1200), (128, 128, 128))
    slicer = SAHISlicer(slice_size=800, overlap_ratio=0.20, min_image_dim_for_slicing=1200)
    assert slicer.should_slice(1600, 1200) is True
    assert slicer.should_slice(800, 600) is False

    slices = slicer.generate_slices(im)
    assert len(slices) >= 4
    for s in slices:
        assert s.width <= 800
        assert s.height <= 800

    # Test coordinate mapping
    s0 = slices[0]
    local_box = {"x0": 50, "y0": 60, "x1": 150, "y1": 180}
    global_box = slicer.map_box_to_global(local_box, s0, 1600, 1200)
    assert global_box["x0"] == s0.x0 + 50
    assert global_box["y0"] == s0.y0 + 60
    assert global_box["x1"] == s0.x0 + 150
    assert global_box["y1"] == s0.y0 + 180


def test_shadow_mask_and_dual_path_gating():
    """Verify shadow mask computation and shadow bounding box gating."""
    from neev.cv.enhancement import compute_shadow_mask, is_candidate_in_shadow

    # Image with top half brightly lit and bottom half dark (shadowed)
    img_arr = np.full((200, 200, 3), 200, dtype=np.uint8)
    img_arr[100:, :] = 40  # dark shadow region

    smask = compute_shadow_mask(img_arr, blur_kernel_size=51, threshold_ratio=0.45)
    assert smask.shape == (200, 200)
    assert np.mean(smask[120:180, :]) > 0.70  # high shadow probability in bottom half
    assert np.mean(smask[10:80, :]) < 0.20    # low shadow probability in top half

    # Box completely inside shadow
    box_shadow = {"x0": 20, "y0": 120, "x1": 80, "y1": 180}
    assert is_candidate_in_shadow(box_shadow, smask, min_shadow_overlap=0.40) is True

    # Box in bright sunlit region
    box_sunlit = {"x0": 20, "y0": 20, "x1": 80, "y1": 70}
    assert is_candidate_in_shadow(box_sunlit, smask, min_shadow_overlap=0.40) is False


def test_collinear_micro_merging():
    """Verify adjacent collinear micro-fragments are merged into compound defects."""
    # Two collinear vertical crack fragments separated by a 10px gap
    frags = [
        {"x0": 50, "y0": 20, "x1": 65, "y1": 80, "model_confidence": 0.85},
        {"x0": 52, "y0": 90, "x1": 66, "y1": 150, "model_confidence": 0.90},
    ]
    merged = NextGenDetectionEngine._merge_collinear_micro_fragments(frags, max_gap_px=20)
    assert len(merged) == 1
    m0 = merged[0]
    assert m0["x0"] == 50
    assert m0["y0"] == 20
    assert m0["x1"] == 66
    assert m0["y1"] == 150
    assert m0["model_confidence"] == 0.90

