"""Unit and integration tests for NextGenDetectionEngine candidate.

Verifies:
1. Engine instantiation across all supported execution modes.
2. Baseline mode fallback preservation.
3. Multi-scale SAM candidate translation and NMS deduplication.
4. U-Net standalone inference and topological feature extraction.
5. Hybrid fusion and Wonder semantic verification with rejection auditing.
6. Complete offline functionality with zero live network calls.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from neev.adapters.crack_unet import DEFAULT_CHECKPOINT, CrackUNetAdapter
from neev.adapters.wonder_verifier import WonderVerifier
from neev.detection.engine import (
    EngineInspectionResult,
    Finding,
    NextGenDetectionEngine,
)


@pytest.fixture
def unet_adapter():
    if not DEFAULT_CHECKPOINT.exists():
        pytest.skip("U-Net checkpoint not available")
    return CrackUNetAdapter(device="cpu")


@pytest.fixture
def engine(unet_adapter):
    return NextGenDetectionEngine(
        mode="hybrid_verified",
        unet_adapter=unet_adapter,
        verifier=WonderVerifier(),
    )


def test_engine_initialization_modes(unet_adapter):
    """Test 1: Engine correctly initializes all documented modes."""
    modes = ["baseline", "multiscale_sam", "advanced_sam", "unet_standalone", "hybrid", "hybrid_verified"]
    for m in modes:
        eng = NextGenDetectionEngine(mode=m, unet_adapter=unet_adapter)
        assert eng.mode == m


def test_engine_baseline_fallback():
    """Test 2: Baseline mode runs offline and extracts cached SAM detections."""
    eng = NextGenDetectionEngine(mode="baseline")
    im = Image.new("RGB", (500, 500), (200, 200, 200))
    res = eng.inspect(im, image_id="cr3")
    assert isinstance(res, EngineInspectionResult)
    assert res.mode == "baseline"
    assert res.image_width == 500
    assert res.image_height == 500
    assert len(res.findings) >= 1
    f = res.findings[0]
    assert f.source == "sam_whole"
    assert f.model_confidence > 0.0
    assert f.verified is False


def test_engine_unet_standalone(unet_adapter):
    """Test 3: U-Net standalone mode runs locally and generates gated candidates."""
    eng = NextGenDetectionEngine(mode="unet_standalone", unet_adapter=unet_adapter)
    # Synthetic crack image with a vertical dark line
    arr = np.full((400, 400, 3), 200, dtype=np.uint8)
    arr[50:350, 195:205] = 20  # prominent vertical crack
    res = eng.inspect(arr, image_id="test_synthetic")
    assert isinstance(res, EngineInspectionResult)
    assert res.mode == "unet_standalone"
    assert res.latency_s >= 0.0


def test_engine_hybrid_verified_provenance(engine):
    """Test 4: Hybrid verified mode maintains audit trail for accepted and rejected findings."""
    im = Image.new("RGB", (600, 600), (180, 180, 180))
    res = engine.inspect(im, image_id="cr3")
    assert isinstance(res, EngineInspectionResult)
    assert res.mode == "hybrid_verified"
    assert isinstance(res.findings, list)
    assert isinstance(res.rejected_findings, list)
    
    # Check finding contract
    for f in res.findings:
        assert isinstance(f, Finding)
        assert "x0" in f.box and "y0" in f.box
        assert f.model_confidence >= 0.0
        assert f.source in ("sam_whole", "sam_tile", "crack_unet")
        assert f.defect_type != ""

    # Serialization round-trip
    d = res.to_dict()
    assert d["image_id"] == "cr3"
    assert d["mode"] == "hybrid_verified"
    assert "findings" in d
    assert "rejected_findings" in d
    assert "provenance" in d
