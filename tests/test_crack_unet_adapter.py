import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neev.adapters.crack_unet import (
    CrackUNetAdapter,
    DEFAULT_CHECKPOINT,
    UNetSegmentationResult,
)


@pytest.fixture
def unet_adapter():
    if not DEFAULT_CHECKPOINT.exists():
        pytest.skip("U-Net checkpoint not downloaded")
    return CrackUNetAdapter(device="cpu")


def test_checkpoint_compatibility(unet_adapter):
    """Test 1: Checkpoint loads strictly and detects expected feature widths."""
    assert unet_adapter.model is not None
    assert unet_adapter.features == [32, 64, 128, 256]
    assert unet_adapter.device.type == "cpu"


def test_input_preprocessing_and_output_shapes(unet_adapter):
    """Test 2 & 3: Input preprocessing, output shapes, and numerical validity."""
    # Synthetic 480x480 RGB image
    img = np.full((480, 480, 3), 128, dtype=np.uint8)
    res = unet_adapter.segment(img, threshold=0.5, extract_instances=False)

    assert isinstance(res, UNetSegmentationResult)
    assert res.probs.shape == (480, 480)
    assert res.binary.shape == (480, 480)
    # Check probabilities bounded in [0, 1]
    assert np.all(res.probs >= 0.0)
    assert np.all(res.probs <= 1.0)
    assert np.isfinite(res.probs).all()
    # Binary mask contains only 0 and 1
    assert set(np.unique(res.binary)).issubset({0, 1})
    assert 0.0 <= res.crack_area_ratio <= 1.0


def test_empty_predictions_handling(unet_adapter):
    """Test 4: Blank image produces zero crack instances without error."""
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    res = unet_adapter.segment(blank, threshold=0.9, extract_instances=True)

    assert res.instances == []
    assert res.binary.sum() == 0
    assert res.crack_area_ratio == 0.0


def test_thin_and_branching_crack_mask_extraction(unet_adapter):
    """Test 5: Synthetic thin crack line produces well-formed instance and polygon."""
    # Create synthetic binary mask with a diagonal crack line
    binary = np.zeros((200, 200), dtype=np.uint8)
    for i in range(30, 150):
        binary[i, i : i + 2] = 1  # 2-pixel thin diagonal crack

    probs = binary.astype(np.float32) * 0.95
    instances = unet_adapter._extract_instances(probs, binary, 200, 200)

    assert len(instances) == 1
    inst = instances[0]
    assert inst.area_px >= 100
    assert inst.box["x0"] <= 30 and inst.box["x1"] >= 149
    assert inst.box["y0"] <= 30 and inst.box["y1"] >= 149
    assert len(inst.polygon) == 4
    # All polygon coordinates are finite and inside image bounds
    for x, y in inst.polygon:
        assert 0 <= x <= 200
        assert 0 <= y <= 200


def test_mask_to_image_alignment_and_coordinate_transforms(unet_adapter):
    """Test 6: Verify instance bounding box aligns with extracted sub-mask."""
    binary = np.zeros((300, 400), dtype=np.uint8)
    binary[50:120, 80:160] = 1  # 70x80 patch
    probs = binary.astype(np.float32) * 0.8

    instances = unet_adapter._extract_instances(probs, binary, 300, 400)
    assert len(instances) == 1
    inst = instances[0]
    assert inst.box == {"x0": 80, "y0": 50, "x1": 160, "y1": 120}
    assert inst.mask.shape == (70, 80)
    assert np.all(inst.mask == 1)


def test_reproducible_threshold_configuration(unet_adapter):
    """Test 7: Threshold setting strictly determines binary output."""
    img = np.full((320, 320, 3), 100, dtype=np.uint8)
    # Draw high contrast fissure
    img[100:200, 150:155] = 0

    res_low = unet_adapter.segment(img, threshold=0.2, extract_instances=False)
    res_high = unet_adapter.segment(img, threshold=0.8, extract_instances=False)

    # Lower threshold must yield >= cracked pixels than higher threshold
    assert res_low.binary.sum() >= res_high.binary.sum()
