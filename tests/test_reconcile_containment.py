import pytest
from neev.multiview import reconcile
from neev.eval.tiles import aabb_iou, aabb_iomin, merge_cross_tile

def make_det(phrase, box, view, tile_box, frame_w, frame_h, mask_payload=""):
    return {
        "phrase": phrase,
        "box": box,
        "view": view,
        "tile_box": tile_box,
        "frame_width": frame_w,
        "frame_height": frame_h,
        "mask_payload": mask_payload,
        "mask_width": frame_w,
        "mask_height": frame_h,
    }

def test_iomin_calculation():
    # Box A: 100x100 (area 10,000) at (0,0) to (100,100)
    # Box B: 50x50 (area 2,500) inside Box A at (10,10) to (60,60)
    box_a = {"x0": 0, "y0": 0, "x1": 100, "y1": 100}
    box_b = {"x0": 10, "y0": 10, "x1": 60, "y1": 60}
    # Intersection = 2,500. Union = 10,000.
    assert aabb_iou(box_a, box_b) == 2500 / 10000 == 0.25
    assert aabb_iomin(box_a, box_b) == 2500 / 2500 == 1.0


def test_adjacent_tile_fragments_suppressed():
    """Test 1: Adjacent-tile fragment like mv1 (r0c0) vs mv14 (r1c1)."""
    # Large crack spanning y=0 to y=925
    det_large = make_det("crack", {"x0": 1208, "y0": 0, "x1": 1312, "y1": 925},
                         "r0c0", {"x0": 0, "y0": 0, "x1": 1372, "y1": 1024}, 1372, 1024)
    # Small stub in overlap tile r1c1 (tile_box x0=1028, y0=768) at source y=768 to 927
    det_stub = make_det("crack", {"x0": 188, "y0": 0, "x1": 286, "y1": 159},
                        "r1c1", {"x0": 1028, "y0": 768, "x1": 2400, "y1": 1792}, 1372, 1024)

    # Standard merge (no iomin): both survive (IoU ~0.156 < 0.50)
    kept_std, supp_std, _ = reconcile.reconcile([det_large, det_stub], 2400, 1792)
    assert len(kept_std) == 2
    assert len(supp_std) == 0

    # Containment-aware merge (iomin=0.85): stub is suppressed
    kept_exp, supp_exp, _ = reconcile.reconcile([det_large, det_stub], 2400, 1792, iomin_threshold=0.85)
    assert len(kept_exp) == 1
    assert len(supp_exp) == 1
    assert supp_exp[0]["reason"] == "duplicate-containment"
    assert supp_exp[0]["iou"] >= 0.95


def test_distinct_nearby_cracks():
    """Test 2: Distinct nearby cracks side by side."""
    det1 = make_det("crack", {"x0": 100, "y0": 100, "x1": 120, "y1": 300},
                    "r0c0", {"x0": 0, "y0": 0, "x1": 500, "y1": 500}, 500, 500)
    det2 = make_det("crack", {"x0": 130, "y0": 100, "x1": 150, "y1": 300},
                    "r0c0", {"x0": 0, "y0": 0, "x1": 500, "y1": 500}, 500, 500)

    kept, supp, _ = reconcile.reconcile([det1, det2], 1000, 1000, iomin_threshold=0.85)
    assert len(kept) == 2
    assert len(supp) == 0


def test_short_crack_near_longer_crack():
    """Test 3: A short branch crack partially overlapping a longer crack."""
    # Main crack: 50x500 at x=200..250, y=100..600 (area 25,000)
    # Branch crack: 100x50 at x=220..320, y=300..350 (area 5,000)
    # Intersection: 30x50 = 1,500. IoMin = 1500 / 5000 = 0.30 (< 0.85)
    det_main = make_det("crack", {"x0": 200, "y0": 100, "x1": 250, "y1": 600},
                        "r0c0", {"x0": 0, "y0": 0, "x1": 500, "y1": 700}, 500, 700)
    det_branch = make_det("crack", {"x0": 220, "y0": 300, "x1": 320, "y1": 350},
                          "r0c0", {"x0": 0, "y0": 0, "x1": 500, "y1": 700}, 500, 700)

    kept, supp, _ = reconcile.reconcile([det_main, det_branch], 1000, 1000, iomin_threshold=0.85)
    assert len(kept) == 2  # Branch crack is preserved!
    assert len(supp) == 0


def test_detections_from_non_overlapping_tiles():
    """Test 4: Detections from non-overlapping tiles in opposite corners."""
    # Tile top-left
    det_tl = make_det("crack", {"x0": 50, "y0": 50, "x1": 100, "y1": 100},
                      "r0c0", {"x0": 0, "y0": 0, "x1": 500, "y1": 500}, 500, 500)
    # Tile bottom-right
    det_br = make_det("crack", {"x0": 50, "y0": 50, "x1": 100, "y1": 100},
                      "r1c1", {"x0": 500, "y0": 500, "x1": 1000, "y1": 1000}, 500, 500)

    kept, supp, _ = reconcile.reconcile([det_tl, det_br], 1000, 1000, iomin_threshold=0.85)
    assert len(kept) == 2
    assert len(supp) == 0


def test_different_masks_with_similar_boxes():
    """Test 5: Different masks with identical bounding boxes."""
    det1 = make_det("crack", {"x0": 100, "y0": 100, "x1": 200, "y1": 200},
                    "r0c0", {"x0": 0, "y0": 0, "x1": 500, "y1": 500}, 500, 500,
                    mask_payload="mask_a_bytes")
    det2 = make_det("crack", {"x0": 100, "y0": 100, "x1": 200, "y1": 200},
                    "r0c1", {"x0": 0, "y0": 0, "x1": 500, "y1": 500}, 500, 500,
                    mask_payload="mask_b_bytes")

    kept, supp, _ = reconcile.reconcile([det1, det2], 1000, 1000, iomin_threshold=0.85)
    assert len(kept) == 1
    assert len(supp) == 1
    assert supp[0]["reason"] == "exact-duplicate"


def test_identical_predictions_from_overlapping_views():
    """Test 6: Identical predictions from overlapping views in source coordinates."""
    # Two overlapping tiles view the exact same physical feature at source (300, 300, 400, 400)
    # View 1: tile x0=0, y0=0 -> local box (300, 300, 400, 400)
    det_v1 = make_det("crack", {"x0": 300, "y0": 300, "x1": 400, "y1": 400},
                      "r0c0", {"x0": 0, "y0": 0, "x1": 600, "y1": 600}, 600, 600)
    # View 2: tile x0=200, y0=200 -> local box (100, 100, 200, 200)
    det_v2 = make_det("crack", {"x0": 100, "y0": 100, "x1": 200, "y1": 200},
                      "r0c1", {"x0": 200, "y0": 200, "x1": 800, "y1": 800}, 600, 600)

    kept, supp, _ = reconcile.reconcile([det_v1, det_v2], 1000, 1000, iomin_threshold=0.85)
    assert len(kept) == 1
    assert len(supp) == 1
    assert supp[0]["reason"] == "exact-duplicate"
    assert supp[0]["iou"] == 1.0
