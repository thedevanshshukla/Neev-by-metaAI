"""Offline tests for mask-instance evaluation (synthetic rasters)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neev.eval.mask_eval import (  # noqa: E402
    MaskInstance,
    evaluate_image,
    match_masks,
    pairwise_iou,
    polygons_to_instance,
    pool_scores,
    rasterize_polygons,
)


def raster(pixels, width=4, height=4):
    grid = bytearray(width * height)
    for x, y in pixels:
        grid[y * width + x] = 1
    return bytes(grid)


def inst(iid, pixels, width=4, height=4, category="crack"):
    return MaskInstance(image_id="t", instance_id=iid, width=width,
                        height=height, raster=raster(pixels, width, height),
                        category=category)


A = [(0, 0), (1, 0), (0, 1), (1, 1)]  # 2x2, area 4
C = [(2, 2), (3, 2), (2, 3), (3, 3)]  # disjoint 2x2
D = [(1, 1), (2, 1), (1, 2), (2, 2)]  # overlaps A in 1 px -> IoU 1/7


def test_identical_disjoint_partial_iou():
    assert pairwise_iou([inst("p", A)], [inst("g", A)]) == [[1.0]]
    assert pairwise_iou([inst("p", A)], [inst("g", C)]) == [[0.0]]
    assert pairwise_iou([inst("p", A)], [inst("g", D)]) == [[pytest.approx(1 / 7)]]


def test_matching_duplicates_and_multi_gt():
    m = match_masks([inst("p0", A), inst("p1", A)], [inst("g0", A)])
    assert (m.tp, m.fp, m.fn) == (1, 1, 0)
    assert m.matches[0][:2] == (0, 0)
    m2 = match_masks([inst("p0", A)], [inst("g0", A), inst("g1", C)])
    assert (m2.tp, m2.fp, m2.fn) == (1, 0, 1)


def test_threshold_selects_matches():
    m = match_masks([inst("p0", A)], [inst("g0", D)], iou_threshold=0.5)
    assert (m.tp, m.fp, m.fn) == (0, 1, 1)
    m2 = match_masks([inst("p0", A)], [inst("g0", D)], iou_threshold=0.1)
    assert (m2.tp, m2.fp, m2.fn) == (1, 0, 0)


def test_category_constrained():
    other = inst("g0", A, category="spalling")
    assert pairwise_iou([inst("p0", A)], [other]) == [[0.0]]
    m = match_masks([inst("p0", A)], [other])
    assert (m.tp, m.fp, m.fn) == (0, 1, 1)


def test_empty_preds_gt_and_negatives():
    e = evaluate_image("neg", [], [])
    assert (e.tp, e.fp, e.fn) == (0, 0, 0)
    assert e.precision is None and e.recall is None and e.f1 is None
    miss = evaluate_image("t", [], [inst("g0", A)])
    assert (miss.tp, miss.fp, miss.fn) == (0, 0, 1)
    false = evaluate_image("neg", [inst("p0", A)], [])
    assert (false.tp, false.fp, false.fn) == (0, 1, 0)
    assert false.precision == 0.0 and false.recall is None


def test_malformed_pred_is_fp_gt_raises():
    m = match_masks([None, inst("p1", A)], [inst("g0", A)])
    assert (m.tp, m.fp, m.fn) == (1, 1, 0)
    assert m.malformed_pred_indices == [0]
    with pytest.raises(ValueError):
        match_masks([inst("p0", A)], [None])


def test_dim_mismatch_raises():
    with pytest.raises(ValueError):
        pairwise_iou([inst("p", A)], [inst("g", A, width=5, height=5)])
    with pytest.raises(ValueError):
        MaskInstance(image_id="t", instance_id="x", width=4, height=4,
                     raster=b"\x01\x01")
    with pytest.raises(ValueError):
        MaskInstance(image_id="t", instance_id="x", width=2, height=2,
                     raster=b"\x01\x02\x00\x01")


def test_deterministic_tie_break():
    gts = [inst("g0", A), inst("g1", A)]
    m = match_masks([inst("p0", A)], gts)
    assert m.matches[0][:2] == (0, 0)  # lower gt index wins ties
    again = match_masks([inst("p0", A)], gts)
    assert m.matches == again.matches


def test_rasterize_square_and_triangle():
    sq = rasterize_polygons([[(1, 1), (4, 1), (4, 4), (1, 4)]], 6, 6)
    assert sum(sq) == 9
    assert sq[1 * 6 + 1] == 1 and sq[0] == 0 and sq[5 * 6 + 5] == 0
    tri = rasterize_polygons([[(0, 0), (4, 0), (0, 4)]], 5, 5)
    assert sum(tri) == 6
    assert tri[0 * 5 + 2] == 1 and tri[2 * 5 + 1] == 0


def test_rasterize_clips_and_rejects_bad_rings():
    big = rasterize_polygons([[(-5, -5), (5, -5), (5, 5), (-5, 5)]], 4, 4)
    assert sum(big) == 16
    assert rasterize_polygons([[(0, 0), (1, 1)]], 4, 4) is None
    assert rasterize_polygons([[(0, 0), (1, "x"), (2, 2)]], 4, 4) is None
    assert rasterize_polygons([[(0, 0), (2, 0), (4, 0)]], 4, 4) == bytes(16)
    assert polygons_to_instance("t", "g", 4, 4,
                                [[(0, 0), (2, 0), (4, 0)]]) is None
    ok = polygons_to_instance("t", "g", 6, 6, [[(1, 1), (4, 1), (4, 4), (1, 4)]])
    assert ok is not None and ok.area == 9


def test_pool_scores_micro_average():
    s1 = evaluate_image("a", [inst("p", A)], [inst("g", A)])
    s2 = evaluate_image("b", [], [inst("g", A)])
    pooled = pool_scores([s1, s2])
    assert (pooled["tp"], pooled["fp"], pooled["fn"]) == (1, 0, 1)
    assert pooled["precision"] == 1.0 and pooled["recall"] == 0.5
    assert pooled["images"] == 2
    assert s1.matches[0]["iou"] == 1.0 and s1.matches[0]["gt_id"] == "g"
