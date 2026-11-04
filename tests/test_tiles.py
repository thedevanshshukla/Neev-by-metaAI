"""Offline tests for tile geometry and cross-tile merging (synthetic)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neev.eval.tiles import (  # noqa: E402
    aabb_iou,
    clip_box,
    compute_tiles,
    merge_cross_tile,
    translate_box,
)


def _covers(tiles, width, height):
    pts = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1),
           (width // 2, height // 2)]
    for x, y in pts:
        assert any(t.x0 <= x < t.x1 and t.y0 <= y < t.y1 for t in tiles), (x, y)


def test_tiles_cover_dm1_and_k1_in_bounds():
    for width, height in [(2400, 1792), (2752, 1536)]:
        tiles = compute_tiles(width, height)
        assert len(tiles) == 4
        assert len({t.tile_id for t in tiles}) == 4
        for t in tiles:
            assert 0 <= t.x0 < t.x1 <= width
            assert 0 <= t.y0 < t.y1 <= height
        _covers(tiles, width, height)


def test_tiles_cover_odd_dimensions():
    tiles = compute_tiles(515, 388)
    assert len(tiles) == 4
    for t in tiles:
        assert 0 <= t.x0 < t.x1 <= 515
        assert 0 <= t.y0 < t.y1 <= 388
    _covers(tiles, 515, 388)


def test_tile_overlap_near_quarter():
    for width, height in [(2400, 1792), (2752, 1536)]:
        tiles = compute_tiles(width, height)
        by_id = {t.tile_id: t for t in tiles}
        left, right = by_id["r0c0"], by_id["r0c1"]
        top, bottom = by_id["r0c0"], by_id["r1c0"]
        assert 0.20 <= (left.x1 - right.x0) / left.width <= 0.30
        assert 0.20 <= (top.y1 - bottom.y0) / top.height <= 0.30


def test_translate_box_offsets_corners():
    out = translate_box({"x0": 1, "y0": 2, "x1": 3, "y1": 4,
                         "tile_id": "r1c0"}, 10, 20)
    assert (out["x0"], out["y0"], out["x1"], out["y1"]) == (11, 22, 13, 24)
    assert out["tile_id"] == "r1c0"


def test_clip_box_cases():
    assert clip_box({"x0": 1, "y0": 1, "x1": 3, "y1": 3}, 10, 10) == {
        "x0": 1, "y0": 1, "x1": 3, "y1": 3}
    assert clip_box({"x0": -5, "y0": 2, "x1": 4, "y1": 20}, 10, 10) == {
        "x0": 0, "y0": 2, "x1": 4, "y1": 10}
    assert clip_box({"x0": 20, "y0": 20, "x1": 30, "y1": 30}, 10, 10) is None
    assert clip_box({"x0": 5, "y0": 5, "x1": 5, "y1": 9}, 10, 10) is None


def test_merge_suppresses_duplicates_largest_survives():
    big = {"x0": 0, "y0": 0, "x1": 10, "y1": 10, "tile_id": "r0c0"}
    small = {"x0": 1, "y0": 1, "x1": 9, "y1": 9, "tile_id": "r0c1"}
    far = {"x0": 50, "y0": 50, "x1": 60, "y1": 60, "tile_id": "r1c1"}
    assert aabb_iou(big, small) >= 0.5
    kept, suppressed = merge_cross_tile([small, far, big])
    assert kept == [big, far]
    assert suppressed == [small]


def test_merge_deterministic_tie_break():
    a = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
    b = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
    assert merge_cross_tile([a, b])[0] == [a]
    assert merge_cross_tile([b, a])[0] == [b]


def test_merge_keeps_disjoint_and_empty():
    assert merge_cross_tile([]) == ([], [])
    a = {"x0": 0, "y0": 0, "x1": 2, "y1": 2}
    b = {"x0": 5, "y0": 5, "x1": 7, "y1": 7}
    kept, suppressed = merge_cross_tile([a, b])
    assert kept == [a, b] and suppressed == []
