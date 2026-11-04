"""Offline tests for crack coverage diagnostics (synthetic geometry)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neev.eval.coverage import (  # noqa: E402
    GRAPH_THRESHOLD,
    coverage_at,
    overlap_graph,
    pool,
    shared_predictions,
    summarize,
)

CAT = "crack"


def box(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def slivers():
    # One 10x10 pred over three stacked GT slivers: IoU 0.4 each.
    pred = [box(0, 0, 10, 10)]
    gts = [box(0, 0, 10, 4), box(0, 3, 10, 7), box(0, 6, 10, 10)]
    return pred, gts


def test_one_pred_many_slivers_fragmentation():
    preds, gts = slivers()
    out = summarize(preds, ["p0"], gts, ["g0", "g1", "g2"], CAT)
    assert (out["strict"]["tp"], out["strict"]["fp"], out["strict"]["fn"]) == (0, 1, 3)
    assert out["pred_coverage"]["0.1"] == {"covered": 1, "total": 1, "fraction": 1.0}
    assert out["pred_coverage"]["0.5"]["covered"] == 0
    assert out["gt_coverage"]["0.25"] == {"covered": 3, "total": 3, "fraction": 1.0}
    assert out["gt_coverage"]["0.5"]["covered"] == 0
    shared = out["many_to_one"]["shared"]["0.1"]
    assert shared == [{"pred_id": "p0", "gt_ids": ["g0", "g1", "g2"], "n_gt": 3}]
    assert out["graph"]["components"] == 1
    assert out["graph"]["one_pred_multi_gt"] == 1


def test_many_preds_one_gt_duplicates():
    gt = [box(0, 0, 10, 10)]
    preds = [box(0, 0, 10, 10)] * 3
    out = summarize(preds, ["p0", "p1", "p2"], gt, ["g0"], CAT)
    assert (out["strict"]["tp"], out["strict"]["fp"], out["strict"]["fn"]) == (1, 2, 0)
    assert out["graph"]["multi_pred_one_gt"] == 1
    assert out["graph"]["components"] == 1
    assert out["pred_coverage"]["0.5"]["covered"] == 3


def test_no_predictions():
    out = summarize([], [], [box(0, 0, 4, 4)], ["g0"], CAT)
    assert (out["strict"]["tp"], out["strict"]["fp"], out["strict"]["fn"]) == (0, 0, 1)
    assert out["pred_coverage"]["0.1"]["fraction"] is None
    assert out["gt_coverage"]["0.1"]["covered"] == 0
    assert out["graph"]["isolated_gts"] == 1


def test_no_ground_truth_and_empty_negative():
    out = summarize([box(0, 0, 4, 4)], ["p0"], [], [], CAT)
    assert (out["strict"]["tp"], out["strict"]["fp"], out["strict"]["fn"]) == (0, 1, 0)
    assert out["strict"]["precision"] == 0.0
    assert out["gt_coverage"]["0.1"]["fraction"] is None
    assert out["graph"]["isolated_preds"] == 1
    empty = summarize([], [], [], [], CAT)
    assert empty["strict"]["precision"] is None
    assert empty["strict"]["recall"] is None
    assert empty["strict"]["f1"] is None
    assert empty["graph"]["components"] == 0


def test_rotated_boxes_exact_iou():
    diamond = [(5, 0), (10, 5), (5, 10), (0, 5)]
    out = summarize([diamond], ["p0"], [diamond], ["g0"], CAT)
    assert out["strict"]["tp"] == 1
    assert out["pred_coverage"]["0.5"]["fraction"] == 1.0
    far = [(50, 50), (60, 50), (60, 60), (50, 60)]
    out2 = summarize([diamond], ["p0"], [far], ["g0"], CAT)
    assert out2["strict"]["tp"] == 0
    assert out2["gt_coverage"]["0.1"]["covered"] == 0
    assert out2["graph"]["components"] == 2


def test_threshold_boundaries_monotone():
    preds, gts = slivers()
    out = summarize(preds, ["p0"], gts, ["g0", "g1", "g2"], CAT)
    cov = out["gt_coverage"]
    assert cov["0.1"]["covered"] >= cov["0.25"]["covered"] >= cov["0.5"]["covered"]
    # IoU is exactly 0.4: inside 0.1/0.25, outside 0.5.
    assert cov["0.25"]["covered"] == 3 and cov["0.5"]["covered"] == 0


def test_malformed_pred_strict_fp_only():
    out = summarize([None], ["p0"], [box(0, 0, 4, 4)], ["g0"], CAT)
    assert out["strict"]["fp"] == 1 and out["strict"]["malformed"] == 1
    assert out["pred_coverage"]["0.1"]["total"] == 0
    assert out["graph"]["isolated_gts"] == 1


def test_graph_deterministic_and_shared_sorted():
    preds = [box(0, 0, 10, 10), box(20, 20, 30, 30)]
    gts = [box(0, 0, 10, 4), box(0, 6, 10, 10), box(20, 20, 30, 30)]
    first = summarize(preds, ["p0", "p1"], gts, ["g0", "g1", "g2"], CAT)
    second = summarize(preds, ["p0", "p1"], gts, ["g0", "g1", "g2"], CAT)
    assert first["graph"] == second["graph"]
    assert first["graph"]["components"] == 2
    assert GRAPH_THRESHOLD == 0.1
    shared = shared_predictions(["p0", "p1"], ["g0", "g1", "g2"],
                                [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert shared["0.1"] == [{"pred_id": "p0", "gt_ids": ["g0", "g1"], "n_gt": 2}]


def test_pool_aggregates_and_graph_sums():
    a = summarize([box(0, 0, 10, 10)], ["p0"], [box(0, 0, 10, 10)], ["g0"], CAT)
    b = summarize([], [], [box(0, 0, 4, 4)], ["g1"], CAT)
    pooled = pool([a, b])
    assert (pooled["strict"]["tp"], pooled["strict"]["fp"], pooled["strict"]["fn"]) == (1, 0, 1)
    assert pooled["strict"]["precision"] == 1.0
    assert pooled["strict"]["recall"] == 0.5
    assert pooled["gt_coverage"]["0.5"] == {"covered": 1, "total": 2, "fraction": 0.5}
    assert pooled["graph"]["components"] == 2
    assert pooled["graph"]["isolated_gts"] == 1
    assert coverage_at([])["0.1"]["fraction"] is None
    assert overlap_graph([], [], [])["components"] == 0
