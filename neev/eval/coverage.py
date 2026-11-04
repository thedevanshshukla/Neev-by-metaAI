"""Geometric coverage diagnostics for crack evaluation (additive only).

The primary one-to-one matcher (matching.match_rotated at IoU 0.5) is
untouched. Everything here measures geometric overlap with the exact
rotated-polygon IoU; overlap is NOT proof a prediction is correct.
Pure and deterministic: same inputs, same outputs.
"""
from __future__ import annotations

from typing import Any

from neev.eval.matching import match_rotated, prf, rotated_iou

THRESHOLDS: tuple[float, ...] = (0.1, 0.25, 0.5)
GRAPH_THRESHOLD = 0.1

Poly = list[tuple[float, float]]


def iou_matrix(pred_polys: list[Poly], gt_polys: list[Poly]
               ) -> list[list[float]]:
    """Exact rotated-polygon IoU for every (pred, gt) pair."""
    return [[rotated_iou(p, g) for g in gt_polys] for p in pred_polys]


def max_per_row(matrix: list[list[float]], n_cols: int) -> list[float]:
    """Row maxima (0.0 when the other side is empty)."""
    if n_cols == 0:
        return [0.0] * len(matrix)
    return [max(row) for row in matrix]


def coverage_at(max_vals: list[float], thresholds: tuple[float, ...] = THRESHOLDS
                ) -> dict[str, dict[str, Any]]:
    """Covered/total/fraction per threshold; fraction None when total is 0."""
    total = len(max_vals)
    out: dict[str, dict[str, Any]] = {}
    for thr in thresholds:
        covered = sum(1 for v in max_vals if v >= thr)
        out[str(thr)] = {
            "covered": covered,
            "total": total,
            "fraction": (covered / total) if total else None,
        }
    return out


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def overlap_graph(pred_ids: list[str], gt_ids: list[str],
                  matrix: list[list[float]],
                  threshold: float = GRAPH_THRESHOLD) -> dict[str, Any]:
    """Bipartite overlap graph (edge when IoU >= threshold).

    Nodes are P0.. and G0...: connected components group mutually
    overlapping predictions and ground-truth boxes deterministically.
    """
    n_p, n_g = len(pred_ids), len(gt_ids)
    parent = list(range(n_p + n_g))
    edges = 0
    for i in range(n_p):
        for j in range(n_g):
            if matrix[i][j] >= threshold:
                edges += 1
                ri, rj = _find(parent, i), _find(parent, n_p + j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    comps: dict[int, dict[str, list[int]]] = {}
    for i in range(n_p):
        comps.setdefault(_find(parent, i),
                         {"preds": [], "gts": []})["preds"].append(i)
    for j in range(n_g):
        comps.setdefault(_find(parent, n_p + j),
                         {"preds": [], "gts": []})["gts"].append(j)
    ordered = sorted(comps.values(),
                     key=lambda c: (min(c["preds"]) if c["preds"] else n_p,
                                    min(c["gts"]) if c["gts"] else n_g))
    components = [{
        "pred_ids": [pred_ids[i] for i in c["preds"]],
        "gt_ids": [gt_ids[j] for j in c["gts"]],
    } for c in ordered]
    one_to_many = [c for c in components
                   if len(c["pred_ids"]) == 1 and len(c["gt_ids"]) >= 2]
    many_to_one = [c for c in components
                   if len(c["pred_ids"]) >= 2 and len(c["gt_ids"]) == 1]
    return {
        "threshold": threshold,
        "edges": edges,
        "components": len(components),
        "one_pred_multi_gt": len(one_to_many),
        "multi_pred_one_gt": len(many_to_one),
        "isolated_preds": sum(1 for c in components
                              if len(c["pred_ids"]) == 1
                              and not c["gt_ids"]),
        "isolated_gts": sum(1 for c in components
                            if len(c["gt_ids"]) == 1 and not c["pred_ids"]),
        "detail": components,
    }


def shared_predictions(pred_ids: list[str], gt_ids: list[str],
                       matrix: list[list[float]],
                       thresholds: tuple[float, ...] = (0.1, 0.25)
                       ) -> dict[str, list[dict[str, Any]]]:
    """Many-to-one diagnostic: preds overlapping >= 2 GT boxes each.

    Counts are geometric overlap, never TP. Sorted by GT count desc,
    then pred id, for determinism.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for thr in thresholds:
        shared = []
        for i, pid in enumerate(pred_ids):
            hits = sorted(gt_ids[j] for j in range(len(gt_ids))
                          if matrix[i][j] >= thr)
            if len(hits) >= 2:
                shared.append({"pred_id": pid, "gt_ids": hits,
                               "n_gt": len(hits)})
        shared.sort(key=lambda e: (-e["n_gt"], e["pred_id"]))
        out[str(thr)] = shared
    return out


def summarize(pred_polys: list[Poly | None], pred_ids: list[str],
              gt_polys: list[Poly], gt_ids: list[str], category: str,
              thresholds: tuple[float, ...] = THRESHOLDS
              ) -> dict[str, Any]:
    """Strict metrics + coverage diagnostics for one image.

    Malformed preds (None polygon) count as strict FPs and stay out of
    the geometry. GT polygons must be valid (match_rotated enforces it).
    """
    well = [(p, i) for i, p in enumerate(pred_polys) if p is not None]
    malformed = sorted(i for i, p in enumerate(pred_polys) if p is None)
    strict = match_rotated(
        [{"polygon": p, "category": category} for p, _ in well],
        [{"polygon": g, "category": category} for g in gt_polys], 0.5)
    strict_fp = strict.fp + len(malformed)
    matrix = iou_matrix([p for p, _ in well], gt_polys)
    pred_max = max_per_row(matrix, len(gt_polys))
    gt_max = max_per_row([[matrix[i][j] for i in range(len(well))]
                          for j in range(len(gt_polys))], len(well))
    well_ids = [pred_ids[i] for _, i in well]
    return {
        "strict": {"tp": strict.tp, "fp": strict_fp, "fn": strict.fn,
                   **prf(strict.tp, strict_fp, strict.fn),
                   "malformed": len(malformed)},
        "pred_coverage": coverage_at(pred_max, thresholds),
        "gt_coverage": coverage_at(gt_max, thresholds),
        "many_to_one": {
            "gt_covered": coverage_at(gt_max, (0.1, 0.25)),
            "preds_covering": coverage_at(pred_max, (0.1, 0.25)),
            "shared": shared_predictions(well_ids, gt_ids, matrix),
        },
        "graph": overlap_graph(well_ids, gt_ids, matrix),
    }


def pool(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-image summaries (strict sums; coverage re-pooled)."""
    tp = sum(s["strict"]["tp"] for s in summaries)
    fp = sum(s["strict"]["fp"] for s in summaries)
    fn = sum(s["strict"]["fn"] for s in summaries)

    def repooled(key: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for thr in THRESHOLDS:
            covered = sum(s[key][str(thr)]["covered"] for s in summaries)
            total = sum(s[key][str(thr)]["total"] for s in summaries)
            out[str(thr)] = {"covered": covered, "total": total,
                             "fraction": (covered / total) if total else None}
        return out

    graph = {"components": 0, "one_pred_multi_gt": 0,
             "multi_pred_one_gt": 0, "isolated_preds": 0,
             "isolated_gts": 0, "edges": 0}
    for s in summaries:
        for k in graph:
            graph[k] += s["graph"][k]
    return {
        "strict": {"tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn)},
        "pred_coverage": repooled("pred_coverage"),
        "gt_coverage": repooled("gt_coverage"),
        "graph": graph,
    }
