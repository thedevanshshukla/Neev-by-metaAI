"""Metrics with explicit unclear handling (docs.md section 11).

Docs rule: count unclear answers separately and report recall with and
without them. Every metric returns raw counts beside the rate so a
report can be rebuilt from numbers alone. Rates are None when their
denominator is zero: never 0.0 by accident, never fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass


def _rate(hits: int, total: int) -> float | None:
    return hits / total if total else None


@dataclass(frozen=True)
class BinaryResult:
    tp: int
    fp: int
    fn: int
    tn: int
    unclear: int
    n: int
    precision: float | None
    recall: float | None
    unclear_rate: float | None


def binary_pr(
    pairs: list[tuple[str, str]], unclear_mode: str = "exclude"
) -> BinaryResult:
    """Precision/recall over (predicted, actual) yes/no/unclear pairs.

    unclear_mode="exclude": unclear predictions leave P/R out but are
    counted in unclear_rate (docs.md Test A default).
    unclear_mode="as_error": unclear predictions count as wrong
    (a predicted-unclear on an actual-yes is a miss, on actual-no a
    false alarm), for the "recall with unclear" column.
    """
    if unclear_mode not in ("exclude", "as_error"):
        raise ValueError("unclear_mode must be exclude or as_error")
    tp = fp = fn = tn = unclear = 0
    for pred, actual in pairs:
        if pred == "unclear":
            unclear += 1
            if unclear_mode == "as_error":
                if actual == "yes":
                    fn += 1
                elif actual == "no":
                    fp += 1
            continue
        if actual == "unclear":
            continue  # unlabelled rows cannot judge the model
        if pred == "yes" and actual == "yes":
            tp += 1
        elif pred == "yes":
            fp += 1
        elif pred == "no" and actual == "no":
            tn += 1
        elif pred == "no":
            fn += 1
        else:
            raise ValueError(f"bad prediction: {pred!r}")
    return BinaryResult(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        unclear=unclear,
        n=len(pairs),
        precision=_rate(tp, tp + fp),
        recall=_rate(tp, tp + fn),
        unclear_rate=_rate(unclear, len(pairs)),
    )


@dataclass(frozen=True)
class SimpleRate:
    matches: int
    n: int
    rate: float | None


def agreement(pairs: list[tuple[str, str]]) -> SimpleRate:
    """Exact agreement over (predicted, actual) answers (Test B items)."""
    matches = sum(1 for pred, actual in pairs if pred == actual)
    return SimpleRate(matches=matches, n=len(pairs), rate=_rate(matches, len(pairs)))


def exact_match_rate(pairs: list[tuple]) -> SimpleRate:
    """Exact matches for storey counts and other discrete values."""
    matches = sum(1 for pred, actual in pairs if pred == actual)
    return SimpleRate(matches=matches, n=len(pairs), rate=_rate(matches, len(pairs)))


def consistency(run_a: list[str], run_b: list[str]) -> SimpleRate:
    """Test C: how often two screenings of the same buildings match."""
    if len(run_a) != len(run_b):
        raise ValueError("runs must cover the same buildings in order")
    return agreement(list(zip(run_a, run_b)))
