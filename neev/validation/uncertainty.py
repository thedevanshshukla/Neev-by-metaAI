"""Uncertainty handling: cross-checks, Re-film rule, scoring (docs.md 8).

Unclear answers add zero points and are always listed, so a missing view
never looks like a safe building. Weights are the demo starting values
from docs.md, not code values; they need engineer review before advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

WEIGHTS = {
    "open_ground_storey": 40,
    "heavy_overhang": 10,
    "short_column": 10,
    "vertical_irregularity": 10,
    "pounding": 10,
    "apparent_quality": 5,
    "falling_hazards": 5,
}
# construction_type is descriptive; it does not move the score.

BAND_MEDIUM_AT = 25
BAND_HIGH_AT = 50
REFILM_UNCLEAR_COUNT = 3


@dataclass(frozen=True)
class Score:
    score: int
    band: str  # low | medium | high | re-film
    refilm: bool
    unclear: list[str] = field(default_factory=list)


def storey_points(total_storeys: int) -> int:
    """G+4 or more: 10. G+2/G+3: 5. Fewer: 0. total = ground + upper."""
    if total_storeys >= 5:
        return 10
    if total_storeys >= 3:
        return 5
    return 0


def storey_crosscheck(measured: int, judge: int | None) -> int | None:
    """Docs.md rule: judge/measured disagreement -> storeys unclear (None)."""
    if judge is None or judge != measured:
        return None
    return measured


def needs_refilm(answers: dict[str, str]) -> bool:
    """Re-film when the soft storey is unknown or views are mostly missing."""
    from neev.validation.schemas import CHECKLIST_IDS

    full = {cid: answers.get(cid, "unclear") for cid in CHECKLIST_IDS}
    unclear = [cid for cid, ans in full.items() if ans != "yes" and ans != "no"]
    return full["open_ground_storey"] == "unclear" or len(unclear) >= REFILM_UNCLEAR_COUNT


def score_building(answers: dict[str, str], storeys_total: int | None) -> Score:
    """Score yes-only points; unclear and no add zero. No colour on re-film."""
    from neev.validation.schemas import CHECKLIST_IDS

    full = {cid: answers.get(cid, "unclear") for cid in CHECKLIST_IDS}
    unclear = [cid for cid, ans in full.items() if ans not in ("yes", "no")]
    points = sum(WEIGHTS[cid] for cid, ans in full.items() if ans == "yes")
    if storeys_total is not None:
        points += storey_points(storeys_total)
    refilm = needs_refilm(full)
    if refilm:
        band = "re-film"
    elif points >= BAND_HIGH_AT:
        band = "high"
    elif points >= BAND_MEDIUM_AT:
        band = "medium"
    else:
        band = "low"
    return Score(score=points, band=band, refilm=refilm, unclear=unclear)
