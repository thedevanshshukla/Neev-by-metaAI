"""Checklist and building-output validation (standard library only).

Rules from docs.md: every yes/no needs an evidence frame from best_frames
plus a reason; unclear carries no evidence and no confidence. Anything
else is a validation error, never silently accepted.
"""

from __future__ import annotations

# Checklist item ids from docs.md sections 7 and 10.
CHECKLIST_IDS = (
    "open_ground_storey",
    "heavy_overhang",
    "pounding",
    "falling_hazards",
    "short_column",
    "vertical_irregularity",
    "apparent_quality",
    "construction_type",
)

ANSWERS = ("yes", "no", "unclear")


def validate_item(item: dict, known_frames: set[str]) -> list[str]:
    """Return error strings for one checklist item (empty means valid)."""
    errors: list[str] = []
    item_id = item.get("id")
    answer = item.get("answer")
    if item_id not in CHECKLIST_IDS:
        errors.append(f"unknown checklist id: {item_id!r}")
    if answer not in ANSWERS:
        errors.append(f"item {item_id!r}: answer must be yes/no/unclear, got {answer!r}")
        return errors
    evidence, reason = item.get("evidence"), item.get("reason")
    confidence = item.get("confidence")
    if answer in ("yes", "no"):
        if evidence not in known_frames:
            errors.append(
                f"item {item_id!r}: evidence {evidence!r} is not a best frame"
            )
        if not reason or not str(reason).strip():
            errors.append(f"item {item_id!r}: {answer} needs a reason")
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            errors.append(f"item {item_id!r}: confidence {confidence!r} not in [0, 1]")
    else:  # unclear
        if evidence is not None:
            errors.append(f"item {item_id!r}: unclear must not cite evidence")
        if confidence is not None:
            errors.append(f"item {item_id!r}: unclear must not carry confidence")
        if not reason or not str(reason).strip():
            errors.append(f"item {item_id!r}: unclear needs a reason")
    return errors


def validate_building(doc: dict) -> list[str]:
    """Return error strings for one building.json-style document."""
    errors: list[str] = []
    if not doc.get("building_id"):
        errors.append("missing building_id")
    frames = doc.get("best_frames")
    if not isinstance(frames, list) or not frames:
        errors.append("best_frames must be a non-empty list")
        frames = []
    known = set(frames)
    storeys = doc.get("storeys", {})
    measured, judge = storeys.get("measured"), storeys.get("judge")
    if not isinstance(measured, int) or measured < 1:
        errors.append(f"storeys.measured must be a positive int, got {measured!r}")
    if judge is not None and (not isinstance(judge, int) or judge < 1):
        errors.append(f"storeys.judge must be a positive int or null, got {judge!r}")
    checklist = doc.get("checklist")
    if not isinstance(checklist, list) or not checklist:
        errors.append("checklist must be a non-empty list")
        return errors
    seen: set[str] = set()
    for item in checklist:
        if not isinstance(item, dict):
            errors.append("checklist items must be objects")
            continue
        if item.get("id") in seen:
            errors.append(f"duplicate checklist id: {item.get('id')!r}")
        seen.add(item.get("id"))
        errors.extend(validate_item(item, known))
    return errors


def missing_ids(doc: dict) -> list[str]:
    """Checklist ids with no entry. Missing counts as unclear when scoring."""
    present = {
        item.get("id") for item in doc.get("checklist", []) if isinstance(item, dict)
    }
    return [cid for cid in CHECKLIST_IDS if cid not in present]
