"""Transparent screening-priority rules (deterministic, documented).

Levels (screening urgency only - never structural severity or safety):
- needs_closer_inspection: look again, preferably closer/clearer.
- review: context finding worth a glance during review.
- informational: scene context, no action implied.

Inputs are only signals the pipeline truly produces: mapped category,
box area fraction, image-quality warnings, mask presence. Every outcome
carries human-readable reasons. Unknown categories stay review-level
with an explicit reason instead of a guessed severity.
"""
from __future__ import annotations

# Categories that must always be re-examined up close.
ALWAYS_INSPECT = {"crack", "rebar", "spalling"}
# Scene/context categories: useful surroundings, routine glance.
CONTEXT = {"building", "window", "column", "balcony", "floor_slab",
           "car", "shutter", "rooftop_unit", "hoarding", "parapet",
           "open_ground_floor"}
TINY_AREA_FRAC = 0.0005  # below this, localization is unreliable


def prioritize(category: str | None, area_frac: float | None,
               quality_warnings: list[dict],
               has_mask: bool) -> dict:
    """Return {level, reasons, next_action}. Pure function of inputs."""
    reasons: list[str] = []
    if quality_warnings:
        names = sorted({w.get("check", "?") for w in quality_warnings})
        reasons.append("image quality warnings: " + ", ".join(names))
    if category in ALWAYS_INSPECT:
        reasons.append(f"'{category}' is a potential defect class")
    elif category is None:
        reasons.append("unmapped prompt: category unknown, kept for review")
    if area_frac is not None and area_frac < TINY_AREA_FRAC:
        reasons.append("detection is very small; position is approximate")
    if not has_mask:
        reasons.append("box only, no segmentation mask decoded")
    if quality_warnings or category in ALWAYS_INSPECT:
        level = "needs_closer_inspection"
        action = ("Re-photograph the marked region closer and sharper, then "
                  "have a qualified person decide whether a site visit is "
                  "needed.")
    elif category is None:
        level = "review"
        action = "Confirm what the marked region shows; map the prompt if recurring."
    else:
        level = "review" if category not in CONTEXT else "informational"
        if category in CONTEXT:
            reasons.append(f"'{category}' is scene context")
            action = "No action implied; context for nearby defect findings."
        else:
            action = "Glance during review; no defect implied."
    return {"level": level, "reasons": reasons, "next_action": action}
