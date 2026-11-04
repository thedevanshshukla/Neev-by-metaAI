"""Manual label format for Tests A/B (JSON, versioned).

A label file looks like this (see eval/example_labels.json):

{
  "version": 1,
  "dataset": "delhi-lane-1",
  "annotator": "name",
  "buildings": [
    {"building_id": "LN-B1",
     "image": "f_0132.jpg",
     "labels": {"open_ground_storey": "yes", "storeys": 4,
                "heavy_overhang": "no", "pounding": "unclear"}}
  ]
}

Rules: labels use yes/no/unclear for checklist ids (docs.md section 7);
"storeys" is a positive int (total floors) or "unclear". Anything the
annotator cannot see must be "unclear", never guessed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neev.validation.schemas import ANSWERS, CHECKLIST_IDS

LABEL_FORMAT_VERSION = 1


def validate_labels(doc: dict) -> list[str]:
    """Return error strings for a label document (empty means valid)."""
    errors: list[str] = []
    if doc.get("version") != LABEL_FORMAT_VERSION:
        errors.append(f"version must be {LABEL_FORMAT_VERSION}")
    buildings = doc.get("buildings")
    if not isinstance(buildings, list) or not buildings:
        errors.append("buildings must be a non-empty list")
        return errors
    seen: set[str] = set()
    for entry in buildings:
        if not isinstance(entry, dict):
            errors.append("building entries must be objects")
            continue
        bid = entry.get("building_id")
        if not bid:
            errors.append("building entry misses building_id")
        elif bid in seen:
            errors.append(f"duplicate building_id: {bid!r}")
        seen.add(bid)
        if not entry.get("image"):
            errors.append(f"{bid}: misses image")
        labels = entry.get("labels")
        if not isinstance(labels, dict):
            errors.append(f"{bid}: labels must be an object")
            continue
        for key, value in labels.items():
            if key == "storeys":
                if value != "unclear" and (
                    not isinstance(value, int) or value < 1
                ):
                    errors.append(f"{bid}: storeys must be a positive int or unclear")
            elif key not in CHECKLIST_IDS:
                errors.append(f"{bid}: unknown label {key!r}")
            elif value not in ANSWERS:
                errors.append(f"{bid}: {key} must be yes/no/unclear, got {value!r}")
    return errors


def load_labels(path: str | Path) -> dict[str, Any]:
    """Load and validate a label file. ValueError lists every problem."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_labels(doc)
    if errors:
        raise ValueError("invalid labels:\n- " + "\n- ".join(errors))
    return doc
