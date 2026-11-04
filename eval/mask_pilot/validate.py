"""Validate a crack mask-pilot artifact against ANNOTATION_SPEC.md.

Offline only: reads the JSON artifact + source dims (Pillow), prints a
report, never modifies the artifact. Exit 0 when valid (warnings
allowed), 1 on any error.

Usage:
  python eval/mask_pilot/validate.py --artifact eval/mask_pilot/gt_masks.v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from neev.eval.mask_eval import rasterize_polygons  # noqa: E402
from PIL import Image  # noqa: E402

EXPECTED = {
    "cr4": {"task_id": 7, "relpath": "dataset/real_world/cracks/cr4.jpeg"},
    "cr5": {"task_id": 8, "relpath": "dataset/real_world/cracks/cr5.jpeg"},
    "dm1": {"task_id": 2, "relpath": "dataset/real_world/confusing_category/dm1.png"},
    "k1": {"task_id": 13, "relpath": "dataset/real_world/Difficult_Visibility/k1.png"},
    "h2": {"task_id": 10, "relpath": "dataset/real_world/Crack_Detection_Edge_Cases/h2.png"},
    "i1": {"task_id": 21, "relpath": "dataset/real_world/Multiple_Defects_in_One_Building/i1.png"},
    "in1": {"task_id": 17, "relpath": "dataset/real_world/intact/in1.png"},
}


def _orientation(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _proper_cross(p1, p2, p3, p4) -> bool:
    """True when open segments p1p2 and p3p4 cross (shared endpoints excluded)."""
    if len({tuple(p1), tuple(p2), tuple(p3), tuple(p4)}) < 4:
        return False
    d1 = _orientation(p3, p4, p1)
    d2 = _orientation(p3, p4, p2)
    d3 = _orientation(p1, p2, p3)
    d4 = _orientation(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0) or (d1 < 0) != (d2 < 0)) and (
        (d3 > 0) != (d4 > 0) or (d3 < 0) != (d4 < 0))


def ring_self_intersects(ring) -> bool:
    """True when any two non-adjacent edges of the closed ring cross."""
    n = len(ring)
    if n < 4:
        return False
    edges = [(ring[i], ring[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if len({i, (i + 1) % n} & {j, (j + 1) % n}):
                continue  # adjacent edges share a vertex by construction
            if _proper_cross(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                return True
    return False


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def validate_artifact(artifact: dict, actual_dims: dict[str, tuple[int, int]]
                      ) -> tuple[list[str], list[str]]:
    """Spec checks against measured dims. Returns (errors, warnings)."""
    errors, warnings = [], []
    if not isinstance(artifact, dict):
        return ["artifact must be a JSON object"], []
    if artifact.get("version") != 1:
        errors.append("version must be 1")
    images = artifact.get("images")
    if not isinstance(images, list):
        return errors + ["images must be a list"], warnings
    seen: set[str] = set()
    for entry in images:
        if not isinstance(entry, dict):
            errors.append("image entry must be an object")
            continue
        iid = entry.get("image_id")
        if iid in seen:
            errors.append(f"{iid}: duplicate image entry")
        seen.add(iid)
        if iid not in EXPECTED:
            errors.append(f"{iid}: unexpected image_id")
            continue
        exp = EXPECTED[iid]
        if entry.get("task_id") != exp["task_id"]:
            errors.append(f"{iid}: task_id {entry.get('task_id')} != "
                          f"expected {exp['task_id']}")
        w, h = entry.get("width"), entry.get("height")
        aw, ah = actual_dims.get(iid, (None, None))
        if (w, h) != (aw, ah):
            errors.append(f"{iid}: dims {(w, h)} != actual {(aw, ah)}")
            continue
        if not isinstance(entry.get("negative"), bool):
            errors.append(f"{iid}: negative must be boolean")
        instances = entry.get("instances")
        if not isinstance(instances, list):
            errors.append(f"{iid}: instances must be a list")
            continue
        if entry.get("negative") and instances:
            errors.append(f"{iid}: negative image must have zero instances")
        if not entry.get("negative") and not instances:
            warnings.append(f"{iid}: non-negative image with zero instances")
        ids: set[str] = set()
        for inst in instances:
            if not isinstance(inst, dict):
                errors.append(f"{iid}: instance must be an object")
                continue
            nid = inst.get("instance_id")
            if not isinstance(nid, str) or not nid:
                errors.append(f"{iid}: instance_id must be a non-empty string")
                continue
            if nid in ids:
                errors.append(f"{iid}: duplicate instance_id {nid}")
            ids.add(nid)
            if not isinstance(inst.get("thin"), bool):
                errors.append(f"{iid}/{nid}: thin must be boolean")
            polys = inst.get("polygons")
            if not isinstance(polys, list) or not polys:
                errors.append(f"{iid}/{nid}: polygons must be a non-empty list")
                continue
            bad = False
            for ring in polys:
                if not isinstance(ring, list) or len(ring) < 3:
                    errors.append(f"{iid}/{nid}: ring needs >= 3 points")
                    bad = True
                    break
                for pt in ring:
                    if (not isinstance(pt, (list, tuple)) or len(pt) != 2
                            or not all(_is_num(v) for v in pt)):
                        errors.append(f"{iid}/{nid}: bad point {pt!r}")
                        bad = True
                        break
                    if not (0 <= pt[0] <= w and 0 <= pt[1] <= h):
                        errors.append(f"{iid}/{nid}: point {pt!r} outside "
                                      f"{w}x{h}")
                        bad = True
                        break
                if bad:
                    break
                if ring_self_intersects(ring):
                    errors.append(f"{iid}/{nid}: self-intersecting ring")
                    bad = True
                    break
            if bad:
                continue
            painted = rasterize_polygons(polys, w, h)
            if painted is None or not any(painted):
                if inst.get("thin") is True:
                    warnings.append(f"{iid}/{nid}: thin instance paints zero "
                                    f"pixels; excluded from scoring")
                else:
                    errors.append(f"{iid}/{nid}: paints zero pixels "
                                  f"(widen it or set thin:true)")
    for iid in sorted(set(EXPECTED) - seen):
        errors.append(f"{iid}: missing from artifact")
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default="eval/mask_pilot/gt_masks.v1.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root)
    try:
        artifact = json.loads((root / args.artifact).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot load artifact: {exc}")
        return 1
    actual_dims: dict[str, tuple[int, int]] = {}
    for iid, exp in EXPECTED.items():
        try:
            with Image.open(root / exp["relpath"]) as im:
                actual_dims[iid] = im.size
        except OSError as exc:
            print(f"ERROR: cannot open {exp['relpath']}: {exc}")
            return 1
    errors, warnings = validate_artifact(artifact, actual_dims)
    n_inst = sum(len(e.get("instances", []))
                 for e in artifact.get("images", [])
                 if isinstance(e, dict))
    print(f"images: {len(EXPECTED)} expected, "
          f"{len(artifact.get('images', []))} present, "
          f"{n_inst} instances")
    for wmsg in warnings:
        print(f"WARNING: {wmsg}")
    for emsg in errors:
        print(f"ERROR: {emsg}")
    if errors:
        print(f"INVALID: {len(errors)} error(s)")
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
