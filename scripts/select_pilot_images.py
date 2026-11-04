"""Select 30 representative CUBIT-Seg pilot images across 6 target categories:
1. Negative images (clean intact facades, 0 GT cracks)
2. Thin / Hairline cracks (w <= 3.5 px)
3. Medium / Wide cracks (w > 5.0 px)
4. Complex cracks (multi-instance networks, branching)
5. Shadowed surfaces (concrete with cast shadows > 20%)
6. Difficult concrete textures (rough aggregate / porous concrete)

Preserves frozen TEST split by drawing exclusively from DEV (264) and VAL (373) partitions.
"""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MANIFEST_PATH = ROOT / "CUBIT_DATASET_MANIFEST.json"
OUT_PATH = ROOT / "data" / "cubit_sam_pilot_manifest.json"

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

dev_val_images = [img for img in manifest["images"] if img["split"] in ("dev", "val")]
img_map = {img["image_id"]: img for img in dev_val_images}

from neev.cv.enhancement import compute_shadow_mask

# 1. Negative Images (all 4 in dev & val)
neg_ids = [img["image_id"] for img in dev_val_images if not img["has_crack"]]
assert len(neg_ids) == 4, f"Expected 4 negative images, got {len(neg_ids)}"

# 2. Thin / Hairline Cracks (has_crack=True, mean_width <= 3.5 px)
thin_candidates = [
    img for img in dev_val_images
    if img["has_crack"] and img.get("mean_crack_width_px", 99) <= 3.5
]
thin_candidates.sort(key=lambda x: x["mean_crack_width_px"])
thin_ids = [x["image_id"] for x in thin_candidates[:5]]

# 3. Complex Cracks (highest GT instances count, branching)
complex_candidates = [
    img for img in dev_val_images
    if img["has_crack"] and img.get("is_branching") and img["image_id"] not in thin_ids
]
complex_candidates.sort(key=lambda x: x["gt_instances_count"], reverse=True)
complex_ids = [x["image_id"] for x in complex_candidates[:5]]

# 4. Shadowed Surfaces (high shadow percentage, not in previous categories)
shadow_candidates = []
for img in dev_val_images:
    iid = img["image_id"]
    if iid in neg_ids or iid in thin_ids or iid in complex_ids:
        continue
    p = ROOT / img["image_path"]
    b = cv2.imread(str(p))
    rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    sh_mask = compute_shadow_mask(rgb)
    sh_pct = float(np.mean(sh_mask) * 100.0)
    if sh_pct >= 20.0:
        shadow_candidates.append((img, sh_pct))

shadow_candidates.sort(key=lambda x: x[1], reverse=True)
shadow_ids = [x[0]["image_id"] for x in shadow_candidates[:5]]

# 5. Difficult Concrete Textures (high Laplacian variance / rough aggregate)
texture_candidates = []
for img in dev_val_images:
    iid = img["image_id"]
    if iid in neg_ids or iid in thin_ids or iid in complex_ids or iid in shadow_ids:
        continue
    p = ROOT / img["image_path"]
    gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    texture_candidates.append((img, lap_var))

texture_candidates.sort(key=lambda x: x[1], reverse=True)
texture_ids = [x[0]["image_id"] for x in texture_candidates[:5]]

# 6. Medium / Wide Cracks (remaining slots to make exactly 30: 30 - (4+5+5+5+5) = 6)
wide_candidates = [
    img for img in dev_val_images
    if img["has_crack"] and img.get("mean_crack_width_px", 0) >= 8.0
    and img["image_id"] not in (neg_ids + thin_ids + complex_ids + shadow_ids + texture_ids)
]
wide_candidates.sort(key=lambda x: x["mean_crack_width_px"], reverse=True)
wide_ids = [x["image_id"] for x in wide_candidates[:6]]

categories = {
    "negative_clean_facade": neg_ids,
    "thin_hairline_crack": thin_ids,
    "complex_multi_crack": complex_ids,
    "shadowed_surface": shadow_ids,
    "difficult_texture": texture_ids,
    "medium_wide_crack": wide_ids,
}

pilot_images = []
seen = set()
for cat_name, ids in categories.items():
    for iid in ids:
        assert iid not in seen, f"Duplicate image {iid}"
        seen.add(iid)
        img_info = dict(img_map[iid])
        img_info["pilot_category"] = cat_name
        pilot_images.append(img_info)

assert len(pilot_images) == 30, f"Expected 30 pilot images, got {len(pilot_images)}"

output_manifest = {
    "total_images": len(pilot_images),
    "split_breakdown": {
        "dev": sum(1 for x in pilot_images if x["split"] == "dev"),
        "val": sum(1 for x in pilot_images if x["split"] == "val"),
        "test": sum(1 for x in pilot_images if x["split"] == "test"),
    },
    "category_counts": {k: len(v) for k, v in categories.items()},
    "images": pilot_images,
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output_manifest, f, indent=2)

print(f"Successfully generated 30-image pilot manifest: {OUT_PATH}")
print(f"Splits: {output_manifest['split_breakdown']}")
print("Categories:")
for k, v in categories.items():
    print(f"  {k} ({len(v)}): {v}")
