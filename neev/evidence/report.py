"""Offline inspection-report builder over cached SAM detections.

Reads cached run traces (never inference), assesses image quality,
assigns transparent screening priority, and renders a static HTML
report with combined overlays and finding cards. All amounts are
verbatim model outputs or documented derivations (areas, fractions);
anything unknown is labeled, never invented.
"""
from __future__ import annotations

import hashlib
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from neev.eval.compare import PROMPT_TO_CATEGORY
from neev.evidence.priority import CONTEXT, prioritize
from neev.evidence.quality import QualityThresholds, assess_image

LEVEL_COLORS = {"needs_closer_inspection": (220, 40, 40),
                "review": (210, 140, 0),
                "informational": (70, 120, 200)}
CONFIDENCE_LABEL = "not provided by model"
MAX_CROPS_PER_IMAGE = 12


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _box_info(box: Any, width: int, height: int) -> dict[str, Any]:
    """Verbatim box plus derived areas; malformed boxes flagged, kept."""
    try:
        x0, y0, x1, y1 = (box[k] for k in ("x0", "y0", "x1", "y1"))
    except (KeyError, TypeError, AttributeError):
        return {"box": None, "malformed": True}
    if not all(isinstance(v, int) and not isinstance(v, bool)
               for v in (x0, y0, x1, y1)) or not (x0 < x1 and y0 < y1):
        return {"box": None, "malformed": True}
    area = (x1 - x0) * (y1 - y0)
    return {"box": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "malformed": False, "area_px": area,
            "area_frac": area / (width * height),
            "in_frame": 0 <= x0 and 0 <= y0 and x1 <= width and y1 <= height}


def gather_image(image_path: str | Path, predictions: dict,
                 mask_backend: Callable | None = None,
                 thresholds: QualityThresholds | None = None) -> dict[str, Any]:
    """Collect findings for one image file from cached predictions."""
    from PIL import Image

    from neev.measurements.masks import MaskError, decode_mask, place_box_mask

    path = Path(image_path)
    image = Image.open(path).convert("RGB")
    width, height = image.size
    quality = assess_image(image, thresholds)
    digest = sha256_file(path)
    cells = sorted((phrase, cell) for (sha, phrase), cell in predictions.items()
                   if sha == digest and isinstance(cell, dict)
                   and "boxes" in cell)
    findings: list[dict[str, Any]] = []
    scene_context: list[dict[str, Any]] = []
    for phrase, cell in cells:
        category = PROMPT_TO_CATEGORY.get(phrase)
        dets = cell.get("detections", [])
        for i, det in enumerate(dets if isinstance(dets, list) else []):
            info = _box_info(det.get("box"), width, height)
            mask_status, placed = "missing", None
            if det.get("mask_payload"):
                if mask_backend is None and not _decoder_available():
                    mask_status = "decoder_unavailable"
                else:
                    try:
                        raster = decode_mask(
                            det["mask_payload"], det["mask_width"],
                            det["mask_height"], backend=mask_backend)
                        placed = place_box_mask(
                            raster, det["mask_width"], det["mask_height"],
                            info["box"], width, height)
                        mask_status = "decoded" if any(placed) else "empty"
                    except Exception as exc:
                        mask_status = ("unplaced" if "placement" in str(exc)
                                       else "decode_failed")
            prio = prioritize(
                category, info.get("area_frac"),
                [vars(w) if hasattr(w, "__dict__") else w
                 for w in quality.warnings],
                has_mask=(placed is not None and any(placed)))
            is_defect = (
                category in ("crack", "spalling", "rebar")
                or any(w in phrase.lower() for w in ("crack", "damage", "spall", "steel", "rebar"))
            ) and (category not in CONTEXT)
            item = {
                "finding_id": f"{phrase}#{i}", "prompt": phrase,
                "category": category, "run_id": det.get("source"),
                "confidence": None, "confidence_label": CONFIDENCE_LABEL,
                **info, "mask_status": mask_status, "mask": placed,
                "priority": prio["level"], "reasons": prio["reasons"],
                "next_action": prio["next_action"],
                "view": det.get("view", "whole"),
                "provenance": det.get("provenance"),
                "is_defect": is_defect,
                "kind": "defect" if is_defect else "context",
            }
            findings.append(item)
            if not is_defect:
                scene_context.append(item)
    order = {"needs_closer_inspection": 0, "review": 1, "informational": 2}
    findings.sort(key=lambda f: (order[f["priority"]], f["prompt"],
                                 f["finding_id"]))
    scene_context.sort(key=lambda f: (order[f["priority"]], f["prompt"],
                                      f["finding_id"]))
    return {"image": str(path), "width": width, "height": height,
            "sha256": digest, "quality": quality, "findings": findings,
            "scene_context": scene_context,
            "prompts_screened": sorted({p for p, _ in cells})}


def _decoder_available() -> bool:
    try:
        import meta_sam_parser  # noqa: F401
        return True
    except ImportError:
        return False


def render_assets(data: dict[str, Any], assets_dir: Path, stem: str) -> dict:
    """Combined overlay + finding crops. Returns asset filenames."""
    from PIL import Image, ImageDraw

    assets_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(data["image"]).convert("RGB")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    outlined_only = 0
    for f in data["findings"]:
        if f.get("is_defect") is False or f.get("category") in CONTEXT:
            continue
        if f["mask"]:
            # Full-frame masks (e.g. whole-wall) would wash out the image;
            # show those as box outlines only. Boxes always draw verbatim.
            if sum(f["mask"]) / len(f["mask"]) > 0.5:
                outlined_only += 1
                continue
            raw = Image.frombytes("L", image.size, bytes(
                b * 110 for b in f["mask"])).convert("RGBA")
            red = Image.new("RGBA", image.size, (220, 40, 40, 0))
            red.putalpha(raw.split()[0])
            overlay = Image.alpha_composite(overlay.convert("RGBA"), red)
            draw = ImageDraw.Draw(overlay, "RGBA")
    for f in data["findings"]:
        if not f["box"]:
            continue
        color = LEVEL_COLORS[f["priority"]]
        b = f["box"]
        draw.rectangle([b["x0"], b["y0"], b["x1"], b["y1"]],
                       outline=color + (255,), width=max(2, data["width"] // 500))
    overlay_name = f"{stem}_overlay.png"
    overlay.convert("RGB").save(assets_dir / overlay_name)
    crops = {}
    ranked = [f for f in data["findings"] if f["box"]][:MAX_CROPS_PER_IMAGE]
    for f in ranked:
        b = f["box"]
        pad = max(8, int(0.1 * max(b["x1"] - b["x0"], b["y1"] - b["y0"])))
        x0 = max(0, b["x0"] - pad)
        y0 = max(0, b["y0"] - pad)
        x1 = min(data["width"], b["x1"] + pad)
        y1 = min(data["height"], b["y1"] + pad)
        crop = image.crop((x0, y0, x1, y1))
        crop.thumbnail((420, 420))
        d = ImageDraw.Draw(crop)
        sx = crop.width / (x1 - x0)
        sy = crop.height / (y1 - y0)
        d.rectangle([(b["x0"] - x0) * sx, (b["y0"] - y0) * sy,
                     (b["x1"] - x0) * sx, (b["y1"] - y0) * sy],
                    outline=LEVEL_COLORS[f["priority"]], width=2)
        name = f"{stem}_{f['finding_id'].replace(' ', '_').replace('/', '_')}.png"
        crop.save(assets_dir / name)
        crops[f["finding_id"]] = name
    caption = ("Boxes: red = needs closer inspection, amber = review, "
               "blue = informational. Red tint = decoded model masks; "
               "masks covering over half the image draw as outlines only "
               "to keep detail visible.")
    if outlined_only:
        caption += f" ({outlined_only} mask(s) outline-only here.)"
    return {"overlay": overlay_name, "crops": crops, "caption": caption}


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def build_html(images: list[dict[str, Any]], assets: dict[str, dict],
               title: str = "Neev inspection screening") -> str:
    """Render the static report. All dynamic text is HTML-escaped."""
    total = sum(len(d["findings"]) for d in images)
    by_cat: dict[str, int] = {}
    need: list[str] = []
    for d in images:
        for f in d["findings"]:
            by_cat[f["category"] or f"prompt:{f['prompt']}"] = \
                by_cat.get(f["category"] or f"prompt:{f['prompt']}", 0) + 1
            if f["priority"] == "needs_closer_inspection":
                need.append(f"{Path(d['image']).name} {f['finding_id']}")
    parts = [f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>{_esc(title)}</title>
<style>body{{font-family:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:24px auto;padding:0 24px;max-width:1140px;background:#F7F5F0;color:#292735;line-height:1.55}}
h1{{font-size:24px;font-weight:700;color:#292735;letter-spacing:-0.2px;margin-bottom:12px}}
h2{{font-size:18px;font-weight:600;color:#292735;margin-top:28px}}
h3{{font-size:13px;font-weight:600;color:#4A4754;text-transform:uppercase;letter-spacing:0.5px}}
.banner{{background:#FAF8F5;border:1px solid #E8E4DE;border-left:4px solid #6558C9;padding:12px 16px;border-radius:10px;color:#292735;font-size:13px}}
.card{{background:#FFFEFC;border:1px solid #E8E4DE;border-radius:12px;margin:12px 0;padding:14px;box-shadow:0 1px 3px rgba(41,39,53,0.03)}}
.lvl-needs_closer_inspection{{border-left:5px solid #D94A4A}}
.lvl-review{{border-left:5px solid #E08C28}}
.lvl-informational{{border-left:5px solid #6558C9}}
img.view{{max-width:100%;border-radius:10px;border:1px solid #E8E4DE}}
table{{border-collapse:collapse;width:100%;margin:12px 0;background:#FFFEFC;border:1px solid #E8E4DE;border-radius:10px;overflow:hidden}}
th{{background:#F0EEE9;color:#5C5865;font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:0.5px;border:1px solid #E8E4DE;padding:8px 12px;text-align:left}}
td{{border:1px solid #E8E4DE;padding:8px 12px;color:#292735}}
.muted{{color:#85818D;font-size:13px}}
ul{{padding-left:20px;color:#292735}}
a{{color:#6558C9;text-decoration:none}}
a:hover{{text-decoration:underline}}</style></head><body>
<h1>{_esc(title)}</h1>
<p class="banner"><b>Visual screening only - not a structural safety
assessment.</b> Findings are model outputs over cached photos. Absence of a
finding is not proof of absence of a defect. Generated {_esc(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))}.</p>
<h2>Summary</h2>
<p>Images inspected: {len(images)} &middot; Findings: {total}</p>
<h3>Findings by category</h3>
<table><tr><th>Category</th><th>Count</th></tr>"""]
    for cat, count in sorted(by_cat.items()):
        parts.append(f"<tr><td>{_esc(cat)}</td><td>{count}</td></tr>")
    parts.append("</table><h3>Needs closer inspection</h3>")
    parts.append("<ul>" + "".join(f"<li>{_esc(n)}</li>" for n in need) +
                 "</ul>" if need else "<p>None.</p>")
    parts.append("<h3>Recommended next steps</h3><ul>"
                 "<li>Open each flagged finding and compare the overlay with the original photo.</li>"
                 "<li>Re-photograph flagged regions closer and sharper where image quality warned.</li>"
                 "<li>Escalate confirmed damage to a qualified inspector; this tool cannot certify safety.</li>"
                 "</ul>")
    for d in images:
        name = Path(d["image"]).name
        a = assets.get(name, {})
        parts.append(f"<h2>{_esc(name)} <span class='muted'>{d['width']}x{d['height']}, "
                     f"{len(d['findings'])} findings</span></h2>")
        for w in d["quality"].warnings:
            parts.append(f"<p class='banner'>Quality warning "
                         f"({_esc(w['check'])}): {_esc(w['detail'])}</p>")
        if a.get("overlay"):
            parts.append(f"<p><img class='view' src='{_esc(a['overlay'])}' "
                         f"alt='combined overlay'><br><span class='muted'>"
                         f"{_esc(a.get('caption', ''))}</span></p>")
        parts.append(f"<p class='muted'>Original (source of truth): "
                     f"{_esc(d['image'])}</p>")
        if not d["findings"]:
            parts.append("<p><b>No findings.</b> Either no cached detections "
                         "exist for this image or every screened prompt "
                         "returned empty. This is not evidence the building "
                         "is defect-free.</p>")
        for f in d["findings"]:
            box = (f"{f['box']['x0']},{f['box']['y0']}-"
                   f"{f['box']['x1']},{f['box']['y1']}px"
                   if f["box"] else "unavailable (malformed box)")
            area = (f"{f['area_px']}px ({f['area_frac'] * 100:.2f}% of image)"
                    if f.get("area_px") is not None else "unknown")
            crop = a.get("crops", {}).get(f["finding_id"], "")
            parts.append(
                f"<div class='card lvl-{f['priority']}'>"
                f"<b>{_esc(f['finding_id'])}</b> &middot; category: "
                f"{_esc(f['category'] or 'unknown (unmapped prompt)')} &middot; "
                f"priority: <b>{_esc(f['priority'])}</b><br>"
                f"Location: {_esc(box)} &middot; Area: {_esc(area)}<br>"
                f"Confidence: {_esc(f['confidence_label'])} &middot; "
                f"Mask: {_esc(f['mask_status'])} &middot; "
                f"Run: {_esc(f['run_id'])}<br>"
                f"Why: {_esc('; '.join(f['reasons']) or 'routine')}<br>"
                f"Next: {_esc(f['next_action'])}"
                + (f"<br><img src='{_esc(crop)}' alt='finding crop'>"
                   if crop else "") + "</div>")
    parts.append("<h2>Priority rules (inspectable)</h2><p class='muted'>"
                 "needs_closer_inspection: any image-quality warning, or a "
                 "potential-defect category (crack, rebar, spalling). review: "
                 "unmapped prompts and non-context findings. informational: "
                 "scene-context categories (building, window, column, ...). "
                 "Small or mask-less detections add cautionary reasons but "
                 "never change the level alone.</p></body></html>")
    return "".join(parts)
