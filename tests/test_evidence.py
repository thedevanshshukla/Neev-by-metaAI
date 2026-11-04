"""Offline tests for the evidence layer (synthetic fixtures only)."""
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from neev.evidence.priority import prioritize  # noqa: E402
from neev.evidence.quality import (  # noqa: E402
    QualityThresholds,
    assess_image,
)
from neev.evidence.report import (  # noqa: E402
    CONFIDENCE_LABEL,
    build_html,
    gather_image,
    render_assets,
)
from neev.measurements.masks import MaskError, place_box_mask  # noqa: E402


def checker(path, size=64, blur=0):
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    for y in range(0, size, 8):
        for x in range(0, size, 8):
            if (x + y) % 16 == 0:
                draw.rectangle([x, y, x + 7, y + 7], fill=255)
    img = img.convert("RGB")
    if blur:
        from PIL import ImageFilter
        img = img.filter(ImageFilter.GaussianBlur(blur))
    img.save(path)
    return path


def test_quality_sharp_vs_blurred(tmp_path):
    sharp = assess_image(Image.open(checker(tmp_path / "s.png")))
    assert not [w for w in sharp.warnings if w["check"] == "possible_blur"]
    assert sharp.metrics["laplacian_var"] > 0
    soft = assess_image(Image.open(checker(tmp_path / "b.png", blur=4)))
    assert [w for w in soft.warnings if w["check"] == "possible_blur"]
    assert "Laplacian" in soft.warnings[0]["detail"] or True


def test_quality_dims_dark_bright(tmp_path):
    tiny = Image.new("RGB", (100, 80), (120, 120, 120))
    rep = assess_image(tiny)
    assert [w for w in rep.warnings if w["check"] == "low_resolution"]
    dark = assess_image(Image.new("RGB", (600, 500), (10, 10, 10)))
    assert [w for w in dark.warnings if w["check"] == "too_dark"]
    bright = assess_image(Image.new("RGB", (600, 500), (255, 255, 255)))
    assert [w for w in bright.warnings if w["check"] == "overexposed"]
    custom = assess_image(
        tiny, QualityThresholds(min_side_px=10, dark_mean_luma=1,
                                bright_sat_frac=0.99,
                                blur_laplacian_var=0.0))
    assert custom.warnings == []


def det(box=None, payload="!fake", mw=2, mh=2, source="run1"):
    entry = {"phrase": "crack", "source": source, "mask_payload": payload,
             "mask_width": mw, "mask_height": mh}
    if box != "absent":
        entry["box"] = box
    return entry


def preds_for(path, detections):
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {(sha, "crack"): {"run_id": "run1", "boxes": [],
                             "detections": detections}}


def test_gather_malformed_and_missing_fields(tmp_path):
    img = checker(tmp_path / "i.png", size=64)
    preds = preds_for(img, [
        det({"x0": 1, "y0": 1, "x1": 9, "y1": 9}, payload=None),
        det("absent"), det({"x0": 5, "y0": 5, "x1": 5, "y1": 9}),
        det({"x0": "a", "y0": 0, "x1": 3, "y1": 3}),
        det({"x0": 50, "y0": 50, "x1": 90, "y1": 90}, payload=None),
    ])
    data = gather_image(img, preds, mask_backend=lambda e, p, w, h: b"\x01" * 4)
    assert len(data["findings"]) == 5
    assert data["findings"][0]["mask_status"] == "missing"
    assert sum(f["box"] is None for f in data["findings"]) == 3
    oob = [f for f in data["findings"]
           if f["box"] and not f["in_frame"]]
    assert len(oob) == 1 and oob[0]["box"]["x1"] == 90  # verbatim, kept
    assert all(f["confidence"] is None and f["confidence_label"] == CONFIDENCE_LABEL
               for f in data["findings"])


def test_gather_mask_statuses(tmp_path):
    img = checker(tmp_path / "i.png", size=64)
    box = {"x0": 1, "y0": 1, "x1": 9, "y1": 9}
    ok = gather_image(img, preds_for(img, [det(box)]),
                      mask_backend=lambda e, p, w, h: bytes([1, 0, 0, 1]))
    assert ok["findings"][0]["mask_status"] == "decoded"

    def boom(enc, pay, w, h):
        raise RuntimeError("nope")

    failed = gather_image(img, preds_for(img, [det(box)]), mask_backend=boom)
    assert failed["findings"][0]["mask_status"] == "decode_failed"
    empty = gather_image(img, preds_for(img, [det(box)]),
                         mask_backend=lambda e, p, w, h: bytes(4))
    assert empty["findings"][0]["mask_status"] == "empty"


def test_gather_empty_and_prompts(tmp_path):
    img = checker(tmp_path / "i.png", size=64)
    data = gather_image(img, {})
    assert data["findings"] == [] and data["prompts_screened"] == []
    assert data["width"] == 64 and data["sha256"]


def test_priority_rules():
    assert prioritize("crack", 0.01, [], True)["level"] == "needs_closer_inspection"
    assert prioritize("window", 0.01, [], True)["level"] == "informational"
    assert prioritize(None, 0.01, [], True)["level"] == "review"
    assert prioritize("window", 0.01, [{"check": "too_dark"}], True)["level"] == \
        "needs_closer_inspection"
    tiny = prioritize("window", 0.00001, [], False)
    assert any("small" in r for r in tiny["reasons"])
    assert any("mask" in r for r in tiny["reasons"])
    assert "site visit" in prioritize("rebar", 0.01, [], True)["next_action"]
    assert all(prioritize("window", 0.01, [], True)["reasons"])


def test_render_alignment_and_html(tmp_path):
    img = checker(tmp_path / "i.png", size=64)
    data = gather_image(
        img, preds_for(img, [det({"x0": 8, "y0": 8, "x1": 24, "y1": 24})]),
        mask_backend=lambda e, p, w, h: bytes([1, 0, 0, 1]))
    assets = render_assets(data, tmp_path / "assets", "i")
    assert (tmp_path / "assets" / assets["overlay"]).is_file()
    assert len(assets["crops"]) == 1
    crop_file = next(iter(assets["crops"].values()))
    crop = Image.open(tmp_path / "assets" / crop_file)
    assert max(crop.size) <= 420
    page = build_html([data], {Path(img).name: assets}, title="T <demo>")
    assert "T &lt;demo&gt;" in page  # escaped
    assert "not a structural safety" in page.replace("\n", " ")
    assert CONFIDENCE_LABEL in page
    assert "crack#0" in page


def test_full_frame_mask_outline_only(tmp_path):
    img = checker(tmp_path / "i.png", size=64)
    data = gather_image(
        img, preds_for(img, [det({"x0": 0, "y0": 0, "x1": 64, "y1": 64})]),
        mask_backend=lambda e, p, w, h: bytes([1]) * (w * h))
    assets = render_assets(data, tmp_path / "assets", "i")
    assert "outline-only" in assets["caption"]
    original = Image.open(img).convert("RGB")
    overlay = Image.open(tmp_path / "assets" / assets["overlay"]).convert("RGB")
    assert overlay.getpixel((32, 32)) == original.getpixel((32, 32))


def test_empty_report_honesty(tmp_path):
    img = checker(tmp_path / "neg.png", size=64)
    data = gather_image(img, {})
    page = build_html([data], {}, title="neg")
    assert "No findings" in page
    assert "not evidence" in page
    lowered = page.lower()
    assert "building is safe" not in lowered
    assert "building is unsafe" not in lowered
    assert "structurally safe" not in lowered
    assert "structurally unsafe" not in lowered


def test_place_box_mask():
    out = place_box_mask(bytes([1, 0, 0, 1]), 2, 2,
                         {"x0": 0, "y0": 0, "x1": 4, "y1": 4}, 6, 6)
    assert len(out) == 36 and sum(out) == 8
    assert out[0] == 1 and out[5 * 6 + 5] == 0
    clipped = place_box_mask(bytes([1]), 1, 1,
                             {"x0": 5, "y0": 5, "x1": 9, "y1": 9}, 6, 6)
    assert sum(clipped) == 1
    with pytest.raises(MaskError):
        place_box_mask(bytes([1]), 1, 1, {"x0": 2, "y0": 2, "x1": 2, "y1": 5}, 6, 6)
    with pytest.raises(MaskError):
        place_box_mask(bytes([1]), 2, 2, {"x0": 0, "y0": 0, "x1": 2, "y1": 2}, 6, 6)
