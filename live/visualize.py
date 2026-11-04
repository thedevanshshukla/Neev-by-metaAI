"""Draw a run's boxes (and decodable masks) over the original image.

Usage:
  python live/visualize.py --run runs/<run_id>
  python live/visualize.py --run runs/<run_id> --parser-wheel <meta_sam_parser wheel>

Reads response.json + meta.json from a smoke-test run, re-parses the saved
output_text with the same parser the smoke test uses, and writes overlay.png
into the run directory. The original image is never modified.

Coordinate mapping: boxes are source pixels, drawn 1:1. The run's frame w/h
must equal the image size or visualization aborts (no silent mis-scaling).
Masks are box-local rasters (per the official parser's
SegmentationMaskBounds type): each raster is scaled to its box and pasted at
the box origin. That placement is inferred from official examples, not stated
in prose docs, so eyeball the overlay before trusting it.

Masks decode only through meta-sam-parser (installed, or via --parser-wheel,
which adds a wheel to sys.path WITHOUT installing anything). Undecodable or
missing masks are reported per detection and boxes still draw: nothing here
is fabricated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neev.adapters.base import extract_output_text  # noqa: E402
from neev.adapters.sam31 import parse_output_text  # noqa: E402
from neev.measurements.masks import MaskError, decode_mask  # noqa: E402

COLORS = [(220, 38, 38), (37, 99, 235), (22, 163, 74), (202, 138, 4),
          (147, 51, 234), (2, 132, 130)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run directory (runs/<id>)")
    parser.add_argument("--parser-wheel", default=None,
                        help="meta_sam_parser wheel for mask decoding (not installed)")
    parser.add_argument("--out", default="overlay.png", help="Output file name")
    args = parser.parse_args()

    if args.parser_wheel:
        sys.path.insert(0, args.parser_wheel)

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("ERROR: Pillow is required: pip install pillow", file=sys.stderr)
        return 2

    run_dir = Path(args.run)
    try:
        response = json.loads((run_dir / "response.json").read_text(encoding="utf-8"))
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot read run {run_dir}: {exc}", file=sys.stderr)
        return 2

    phrase = meta.get("notes", {}).get("phrase", "?")
    image_path = meta.get("notes", {}).get("image")
    if not image_path or not Path(image_path).is_file():
        print(f"ERROR: original image missing: {image_path!r}", file=sys.stderr)
        return 2
    try:
        text = extract_output_text(response)
        detections = parse_output_text(text, phrase)
    except Exception as exc:
        print(f"ERROR: saved response does not parse: {exc}", file=sys.stderr)
        return 2

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    for det in detections:
        if det.frame_width is not None and (det.frame_width, det.frame_height) != (img_w, img_h):
            print(f"ERROR: frame size {det.frame_width}x{det.frame_height} != "
                  f"image {img_w}x{img_h}; refusing to mis-scale.", file=sys.stderr)
            return 2

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    line = max(2, min(img_w, img_h) // 300)
    masks_drawn = 0
    mask_problems: list[str] = []
    for i, det in enumerate(detections):
        color = COLORS[i % len(COLORS)]
        box = det.box
        draw.rectangle([box.x0, box.y0, box.x1, box.y1], outline=color + (255,), width=line)
        draw.text((box.x0 + 3, max(0, box.y0 - 12)), f"{phrase}#{det.track_id}",
                  fill=color + (255,))
        if not det.mask_payload:
            mask_problems.append(f"#{det.track_id}: no mask token on the record")
            continue
        try:
            raster = decode_mask(det.mask_payload, det.mask_width, det.mask_height)
        except MaskError as exc:
            mask_problems.append(f"#{det.track_id}: {exc}")
            continue
        mask_img = Image.frombytes("L", (det.mask_width, det.mask_height),
                                   bytes(b * 255 for b in raster))
        mask_img = mask_img.resize((max(1, box.width), max(1, box.height)), Image.NEAREST)
        tint = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        tint_draw = ImageDraw.Draw(tint)
        tint_draw.bitmap((box.x0, box.y0), mask_img, fill=color + (110,))
        overlay = Image.alpha_composite(overlay.convert("RGBA"), tint).convert("RGB")
        draw = ImageDraw.Draw(overlay, "RGBA")
        masks_drawn += 1

    out_path = run_dir / args.out
    overlay.save(out_path)
    print(f"run        : {run_dir}")
    print(f"detections : {len(detections)}")
    print(f"boxes drawn: {len(detections)}")
    print(f"masks drawn: {masks_drawn}")
    for problem in mask_problems:
        print(f"mask N/A   : {problem}")
    print(f"overlay    : {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
