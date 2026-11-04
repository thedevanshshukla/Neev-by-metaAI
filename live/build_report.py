"""Build a static inspection-screening report from cached detections.

Offline only: replays cached SAM run traces, never inference or API
calls. Images without cached cells are listed as not screened.

Usage:
  python live/build_report.py <image>... --out runs/inspection.html
  python live/build_report.py dataset/real_world/cracks/cr5.jpeg --out runs/demo.html --parser-wheel <wheel>
Then open the HTML file in a browser (no server needed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neev.eval.compare import collect_predictions  # noqa: E402
from neev.evidence.report import build_html, gather_image, render_assets  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default="runs/inspection_report.html")
    parser.add_argument("--parser-wheel", default=None,
                        help="meta_sam_parser wheel enabling mask overlays")
    parser.add_argument("--title", default="Neev inspection screening")
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        print(f"ERROR: refusing to overwrite {out_path}", file=sys.stderr)
        return 2
    for image in args.images:
        if not Path(image).is_file():
            print(f"ERROR: no such image: {image}", file=sys.stderr)
            return 2
    if args.parser_wheel:
        sys.path.insert(0, args.parser_wheel)
    predictions = collect_predictions(args.runs_dir)
    images_data = [gather_image(image, predictions) for image in args.images]
    assets_dir = out_path.parent / (out_path.stem + "_assets")
    assets = {}
    for data in images_data:
        stem = Path(data["image"]).stem
        rendered = render_assets(data, assets_dir, stem)
        assets[Path(data["image"]).name] = {
            "overlay": f"{assets_dir.name}/{rendered['overlay']}",
            "crops": {fid: f"{assets_dir.name}/{name}"
                      for fid, name in rendered["crops"].items()},
            "caption": rendered["caption"],
        }
    out_path.write_text(build_html(images_data, assets, args.title),
                        encoding="utf-8")
    total = sum(len(d["findings"]) for d in images_data)
    print(f"images: {len(images_data)} findings: {total}")
    print(f"report: {out_path}")
    print("Open the HTML file in a browser (file:// works, no server).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
