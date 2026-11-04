"""Prompt sweep (docs.md section 10): all 16 prompts x N images, cached.

Dry run by default: prints the plan, per-cell cache state, uncached count
and cost estimate, and sends NOTHING. Add --confirm-live to execute.

Each cell reuses SAM31Adapter + RunStore: identical repeats hit the cache
instead of re-billing. A manifest JSON records per-cell status, detection
counts and mask-payload validity; failures stay failures, never zeros.
Overlays via --overlays (needs Pillow; masks need meta-sam-parser).

Usage:
  python live/sweep_sam31.py photo_Sample                               # plan only
  python live/sweep_sam31.py photo_Sample --confirm-live               # run it
  python live/sweep_sam31.py photo_Sample --confirm-live --overlays

Each positional argument is an image file or a directory (directories
expand to the image files directly inside, sorted by name).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neev.adapters.base import UrllibTransport  # noqa: E402
from neev.adapters.sam31 import (  # noqa: E402
    SAM31Adapter,
    SAM31_PROMPTS,
    build_image_request,
    sanitized_request,
    sniff_image_mime,
)
from neev.config import auth_headers, load_config  # noqa: E402
from neev.persistence.runs import RunMeta, RunStore, sha256_bytes  # noqa: E402

# Official pricing (dev.meta.ai/docs/pricing-rate-limits.md): SAM 3.1 image
# segmentation at $2.50 per 1,000 images, streaming or not.
PRICE_PER_IMAGE_USD = 2.50 / 1000


def default_prompts() -> list[str]:
    """The 16 docs.md prompts in table order."""
    return [phrase for phrases in SAM31_PROMPTS.values() for phrase in phrases]


def load_visualize():
    path = Path(__file__).resolve().parent / "visualize.py"
    spec = importlib.util.spec_from_file_location("neev_visualize", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Image files to sweep")
    parser.add_argument("--phrase", action="append", default=None,
                        help="Prompt (repeatable). Default: all 16 docs prompts")
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--manifest", default=None, help="Manifest JSON path")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--overlays", action="store_true",
                        help="Render overlay.png per completed cell")
    parser.add_argument("--parser-wheel", default=None)
    parser.add_argument("--confirm-live", action="store_true",
                        help="Actually POST uncached cells (billable)")
    args = parser.parse_args()

    phrases = args.phrase or default_prompts()
    phrases = [p for p in phrases if p and p.strip()]
    if not phrases:
        print("ERROR: no prompts selected.", file=sys.stderr)
        return 2

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    image_args: list[str] = []
    for raw_arg in args.images:
        candidate = Path(raw_arg)
        if candidate.is_dir():
            image_args += sorted(
                str(f) for f in candidate.iterdir()
                if f.is_file() and f.suffix.lower() in exts
            )
        else:
            image_args.append(raw_arg)
    if not image_args:
        print("ERROR: no images selected.", file=sys.stderr)
        return 2

    config = load_config()
    if args.runs_dir:
        import os

        config = load_config(env={**os.environ, "NEEV_RUNS_DIR": args.runs_dir})
    store = RunStore(config.runs_dir)

    import base64

    # Plan cells. Image bytes are read locally; no network happens here.
    cells: list[dict] = []
    for image_arg in image_args:
        image_path = Path(image_arg)
        raw = image_path.read_bytes() if image_path.is_file() else b""
        mime = sniff_image_mime(raw) if raw else None
        for phrase in phrases:
            cell: dict = {"image": str(image_path), "phrase": phrase,
                          "status": "pending"}
            if mime is None:
                cell["status"] = "invalid"
                cell["error"] = "missing, empty, or not an image file"
            else:
                image_b64 = base64.b64encode(raw).decode("ascii")
                request = build_image_request(phrase, image_b64, mime)
                trace = sanitized_request(request, sha256_bytes(raw))
                cell["run_id"] = store.find_cached(trace)
                cell["_trace"] = trace
                cell["_mime"] = mime
                cell["_sha"] = sha256_bytes(raw)
            cells.append(cell)

    runnable = [c for c in cells if c["status"] == "pending"]
    uncached = [c for c in runnable if not c.get("run_id")]
    estimate = len(uncached) * PRICE_PER_IMAGE_USD

    print("=== SAM 3.1 prompt sweep ===")
    print(f"images : {len(image_args)}  phrases: {len(phrases)}  "
          f"cells: {len(cells)}")
    print(f"cached : {len(runnable) - len(uncached)}  "
          f"uncached: {len(uncached)}  "
          f"invalid: {len(cells) - len(runnable)}")
    print(f"estimate: {len(uncached)} image request(s) x "
          f"${PRICE_PER_IMAGE_USD:.4f} = ${estimate:.4f}")
    if not args.confirm_live:
        print("DRY RUN: nothing sent. Re-run with --confirm-live to execute.")
        return 0

    if uncached and not config.model_api_key:
        print("ERROR: MODEL_API_KEY is not set (env or .env). Nothing sent.",
              file=sys.stderr)
        return 2
    if uncached:
        auth_headers(config)  # raises before any billing if the key vanished
    adapter = SAM31Adapter(config, UrllibTransport())

    failures = 0
    for cell in runnable:
        if cell.get("run_id"):
            cell["status"] = "cached"
            continue
        raw = Path(cell["image"]).read_bytes()
        try:
            result = adapter.segment_image(raw, cell["phrase"], cell["_mime"],
                                           timeout=args.timeout)
        except Exception as exc:
            cell["status"] = "failed"
            cell["error"] = f"{type(exc).__name__}: {exc}"
            failures += 1
            continue
        run_id = store.save(
            cell["_trace"],
            result.raw_response,
            RunMeta(
                purpose="sam31-sweep",
                models={"segment": config.sam_model},
                input_hashes={"image": cell["_sha"]},
                notes={"phrase": cell["phrase"], "image": cell["image"],
                       "mime": cell["_mime"],
                       "detections": len(result.detections)},
            ),
        )
        cell["run_id"] = run_id
        cell["status"] = "ok"
        cell["detections"] = len(result.detections)
        cell["empty"] = not result.detections
        payloads = [d for d in result.detections if d.mask_payload]
        cell["masks_with_payload"] = len(payloads)
        cell["masks_known_encoding"] = sum(
            1 for d in payloads if d.mask_encoding is not None)
        print(f"[{cell['image']} :: {cell['phrase']}] "
              f"{len(result.detections)} detection(s) -> runs/{run_id}/")

    if args.overlays:
        visualize = load_visualize()
        for cell in runnable:
            if cell["status"] not in ("ok", "cached") or not cell.get("run_id"):
                continue
            argv = ["visualize.py", "--run",
                   str(Path(config.runs_dir) / cell["run_id"])]
            if args.parser_wheel:
                argv += ["--parser-wheel", args.parser_wheel]
            old_argv = sys.argv
            sys.argv = argv
            try:
                code = visualize.main()
            finally:
                sys.argv = old_argv
            if code == 0:
                cell["overlay"] = str(
                    Path(config.runs_dir) / cell["run_id"] / "overlay.png")
            else:
                cell["overlay_error"] = f"visualize exited {code}"

    for cell in cells:
        cell.pop("_trace", None)
        cell.pop("_mime", None)
        cell.pop("_sha", None)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "sam31-prompt-sweep",
        "model": config.sam_model,
        "price_per_image_usd": PRICE_PER_IMAGE_USD,
        "runs_dir": str(config.runs_dir),
        "images": image_args,
        "phrases": phrases,
        "cells": cells,
        "summary": {
            "cells": len(cells),
            "ok": sum(1 for c in cells if c["status"] == "ok"),
            "cached": sum(1 for c in cells if c["status"] == "cached"),
            "empty": sum(1 for c in cells if c.get("empty")),
            "failed": sum(1 for c in cells if c["status"] == "failed"),
            "invalid": sum(1 for c in cells if c["status"] == "invalid"),
            "sent": sum(1 for c in cells if c["status"] == "ok"),
        },
    }
    stamp = manifest["created_utc"].replace(":", "").replace("+", "")[:15]
    manifest_path = Path(args.manifest or str(Path(config.runs_dir)
                                               / f"sweep_{stamp}.json"))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path} "
          f"(ok={manifest['summary']['ok']} cached={manifest['summary']['cached']} "
          f"empty={manifest['summary']['empty']} failed={manifest['summary']['failed']} "
          f"invalid={manifest['summary']['invalid']})")
    bad = manifest["summary"]["failed"] + manifest["summary"]["invalid"]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
