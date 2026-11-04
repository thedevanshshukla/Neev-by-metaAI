"""Minimal SAM 3.1 smoke test: one image, a small prompt set.

SAFETY: this script is a dry run by default. It prints the exact planned
request and exits WITHOUT sending anything. Add --confirm-live to send,
which spends a fraction of a cent per image call.

Usage:
  python live/smoke_sam31.py --image samples/street.jpg            # dry run
  python live/smoke_sam31.py --image samples/street.jpg --confirm-live

Requires MODEL_API_KEY in the environment or .env (see .env.example).
Every sent request is traced under runs/<run_id>/ for repeatability.
Identical repeats are served from cache instead of re-billing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neev.adapters.base import UrllibTransport  # noqa: E402
from neev.adapters.sam31 import (  # noqa: E402
    SAM31Adapter,
    build_image_request,
    sanitized_request,
    sniff_image_mime,
)
from neev.config import auth_headers, load_config  # noqa: E402
from neev.persistence.runs import RunMeta, RunStore, sha256_bytes  # noqa: E402

DEFAULT_PHRASES = ["building", "window"]  # exercises building + facade passes


def preview_request(request: dict) -> dict:
    """Copy of the request with base64 payloads truncated for display."""
    import copy

    shown = copy.deepcopy(request)
    for item in shown.get("input", []):
        for part in item.get("content", []):
            for key in ("image_url", "video_url"):
                if key in part:
                    value = part[key]
                    head, _, _ = value.partition(";base64,")
                    part[key] = f"{head};base64,<{len(value)} chars>"
    return shown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Street photo to segment")
    parser.add_argument(
        "--phrase",
        action="append",
        default=None,
        help="Text prompt (repeatable). Default: building, window",
    )
    parser.add_argument("--runs-dir", default=None, help="Trace directory")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Actually POST to the Meta Model API (billable)",
    )
    args = parser.parse_args()

    phrases = args.phrase or list(DEFAULT_PHRASES)
    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"ERROR: no such image: {image_path}", file=sys.stderr)
        return 2
    raw = image_path.read_bytes()
    if not raw:
        print(f"ERROR: empty image: {image_path}", file=sys.stderr)
        return 2
    mime = sniff_image_mime(raw)
    if mime is None:
        print(f"ERROR: not a JPEG/PNG/GIF/WebP/BMP image: {image_path}",
              file=sys.stderr)
        return 2

    config = load_config()
    if args.runs_dir:
        import os

        config = load_config(env={**os.environ, "NEEV_RUNS_DIR": args.runs_dir})
    adapter = SAM31Adapter(config, UrllibTransport())
    store = RunStore(config.runs_dir)

    import base64

    image_b64 = base64.b64encode(raw).decode("ascii")
    image_sha = sha256_bytes(raw)
    planned = [build_image_request(p, image_b64, mime) for p in phrases]

    print("=== SAM 3.1 smoke test ===")
    print(f"endpoint : POST {adapter.url}")
    print(f"image    : {image_path} ({len(raw)} bytes, mime {mime}, sha256 {image_sha[:16]}...)")
    print(f"phrases  : {phrases} ({len(phrases)} request(s))")
    print("auth     : Bearer [REDACTED] (MODEL_API_KEY configured: "
          f"{bool(config.model_api_key)})")
    print("--- planned request #1 (payload truncated) ---")
    print(json.dumps(preview_request(planned[0]), indent=2)[:2000])
    if len(planned) > 1:
        print(f"--- ... plus {len(planned) - 1} more identical-shape request(s) ---")

    # Cache identity uses the sanitized form: same model/phrase/image-sha
    # means an exact repeat, without hashing megabytes of base64.
    traceable = [sanitized_request(request, image_sha) for request in planned]
    cached = [store.find_cached(request) for request in traceable]
    if all(cached):
        print(f"CACHE HIT: all {len(planned)} request(s) already traced; "
              "no new inference needed.")
        for phrase, run_id in zip(phrases, cached):
            print(f"  {phrase!r} -> runs/{run_id}/")
        return 0

    if not args.confirm_live:
        print("DRY RUN: nothing sent. Re-run with --confirm-live to send.")
        return 0

    if not config.model_api_key:
        print("ERROR: MODEL_API_KEY is not set (env or .env). Nothing sent.",
              file=sys.stderr)
        return 2
    # Touch auth early so a missing key fails before any billing.
    auth_headers(config)

    failures = 0
    for phrase, trace in zip(phrases, traceable):
        hit = store.find_cached(trace)
        if hit:
            print(f"[{phrase}] cache hit -> runs/{hit}/ (not re-sent)")
            continue
        try:
            result = adapter.segment_image(raw, phrase, mime,
                                           timeout=args.timeout)
        except Exception as exc:  # report, keep other phrases running
            print(f"[{phrase}] FAILED: {exc}")
            failures += 1
            continue
        run_id = store.save(
            trace,
            result.raw_response,
            RunMeta(
                purpose="sam31-smoke",
                models={"segment": config.sam_model},
                input_hashes={"image": image_sha},
                notes={"phrase": phrase, "image": str(image_path),
                       "mime": mime, "detections": len(result.detections)},
            ),
        )
        print(f"[{phrase}] {len(result.detections)} detection(s) -> runs/{run_id}/")
        for det in result.detections[:10]:
            box = det.box
            print(f"    track={det.track_id} frame={det.frame_index} "
                  f"box=({box.x0},{box.y0},{box.x1},{box.y1})")
        if result.detections and len(result.detections) > 10:
            print(f"    ... and {len(result.detections) - 10} more")
        if result.raw_output_text:
            print(f"    raw[0:300]: {result.raw_output_text[:300]!r}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
