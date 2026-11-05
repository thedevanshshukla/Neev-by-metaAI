---
title: Neev AI Building Facade Defect Screening
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Neev experimentation harness

Validate the core AI capabilities behind Neev (see `docs.md` for the full
product scope) before building the application: SAM 3.1 building and
facade-part segmentation, Muse Wonder checklist judging, and a repeatable
evaluation process. Phase 7 adds a thin local inspection web app
(`neev/app/`, file-based, no database, no workers, no deployment) on top
of the same pipeline.

Core, unit tests, and live scripts are **standard-library only** and run
on stock Python 3.13.

## Setup

```powershell
python --version        # want 3.13.x
copy .env.example .env  # then fill in MODEL_API_KEY (never commit .env)
python -m pytest        # unit tests, no network, no billing
```

`python -m unittest` works too. No install step: run from the repo root.

## Layout

```text
neev/adapters/      SAM 3.1 + Muse Wonder request builders, parsers, transport
neev/prep/          frame sampling plans, IoU tracking fallback (pure)
neev/measurements/  storeys, openness, overhang, pounding (pure)
neev/validation/    checklist/building checks, Re-film rule, scoring
neev/persistence/   file-based run store + cache (JSON, no database)
neev/app/           Phase 7 inspection web app (FastAPI + static viewer)
neev/eval/          manual label format + metrics with unclear handling
neev/config.py      env/.env loading, presence checks, redaction
prompts/            judge system prompt + judge-reply JSON schema
eval/               label-format example (not measurements)
tests/              unit tests: offline only, zero API calls
live/               billable scripts; dry-run by default, need --confirm-live
```

## Inspection web app (Phase 7)

Local end-to-end flow: image upload → cached-first inspection →
interactive evidence viewer → finding cards → report access. The app
reuses the CLI pipeline (`SAM31Adapter`, `RunStore`, sanitized-trace
cache, Phase 6 evidence/report modules) and shares cache entries with
it in both directions. No database; inspections live under
`runs/inspections/<id>/`. No frontend build step: the viewer is static
HTML/CSS/JS served by FastAPI.

Architecture:

```text
neev/app/main.py       FastAPI routes (health, upload, run, report-data,
                       image/assets/report, reports index, static viewer)
neev/app/validation.py upload checks: size, magic bytes, PIL readability
neev/app/store.py      InspectionStore: runs/inspections/<id>/ JSON + bytes
neev/app/runner.py     cached-first execution, default 16-prompt set,
                       explicit confirm_live gate, per-cell run records
neev/app/reports.py    report-data payloads, overlay/crop/report rendering
                       over collect_predictions + gather_image (offline)
neev/app/static/       dependency-free viewer (index.html, app.js, styles.css)
```

Prerequisites and environment (same secrets policy as the CLI):

```powershell
pip install -e ".[app]"   # fastapi, uvicorn, multipart, httpx, pillow
copy .env.example .env    # MODEL_API_KEY only needed for live inference
```

Launch the backend (from the repo root):

```powershell
python -m uvicorn neev.app.main:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000/ in a browser. Health check:
`GET /health` reports liveness plus `masks_available` and whether live
inference is configured (presence only, never the key).

Run the tests:

```powershell
python -m pytest -q -p no:asyncio          # full suite (222 + app tests)
python -m pytest tests/test_app.py -q -p no:asyncio   # app tests only
```

Upload and inspect (UI or curl):

```powershell
curl -F "file=@photo_Sample/crack_wall.jpeg" http://127.0.0.1:8000/inspections
curl -X POST http://127.0.0.1:8000/inspections/<id>/run `
  -H "Content-Type: application/json" -d "{}"
curl http://127.0.0.1:8000/inspections/<id>/report-data
```

Uploading never triggers inference. A run without `confirm_live`
reuses only compatible cached results (same model, prompt, image hash,
and request form) and leaves uncached prompts as `uncached` — zero
transport calls. Dashboard loads, refreshes, and report rendering are
offline too.

Live inference (billable) needs both explicit confirmation and a key:

```powershell
curl -X POST http://127.0.0.1:8000/inspections/<id>/run `
  -H "Content-Type: application/json" -d '{"confirm_live": true}'
```

Without `MODEL_API_KEY` this fails before anything is sent. There are
no retries and no automatic loops; each uncached prompt gets at most
one live attempt per call, and failures are recorded per prompt while
cached results are preserved.

Reports: each inspection renders the same static HTML report as
`live/build_report.py` at `GET /inspections/<id>/report` (built
offline on first access); `GET /reports` lists all inspections with
status, timestamps, cache/live counts, and report availability. The
offline CLI report workflow is unchanged.

Known limitations:

- Mask overlays decode only with Meta's optional `meta-sam-parser`
  package; without it the viewer is boxes-only and says so
  (`mask_status: decoder_unavailable`, `boxes_only: true`).
- The model provides no confidence score; findings always report
  `confidence: null` ("not provided by model") — never fabricated.
- The prompts `crack`, `wall crack`, and `damaged wall` have no
  evaluator category mapping; report-data keeps the raw prompt with
  `category: null` and adds a presentation-only `display_label`
  (see `PRESENTATION_LABELS` in `neev/app/reports.py`). Evaluator
  semantics are untouched.
- Zero findings means "nothing detected", never "defect-free"; the UI
  and reports carry the screening-only disclaimer throughout.
- Pre-existing quirk (CLI reports only): crop hrefs in
  `live/build_report.py` output contain a raw `#` from finding ids,
  which browsers read as a fragment. The app's report-data URLs and
  per-inspection reports URL-encode these correctly.

## Live smoke test (SAM 3.1)

```powershell
python live/smoke_sam31.py --image samples/street.jpg              # dry run: prints plan, sends nothing
python live/smoke_sam31.py --image samples/street.jpg --confirm-live  # sends 2 requests (~half a cent)
```

Default phrases are `building` and `window`. Every sent request is traced
to `runs/<run_id>/` (sanitized request, verbatim response envelope,
models, hashes); exact repeats hit the cache instead of re-billing.

## Visualization

```powershell
python live/visualize.py --run runs/<run_id>   # boxes + masks -> overlay.png
```

Boxes always draw (needs Pillow). Masks decode only through Meta's
official `meta-sam-parser` package (`pip install meta-sam-parser`, or
`--parser-wheel <wheel>` to use a wheel without installing); without it
the overlay is boxes-only and says so. The original image is untouched.

## Prompt sweep (docs.md section 10)

```powershell
python live/sweep_sam31.py photo_Sample                  # plan only: cache state + cost
python live/sweep_sam31.py photo_Sample --confirm-live   # run uncached cells
```

Runs the 16 documented prompts over N images through the same adapter,
cache, and traces as the smoke test. A `sweep_*.json` manifest records
per-cell status (`ok`/`cached`/`failed`/`invalid`), detection counts,
and mask-payload validity; `--overlays` renders one overlay per cell.
`--phrase` narrows to a subset. Failures never become zeros.

## Local benchmark (no API calls to score)

Hand labels live in `eval/benchmark_v1.json` (all categories start
unlabelled; unlabelled images are excluded, never negatives). The plan
is `eval/EVAL_PLAN.md`; unvalidated ideas stay in `eval/candidates.md`.
`neev.eval.run_evaluation` joins labels with cached runs for per-prompt
TP/FP/FN, precision, recall, F1, and mean mask IoU; `eval/crops_*.json`
+ `neev.eval.materialize` prepare hand-defined crops for future sweeps.

```powershell
python live/evaluate.py                        # offline scoring, no network
python live/evaluate.py --decode-masks --parser-wheel <wheel>  # + mask IoU
python live/validate_labels.py                 # validate Label Studio export
python live/evaluate_ls.py                     # score cache vs LS labels (rotated IoU)
```

Native Label Studio exports parse via `neev.eval.labelstudio` with
rotation preserved (clockwise degrees about the top-left corner, per
official docs); matching uses exact rotated-polygon IoU. The export
file itself is never modified.

## Evaluation

Hand labels are versioned JSON (`neev/eval/labels.py`, example in
`eval/example_labels.json`): per building, `yes`/`no`/`unclear` per
checklist id plus integer `storeys`. Anything not visible is `unclear`.

Metrics (`neev/eval/metrics.py`) always return raw counts beside rates and
count unclear separately: `binary_pr` in both `exclude` and `as_error`
modes (recall with and without unclear, per docs.md), plus per-item
`agreement`, storey `exact_match_rate`, and two-video `consistency`.

## Secrets policy

Credentials come from the environment or `.env`, never from code. Traces
scrub secret-looking values; credential checks report presence only.
`.env` and `runs/` are git-ignored.

## Status: not validated

No model output has been examined yet. Do not claim validation until real
inference responses are eyeballed and the parsers are confirmed against
them. Known unverified items: the Muse Wonder image field names and the
SAM 3.1 mask-raster decoding (boxes parse; rasters stay opaque payloads).
