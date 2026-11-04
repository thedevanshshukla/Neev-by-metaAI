# SAM 3.1 evaluation plan (local benchmark, no API calls to score)

Ground truth lives in `eval/benchmark_v1.json` (schema: `neev/eval/benchmark.py`).
Status today: **all categories insufficient-labels — nothing is measured yet.**
An unlabelled image is excluded from every metric, never a negative.

Box rules (all categories): source pixels, x1/y1 exclusive, tight on the
visible object, one box per instance. Masks optional (COCO-style RLE);
add them only where box edges decide downstream math (building, balcony).

## 1. Buildings — labels needed: all 5 sweep images

Positives: every image (villa-01, chawl ×2, lane, terraces). Negatives:
none yet; add 1–2 road/sky-only frames later. Prompt: `building` (kept
from sweep). Good means: one box per building + masks that follow the
silhouette (overhang and pounding math needs edges, not blobs).

## 2. Windows — labels needed: all 5 sweep images

Positives: every image; box every glazed opening incl. glass doors.
Prompt: `window` (kept). Good means: boxes cluster into the correct
storey ROWS (the downstream use) — row-count agreement vs ground truth
matters more than per-box precision. Watch oblique recall (img3 gave 1).

## 3. Columns — labels needed: img2, img3 positive; rest to confirm

Positives: chawl-frontal (pillars, 3 floors), chawl-corner. Negatives:
confirm while labelling (the villa porch HAS pillars — do not pre-mark
negatives blind). Prompt: `column` (kept). Good means: one box per
pillar; flag duplicates (img3 gave 20 — check overlaps before counting).

## 4. Balconies — labels needed: img2, img3, img4, villa to confirm

Rule first: box the slab-plus-railing unit as one instance (veranda
walkways count; lone railings on flat facades do not — record the rule
with the labels). Prompt: `balcony` (kept tentatively). Good means:
band boxes whose widths feed the overhang ratio.

## 5. Open ground floors / stilt parking — CRITICAL GAP, no positives

Rule: box the open band between ground and first slab, pillar to pillar.
No true stilt-parking image exists yet. Prompts: `open ground floor`
is FAILING (fires on streets/courtyards); reword candidates in
`candidates.md` (C-01, C-02), all unvalidated. `car parked under
building` and `column` are corroborating evidence, scored separately.
Need: 3+ stilt positives + 3+ enclosed-ground-floor negatives.

## 6. Floor slabs — labels needed: img2, img3, img4, img5

Rule: structural slab edges only (thin boxes); exclude cornices and
string courses unless they sit at a floor line — record the call.
Prompt: `floor slab` returns ZERO everywhere despite visible slabs:
reword candidates in `candidates.md` (C-03). Good means: slab rows
align per building (pounding needs cross-building comparison later).

## 7. Wall cracks — INSUFFICIENT, do not measure yet

Positive CANDIDATE (unlabelled): crack_wall.jpeg, ~4 thin traces
(738×414; see `eval/crack_annotation_guide.md` for the region audit).
crack_wall2.jpeg is SPALLING, not a thin crack — see §10. Negatives
needed: 3 verified (villa, terraces, outlier are candidates — confirm
first); img2/img3 stain bands are ambiguous, never negatives. Rule: one
tight box per connected trace; peeling, stains, and joints are NOT
cracks. Box IoU at 0.5 is harsh on thin diagonals: inspect near-misses
by hand (see C-05). Prompts: baseline `crack in wall` + C-07–C-10
variants. No precision/recall until labels land; one positive image is
preliminary evidence only.

## 10. Spalling (missing masonry/render) — INSUFFICIENT, no prompt yet

Separate category from thin cracks; the two must never share boxes.
Positive candidate: crack_wall2.jpeg central exposed-masonry zone
(unlabelled). No sweep prompt targets spalling, so labelled spalling
reports "no prompts mapped" until one is proposed. Rule: box the
missing-material zone, not surrounding stains.

## 8. Exposed steel / rebar — INSUFFICIENT, do not measure yet

Zero positive examples anywhere (crack_wall2 shows spalled masonry, no
visible bars). Need: construction-site and damage close-ups with visible
bars + intact-concrete negatives. Prompt: `exposed steel bar` (zero
data). Same ≥3/≥3 labelling bar as cracks.

## 9. Parapets and rooftop hazards — labels needed, images missing

Rule: parapet = low roof-edge wall (chimneys do not count);
rooftop_unit = tanks and AC units; hoarding = boards/signage. Positives:
none confirmed (villa roofline is a maybe — confirm while labelling).
Need: rooftop-view images. Prompts: `parapet` (weak, reword C-04),
`water tank` + `air conditioner unit` (shared category, genuine variant
comparison once labelled), `hoarding` (zero false alarms so far — mild
positive, still unmeasured).

## Scoring

`neev/eval/compare.py:run_evaluation` joins labels with cached runs:
IoU 0.5 greedy matching (provisional default, see C-05), duplicates and
malformed predictions count as false positives, unlabelled cells are
excluded. Report per (category, prompt): TP/FP/FN, precision, recall,
F1, images labelled/excluded, mean mask IoU where masks exist. No
threshold or prompt changes without a measured lift on this benchmark.
