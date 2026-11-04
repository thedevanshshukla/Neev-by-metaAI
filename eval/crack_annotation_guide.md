# Crack annotation guide (human labelling instructions)

Draw boxes on the IMAGES, blind to SAM predictions. Ground truth must
never be derived from model output. When unsure about a region, leave it
unknown — an unlabelled region is excluded from scoring; a wrong label
corrupts every metric.

Box policy: source pixels, x1/y1 exclusive, tight on the visible trace
(not the wall, not the stain), one box per connected trace. Cracks,
peeling, spalling, and joints are DIFFERENT classes: only thin traces
go under `crack`.

## crack_wall.jpeg (738x414) — thin-crack POSITIVE candidate

- R1 POSITIVE: long diagonal crack, left third, top edge to bottom-left.
- R2 POSITIVE: mid-left branch running toward image center. One box.
- R3 POSITIVE: vertical crack cluster right-of-center under the coping.
  One box per connected trace (expect 1–2; annotator decides).
- R4 AMBIGUOUS — do not box: fine center crazing; paint crazing and
  render cracks cannot be separated at this resolution. Stays unknown.
- R5 NOT crack: base-band peeling paint, exposed brick, damp stains.
- R6 NOT crack: coping-stone mortar joints. Joints are never cracks.

## crack_wall2.jpeg (480x359) — SPALLING, not a thin crack

- D1 SPALLING: central exposed-masonry zone. Label under `spalling`
  when that workflow starts. For thin-crack scoring this image stays
  UNLABELLED (unknown) — never mark it a crack negative.
- D2 AMBIGUOUS: dark hole at right edge. A hole, not a crack. Unknown.
- D3 NOT crack: stone mortar joints in the exposed masonry.
- Upper white render shows no clear cracks but image-level negative is
  INVALID here (damage exists elsewhere). Region negatives need crops.

## Proposed negatives — VERIFY FIRST at full resolution

The reviewer saw no cracks in these, but a human must confirm each
before writing `present: false`. One missed crack poisons precision.

- villa-01 (images.jpeg): clean modern render. Candidate negative.
- terraces-01 (img5.jpeg): brick facades. Candidate negative.
- portrait-outlier-01 (outlier1.jpeg): no building. Trivial negative.
- chawl-frontal-01, chawl-corner-01: AMBIGUOUS (stains may hide
  cracks). Must stay unknown; never negatives.
- lane-oblique-01: REVIEW REQUIRED (aged walls, possible cracks).

## Schema snippets (eval/benchmark_v1.json)

Positive: "crack": {"present": true, "boxes": [{"x0":..,"y0":..,
  "x1":..,"y1":..}], "masks": []}
Negative:  "crack": {"present": false, "boxes": [], "masks": []}
Unknown:   omit the key, or "present": null with empty boxes/masks.

Region-level ambiguity (R4, D2) has no box representation yet: leave
such traces unboxed and list them here. See C-11 for the proposed
ignore-regions extension if this recurs.
