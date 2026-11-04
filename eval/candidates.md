# Candidate changes — PROPOSED, none validated

Rule: nothing here may change code, prompts, or thresholds until it shows
a measured lift on `eval/benchmark_v1.json` via `run_evaluation`.
Intuition goes here, not into the adapter.

| ID | Type | Proposal | Rationale | Status | Evidence required |
|---|---|---|---|---|---|
| C-01 | prompt | `stilt parking` for open_ground_floor | `open ground floor` fires on streets/courtyards (sweep) | proposed | sweep on ≥3 stilt positives + negatives; beat baseline recall |
| C-02 | prompt | `open parking floor` for open_ground_floor | alternate phrasing, same failure | proposed | same as C-01, compared as variant |
| C-03 | prompt | `concrete slab` / `slab edge` for floor_slab | `floor slab` returns zero on visible slabs | proposed | sweep on img2–img5; any true slab rows |
| C-04 | prompt | `roof wall` for parapet | `parapet` zero on plausible edges | proposed | rooftop images first, then variant compare |
| C-05 | threshold | IoU 0.5 for box matching | provisional default from convention | provisional | sensitivity check (0.3/0.5/0.75) after first labels land |
| C-06 | method | NMS/duplicate-merge before counting | column img3 gave 20 boxes, possible overlaps | idea | overlap analysis on labelled columns; must not hide true neighbors |
| C-07 | prompt | `wall crack` for crack | shorter noun phrase; `crack in wall` may over-constrain | proposed | run on crack_wall pair + negatives after GT lands; variant compare |
| C-08 | prompt | `cracked wall` for crack | adjective form names the damaged surface | proposed | same as C-07, compared as variant |
| C-09 | prompt | `crack` for crack | minimal single-concept phrase | proposed | same as C-07; watch false alarms on stains/wires |
| C-10 | prompt | `plaster crack` for crack | names the material failure directly | proposed | same as C-07, compared as variant |
| C-11 | method | ignore-regions for ambiguous zones | R4 crazing, D2 hole, spall edges cannot be boxed fairly yet | idea | implement only if ambiguous FPs recur after first labels; predictions inside ignore boxes dropped before scoring |

Validated improvements (empty until measured): none yet.
