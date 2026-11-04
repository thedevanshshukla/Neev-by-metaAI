# Crack Mask Pilot — Annotation Specification (v1)

Status: SPECIFICATION ONLY. No pilot masks have been drawn yet.
Nothing here modifies `eval/annotations_label_studio.json`, which stays
the authoritative box-level ground truth.

## 1. Goal

Decide whether strict rotated-box metrics underestimate SAM 3.1 crack
localization. Seven images get hand-drawn crack polygons; the additive
mask evaluator (`neev/eval/mask_eval.py`) then scores cached SAM mask
rasters against them. No new inference is needed for the pilot.

## 2. Pilot images (verified IDs, paths, native dimensions)

| image_id | task_id | path | W×H | crack GT boxes | role |
|---|---|---|---|---|---|
| cr4 | 7 | `dataset/real_world/cracks/cr4.jpeg` | 638×480 | 52 | dense network, close-up |
| cr5 | 8 | `dataset/real_world/cracks/cr5.jpeg` | 554×554 | 25 | branching cracks, close-up |
| dm1 | 2 | `dataset/real_world/confusing_category/dm1.png` | 2400×1792 | 27 | cracks + spalling discrimination |
| k1 | 13 | `dataset/real_world/Difficult_Visibility/k1.png` | 2752×1536 | 4 | wide scene, strict-TP control |
| h2 | 10 | `dataset/real_world/Crack_Detection_Edge_Cases/h2.png` | 1448×1086 | 11 | close-up, all prompts silent |
| i1 | 21 | `dataset/real_world/Multiple_Defects_in_One_Building/i1.png` | 2752×1536 | 6 | wide scene + spalling/stains |
| in1 | 17 | `dataset/real_world/intact/in1.png` | 2752×1536 | 0 | explicit negative |

Do not relabel stains, mortar joints, spalling, shadows, wires, or
compression artifacts as cracks. `dm1` and `i1` intentionally mix
cracks with lookalikes; when in doubt, leave it out and note the
region in the annotator log.

## 3. Instance policy

- Trace only the visibly identifiable crack path (dark line / surface
  separation). Polygon edges stay within ~2 px of the visible line at
  100% zoom; never paint broad wall regions.
- One connected, visibly continuous crack network = one instance
  (branches included). A visibly disconnected crack = a separate
  instance. Never bridge gaps you cannot see.
- Never infer hidden extent behind pipes, frames, foliage, or glare.
- Minimum instance: at least 3 polygon points and nonzero area after
  rasterization; hairlines below 1 px width are traced at 1 px and
  flagged `thin: true`.

## 4. Coordinates and edge handling

- Native pixel coordinates, origin top-left, x right, y down.
- All coordinates must satisfy 0 ≤ x ≤ W, 0 ≤ y ≤ H for the image's
  native W/H above. A crack running out of frame ends its polygon at
  the frame edge (clamp to 0/W/H); do not extrapolate beyond the edge.
- Rings are implicitly closed; self-intersecting rings are invalid.

## 5. Output schema (`eval/mask_pilot/gt_masks.v1.json`)

```json
{
  "version": 1,
  "spec": "eval/mask_pilot/ANNOTATION_SPEC.md",
  "images": [
    {
      "image_id": "cr4",
      "task_id": 7,
      "width": 638,
      "height": 480,
      "negative": false,
      "instances": [
        {"instance_id": "cr4-c1", "thin": false,
         "polygons": [[[x, y], ...]]}
      ]
    },
    {
      "image_id": "in1",
      "task_id": 17,
      "width": 2752,
      "height": 1536,
      "negative": true,
      "instances": []
    }
  ]
}
```

Rules: `image_id` unique; `instance_id` unique per image;
`negative: true` requires empty `instances`; coordinates are numbers
(floats allowed); each polygon ring has ≥ 3 points; width/height must
equal the native image dimensions or evaluation fails loudly.

## 6. Prediction-mask side (already cached, no API needed)

Cached SAM runs store one_bit mask payloads + dims per detection
(verified: 18/18 baseline + 47/47 experiment crack detections carry
payloads). Decode offline with `neev.measurements.masks.decode_mask`
(the same path overlays use). Never substitute a bounding box for a
predicted mask. A detection without a decodable payload is reported
as missing, never zero-filled.

## 7. Quality bar before scoring

- Second-look pass on `dm1`/`i1` for crack-vs-spalling leaks.
- Automated check: schema validation + dimension match +
  rasterization smoke test (every instance yields > 0 pixels unless
  `thin` and sub-pixel, which is logged, not silently dropped).
- The pilot answers one question: does mask-IoU scoring change the
  verdict on `crack` vs baseline? It does not certify production use.
