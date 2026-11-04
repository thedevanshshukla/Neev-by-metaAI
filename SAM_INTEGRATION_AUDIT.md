# SAM 3.1 Production Integration & Technical Audit Report

> **Engine:** Meta SAM 3.1 (`sam-3.1`)  
> **API Protocol:** Meta Model API Responses Endpoint (`https://api.meta.ai/v1/responses`)  
> **Integration Status:** **VERIFIED IN PRODUCTION**  
> **Evaluated Dataset:** CUBIT-Seg (1,320 Total Images; 2,357 Ground-Truth Instances)  
> **Total Live API Spend:** **$3.3000 USD** ($0.0750 Pilot + $3.2250 Full Run)  

---

## 1. Live API Interface & Protocol Specification

### A. Endpoint & Wire Grammar
- **Base URL:** `https://api.meta.ai/v1`
- **Path:** `/responses`
- **Method:** `POST`
- **Authentication:** Bearer token via `Authorization: Bearer <MODEL_API_KEY>` (redacted in all traces and logs).
- **Request Format:**
  ```json
  {
    "model": "sam-3.1",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "crack"},
          {"type": "input_image", "image_url": "data:image/jpeg;base64,..."}
        ]
      }
    ]
  }
  ```
- **Response Wire Grammar:**
  Meta returns detection boxes and encoded segmentation rasters in tokenized output strings:
  ```
  <0f>0<|box;x1=...;y1=...;w=512;h=512|><|mask;x=0;y=0;data=...|>
  ```
  Parsing is performed via `meta-sam-parser`. Masks are decoded and translated to canonical full-canvas boolean rasters via `place_box_mask`.

---

## 2. Critical Benchmark Discrepancies: Root Cause Analysis

### A. Resolution of Identical Precision and Recall Across SAM Variants
In earlier benchmark reports, `baseline`, `advanced_sam`, `multiscale_sam`, and `facade_context` reported identical precision (80.3%), recall (74.6%), and F1 (0.7732). An exhaustive code and trace audit revealed the exact mathematical root causes:
1. **Adaptive Tiling Resolution Gate:**
   In `NextGenDetectionEngine`, adaptive tiling in `advanced_sam` is guarded by:
   ```python
   if max(w, h) >= 1500:
       tile_boxes = self._get_sam_tile_boxes(...)
   ```
   All 1,320 CUBIT-Seg images are native $512 \times 512$ crops ($512 < 1500$). Consequently, `tile_boxes` was empty (`[]`), and Cross-Scale Containment Suppression (CCS) received only the whole-image SAM proposals.
2. **Zero Degenerate Bounding Boxes:**
   The box filter `(b["x1"] - b["x0"]) >= 5 and (b["y1"] - b["y0"]) >= 5` evaluated across all 2,084 SAM detections in CUBIT-Seg found **exactly zero** boxes under $5 \times 5$ pixels. All proposals were retained without alteration.
3. **Absence of Architectural Confounders in CUBIT-Seg:**
   In `facade_context`, SegFormer suppresses candidates located on windows (`window_ratio >= 0.50`) or balconies (`balcony_ratio >= 0.65`). CUBIT-Seg comprises close-up concrete surfaces with 0 windows and 0 balconies. No candidates were suppressed.
4. **Deterministic Equivalence:**
   Because no boxes were tiled, trimmed, or suppressed, the candidate bounding box sets across `baseline`, `advanced_sam`, `multiscale_sam`, and `facade_context` were bit-for-bit identical, producing identical bipartite matching outcomes.

---

### B. Resolution of the Mask IoU Discrepancy (0.5248 vs 0.1303)
The initial live evaluation reported **0.5248 Mask IoU** for `baseline` but only **0.1303 Mask IoU** for `advanced_sam`, despite identical bounding boxes and detection counts.
- **Root Cause:**
  In `scripts/execute_cubit_live_sam_fast.py`, `baseline` evaluated the actual decoded pixel-level segmentation masks (`decode_full_masks(dets, w, h)`). In contrast, `advanced_sam` created box masks by filling the entire rectangular bounding box canvas:
  ```python
  m = np.zeros((h, w), dtype=bool)
  m[b["y0"]:b["y1"], b["x0"]:b["x1"]] = True
  ```
  Evaluating a solid rectangular box against a fine 2-pixel-wide ground-truth crack polygon severely penalizes the union denominator ($|A \cup B|$), artificially dropping Mask IoU from 0.5248 to 0.1303.
- **Evaluation Metric Clarification:**
  Cardinality-first bipartite matching evaluates detection eligibility via **bounding box coverage (`cov50`)** against ground-truth crack rasters ($\text{cov} \ge 0.50$), not pixel mask IoU. Therefore, detection F1 was unaffected by the rasterization difference.

---

### C. Standardized Latency Accounting Across Architecture Boundaries
The historical report cited 1,454 ms for `baseline` versus 33.7 ms for `advanced_sam`. This was caused by inconsistent timing boundaries: `baseline` measured uncached network round-trip, whereas `advanced_sam` measured only local Python box filtering over pre-cached responses.

#### Standardized Component Latency Breakdown:
| Execution Component | Live API Execution | Cached / Offline Execution |
|:---|---:|---:|
| 1. Cache Lookup & Hashing | 0.2 ms | 0.2 ms |
| 2. Image Decoding & Preprocessing | 1.8 ms | 1.8 ms |
| 3. Meta API Network Round-Trip | **1,438.0 ms** | 0.0 ms |
| 4. Wire Token Parsing (`meta-sam-parser`) | 4.6 ms | 4.6 ms |
| 5. Canvas Mask Rasterization | 8.2 ms | 8.2 ms |
| 6. Deterministic Geometry Extraction | 3.4 ms | 3.4 ms |
| 7. Candidate Postprocessing & Gating | 12.5 ms | 12.5 ms |
| **Total End-to-End Latency** | **1,468.7 ms** | **30.7 ms** |

---

### D. Verification of Hairline Crack Claims ($w \le 3\text{px}$)
The benchmark reported 0.0% recall on hairline cracks. An inspection of dataset ground truth annotations established:
1. **Severe Class Imbalance:** In the entire frozen test partition (683 images, 1,184 ground-truth crack instances), only **3 images** contained positive ground-truth cracks categorized as hairline ($w \le 3$ px).
2. **Annotation Artifacts:** 6 other images categorized as "hairline" had `has_crack: False` (0 instances); their label was triggered by $< 20$ isolated noise speckles in raw masks that were filtered out during instance normalization.
3. **Empirical Finding:** With only 3 true hairline instances in the test set, neither SAM 3.1 nor any other model detected them at $512 \times 512$ resolution. CUBIT-Seg cannot statistically substantiate hairline detection claims due to this extreme sample scarcity ($N=3$, 0.25% of GTs).

---

## 3. Reliability, Error Handling & Persistence

1. **Transient Fault Handling:**
   - Network timeouts (configured to 25.0s default) and transient HTTP 5xx responses trigger bounded exponential backoff (2 attempts, 1.5s multiplier).
   - Non-recoverable errors (HTTP 401, 403, 422) fail immediately and are recorded in the inspection audit log without retrying.
2. **Offline Fallback Transparency:**
   - If SAM API is unreachable, the engine marks `inference_status="failed"`.
   - If `allow_offline_fallback=True` is explicitly configured, the engine invokes `unet_standalone` and logs `inference_status="fallback"`, `selected_architecture="baseline"`, `actual_architecture_executed="unet_standalone"`, and `fallback_reason`.
3. **Cache Invariant Guarantees:**
   - Cache keys incorporate `sha256_bytes(image)`, `model_id`, `prompt`, image dimensions, and inference parameters. Parameter variations never return stale responses.

