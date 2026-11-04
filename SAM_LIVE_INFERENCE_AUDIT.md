# SAM 3.1 Live Inference Audit & Benchmark Configuration Review

> **Audit Date:** 2026-11-04  
> **Target Dataset:** CUBIT-Seg Benchmark (1,320 total images; DEV: 264, VAL: 373, TEST: 683)  
> **Official Meta Model Identifier:** `sam-3.1`  
> **API Endpoint:** `https://api.meta.ai/v1/responses`  
> **Authentication Method:** `Authorization: Bearer [REDACTED]` (via `MODEL_API_KEY`)  
> **Official Billing Rate:** **$2.50 per 1,000 images** ($0.0025 USD per image request)  

---

## 1. Audit of Meta SAM 3.1 Adapter & API Wire Contract

### 1.1 Wire Contract & HTTP Specification
The official Meta Model API Responses wire format operates over HTTPS:
- **Method & Route:** `POST https://api.meta.ai/v1/responses`
- **Request Headers:**
  - `Authorization: Bearer [REDACTED]` (loaded securely from environment or `.env`; never printed or stored raw)
  - `Content-Type: application/json`
- **Request Payload Schema:**
  ```json
  {
    "model": "sam-3.1",
    "input": [
      {
        "type": "message",
        "role": "user",
        "content": [
          {"type": "input_text", "text": "crack"},
          {"type": "input_image", "image_url": "data:image/jpeg;base64,<base64_data>"}
        ]
      }
    ],
    "stream": false,
    "metadata": {
      "mask_encoding": "one_bit"
    }
  }
  ```
- **Response Payload Schema:**
  ```json
  {
    "id": "resp_6aeac9c7e40c861e15564e69",
    "object": "response",
    "model": "sam-3.1",
    "status": "completed",
    "usage": {
      "images_processed": 1,
      "total_tokens": 0,
      "input_tokens": 0,
      "output_tokens": 0
    },
    "output": [
      {
        "role": "assistant",
        "type": "message",
        "content": [
          {
            "type": "output_text",
            "text": "<0f>0<|box;x1=0;y1=61;x2=81;y2=70;w=512;h=512|><|mask;x=0;y=0;data=10,82,!!!!!...|>\n"
          }
        ]
      }
    ]
  }
  ```

### 1.2 Parsing & Mask Decoding Protocol
- Output tokens are parsed by regular expression `_TOKEN_RE` in [`neev/adapters/sam31.py`](file:///d:/projects/metaglobal/neev/adapters/sam31.py):
  - Bounding box coordinates: $(x_1, y_1, x_2, y_2)$ along with original canvas dimensions $(w, h)$.
  - Mask payload: context-adaptive range coded string starting with `!` (`one_bit`).
  - Dimensions: mask width and height.
- Decoded via Meta's `meta-sam-parser` library and pasted into canvas coordinate space via `place_box_mask` in [`neev/measurements/masks.py`](file:///d:/projects/metaglobal/neev/measurements/masks.py).
- Undecodable payloads raise `MaskError` and are never silently fabricated or treated as empty.

---

## 2. Audit of Existing Inference Cache & Historical Predictions

### 2.1 File-Based RunStore Structure
Neev maintains an immutable file-based run store under `runs/`:
- `runs/<run_id>/request.json`: Trace-safe sanitized request (base64 image replaced with SHA-256 hash).
- `runs/<run_id>/response.json`: Raw model response verbatim.
- `runs/<run_id>/meta.json`: Timestamps, model identifiers, prompt hashes, and telemetry notes.
- `runs/cache.json`: Maps `sha256_json(sanitized_request)` to `<run_id>` for exact-repeat queries.

### 2.2 Historical Cache Coverage
- Historical cache contains 586 total requests across earlier experiments (`DEV-10`, `VAL-19`, `photo_Sample`, and visual prompt studies).
- **CUBIT-Seg Overlap Audit:** Exactly **0 out of 1,320 CUBIT-Seg images** existed in the historical cache prior to this benchmark. Every CUBIT image represents completely unseen external UAV concrete facade data.

---

## 3. Investigation: The Identical Prediction Issue Among Local Architectures

In the initial offline CUBIT benchmark, three architectures (`hybrid`, `unet_standalone`, and `hybrid_verified`) produced identical scores ($P=63.7\%$, $R=28.1\%$, $F_1=0.3902$). We audited the underlying implementation in [`neev/detection/engine.py`](file:///d:/projects/metaglobal/neev/detection/engine.py):

```python
# Lines 444-463 in neev/detection/engine.py:
if self.mode in ("unet_standalone", "hybrid", "hybrid_verified"):
    unet_boxes = self._run_unet_detection(np.array(im_pil))
    if self.mode == "unet_standalone":
        candidates = unet_boxes
    else:
        fused = list(unet_boxes)
        for sb in candidates:  # candidates comes from SAM whole/tile boxes
            is_covered = any(aabb_iou(sb, ub) >= self.nms_iou_threshold for ub in unet_boxes)
            if not is_covered:
                fused.append(sb)
        candidates = fused
```

### Root Cause Analysis:
1. **Empty SAM Candidate Input:** Under the strict $0.00 spend constraint during the initial run, SAM inference was not executed live, and no pre-existing cached runs existed for CUBIT images. Consequently, `candidates` (the SAM proposal list) was completely empty (`[]`).
2. **Degeneracy of Fusion:** When `candidates == []`, the fusion step `fused = list(unet_boxes) + []` yields purely `unet_boxes`.
3. **Offline Verification Inactive:** In `hybrid_verified`, offline verification mode (`force_offline=True`) retained all proposals because cloud-based Muse Wonder verification was disabled to prevent API costs.
4. **Conclusion:** The three architectures are distinct algorithmic pipelines in the engine. Their identical scores in the previous benchmark was a direct mathematical consequence of SAM proposals being empty when no live API calls were executed. When live SAM 3.1 inference runs, `hybrid` will fuse actual SAM detections with U-Net proposals, and `hybrid_verified` will filter the combined candidate pool.

---

## 4. Evaluation Protocol Audit: Cardinality-First Bipartite Matching

The evaluation framework in [`neev/eval/bipartite_matching.py`](file:///d:/projects/metaglobal/neev/eval/bipartite_matching.py) was audited against mathematical standards:
1. **Cardinality-First Guarantee:** Rather than unconstrained weight-sum maximization, the bipartite cost matrix is augmented with a large constant $K = \max(M, N) + 10.0$ for all edges where $\text{Coverage} \ge 0.50$ (`cov50`). This mathematically proves that any assignment with $c+1$ valid matches strictly outranks any assignment with $c$ matches, eliminating sub-threshold trade-offs.
2. **One-to-One Non-Duplicate Invariant:** One prediction box cannot match multiple ground truth annotations, and one ground truth annotation cannot match multiple predictions. Unmatched predictions become False Positives; unmatched ground truths become False Negatives.
3. **Pixel-Level Mask Decoupling:** In addition to bounding box bipartite matching, decoded prediction masks are compared against ground truth binary raster masks to compute true pixel-level Mask IoU, Dice coefficient, Mask Precision, and Mask Recall.

---

## 5. Summary of Audit Status

| Component | Status | Notes |
|:---|:---:|:---|
| **API Key & Auth** | VERIFIED | `MODEL_API_KEY` loaded, credentials tested successfully |
| **API Endpoint** | VERIFIED | `POST https://api.meta.ai/v1/responses` active and responding |
| **Model ID** | VERIFIED | `sam-3.1` wire model recognized |
| **Parsing Pipeline** | VERIFIED | Bounding box & `meta-sam-parser` range decoder verified |
| **RunStore & Cache** | VERIFIED | Dual-level cache (`runs/<id>/` + `cubit_sam31_cache.json`) active |
| **Exponential Backoff**| VERIFIED | Jittered backoff on 429/5xx implemented in runner |
| **Matching Protocol** | VERIFIED | Cardinality-first bipartite matching ($T=0.50$) ready |
