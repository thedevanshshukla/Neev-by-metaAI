# Muse Wonder Integration & Semantic Verification Audit Report

> **Model Identifier:** `muse-wonder-1.0`  
> **API Protocol:** Meta Model API Responses Endpoint (`https://api.meta.ai/v1/responses`)  
> **Operational Role:** Optional Semantic Verification & Contextual Interpretation Layer  
> **Integration Status:** **VERIFIED WITH LIVE API CALLS**  
> **Controlled Validation Spend:** **$0.1405 USD** (5 Live Calls, $2.00 Ceiling Respected)  
> **Default Production Mode:** `shadow`  

---

## 1. Architectural Scope & Operational Boundaries

Muse Wonder acts exclusively as a contextual reasoning and visual semantic verifier. It is **NOT** a primary defect detector. Segmentation, bounding box localization, and pixel geometry are governed deterministically by SAM 3.1 and Crack-Seg U-Net.

### Strict Operational Boundaries:
1. **No Coordinate Invention:** Muse Wonder never invents bounding box coordinates or centroid locations.
2. **No Mask Alteration:** Muse Wonder never mutates, smooths, or rasterizes segmentation masks.
3. **No Silent Finding Deletion:** Muse Wonder cannot delete or hide a SAM candidate. Findings flagged as doubtful are preserved with explicit advisory provenance for human engineering review.
4. **No Structural Integrity Certification:** The model is prohibited by prompt instructions from claiming building safety, structural stability, or seismic compliance.
5. **Deterministic Precedence:** Morphological geometry (area, length, width, tortuosity) remains purely mathematical and is not overridden by LLM prose.

---

## 2. API Protocol, Schemas & Wire Verification

### A. Endpoint & Authentication
- **Endpoint:** `POST https://api.meta.ai/v1/responses`
- **Headers:** `Authorization: Bearer <MODEL_API_KEY>`, `Content-Type: application/json`
- **Model String:** `"muse-wonder-1.0"`
- **Wire Input Format:** Supports multi-modal content arrays combining JSON feature text with base64 data URIs:
  ```json
  {
    "model": "muse-wonder-1.0",
    "instructions": "...",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "{\"candidate_id\": \"...\", \"features\": {...}}"},
          {"type": "input_image", "image_url": "data:image/jpeg;base64,..."}
        ]
      }
    ]
  }
  ```

### B. Strict Structured Response Schema
Every live and offline verification output validates against the following normalized schema:
```json
{
  "candidate_id": "string",
  "visual_class": "crack | non_crack | uncertain",
  "visual_evidence": "string",
  "contextual_observations": "string",
  "possible_confounders": ["string"],
  "verification_confidence": 0.0,
  "reasoning_summary": "string",
  "limitations": "string | null",
  "is_crack": true,
  "defect_type": "string",
  "advisory_status": "supports | doubts | inconclusive | skipped | failed",
  "status": "live_verified | cached_verified | offline_heuristic | skipped_budget",
  "token_usage": {"input_tokens": 0, "output_tokens": 0},
  "cost_usd": 0.0
}
```

---

## 3. Selective Verification Policy & Budget Management

To prevent unbounded API cost growth, candidate verification is governed by a selective priority filter:
1. **Per-Inspection Ceiling:** A maximum of **10 candidates** are submitted to Wonder per inspection.
2. **Candidate Priority Ranking:**
   - Priority 1: Candidates with intermediate SAM confidence ($0.40 \le c \le 0.75$).
   - Priority 2: Candidates in shadowed surface regions ($> 4\%$ shadow mask coverage).
   - Priority 3: Candidates near facade boundaries, window openings, or architectural edges.
3. **Session Spend Ceiling:** A hard stop at **$2.00 USD total**. When the spend ceiling is reached, live calls halt automatically, and subsequent candidates are assigned `status="skipped_budget"`.
4. **Token Pricing Model:**
   - Input Tokens: **$4.00 per million tokens** ($0.000004 / token).
   - Output Tokens: **$20.00 per million tokens** ($0.000020 / token).
   - Cost Formula: $\text{Cost} = \frac{N_{\text{in}} \times 4.0 + N_{\text{out}} \times 20.0}{1,000,000}$.

---

## 4. Live Meta API Validation Experiment Results

A controlled 5-candidate live experiment was conducted on representative facade conditions using the live Meta Model API (`muse-wonder-1.0`).

### Empirical Case Results:
| Case ID | Feature Type | Ground Truth | Visual Class | Advisory | Confidence | Latency | Tokens (In / Out) | Cost (USD) |
|:---|:---|:---:|:---:|:---:|---:|---:|:---:|---:|
| `genuine_crack` | Tensile shear fracture | Crack | `crack` | **SUPPORTS** | 0.93 | 9.33 s | 637 / 1,089 | $0.024328 |
| `branching_crack` | Bifurcating concrete fracture | Crack | `crack` | **SUPPORTS** | 0.93 | 11.15 s | 635 / 1,197 | $0.026480 |
| `shadowed_crack` | Fracture in heavy shadow | Crack | `crack` | **SUPPORTS** | 0.90 | 14.34 s | 635 / 1,396 | $0.030460 |
| `window_mullion` | Rectilinear aluminum frame | Non-Crack | `non_crack` | **DOUBTS** | 0.98 | 14.32 s | 635 / 1,424 | $0.031020 |
| `surface_stain` | Rounded moisture discoloration | Non-Crack | `non_crack` | **DOUBTS** | 0.95 | 10.27 s | 633 / 1,282 | $0.028172 |

### Summary Telemetry:
- **Total Live Calls:** 5
- **Total Input Tokens:** 3,175
- **Total Output Tokens:** 6,388 (including ~5,400 internal reasoning tokens)
- **Total Experiment Spend:** **$0.1405 USD** (strictly within the $2.00 ceiling)
- **Mean Latency:** **11.88 seconds / call**
- **Sensitivity (Crack Recall):** **100.0%** (3/3 true cracks supported)
- **Specificity (Non-Crack Rejection):** **100.0%** (2/2 architectural confounders doubted)
- **False Rejection Rate:** **0.0%** (0 genuine cracks falsely rejected)

---

## 5. Production Modes & Recommendations

1. **`disabled` Mode:** Bypasses Wonder entirely; zero network calls or billing.
2. **`shadow` Mode (Production Default):** Calls Wonder selectively, persists reasoning in the inspection audit log, but does not alter finding acceptance. Recommended for ongoing deployment.
3. **`advisory` Mode:** Surfaces Wonder reasoning in the web review dashboard and generates flagged review cards for conflicting findings without mutating raw detections.

