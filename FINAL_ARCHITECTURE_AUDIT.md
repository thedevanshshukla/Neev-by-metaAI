# Final Architecture Audit: Neev Defect Detection & Segmentation Core

> **Audit Date:** November 4, 2026  
> **Status:** ARCHITECTURE FROZEN FOR PRODUCTION  
> **Primary Detection Engine:** Meta SAM 3.1 (`sam-3.1`, Baseline Mode)  
> **Secondary High-Recall Mode:** SAM + Crack-Seg U-Net Hybrid (`hybrid` Mode)  
> **Offline Edge Fallback:** Standalone Crack-Seg U-Net (`unet_standalone` Mode)  
> **Semantic Verification Layer:** Muse Wonder (`muse-wonder-1.0`, Shadow/Advisory Mode)  

---

## 1. Executive Summary & Freeze Confirmation

Following extensive comparative benchmarking across 1,320 CUBIT-Seg facade defect images (2,357 annotated ground-truth cracks, evaluated under strict cardinality-first bipartite matching with $\text{IoU} \ge 0.50$, `cov50`), the detection architecture for **Neev** is definitively **frozen**.

Further architecture search and speculative algorithmic exploration are terminated. Engineering focus is transitioned strictly to production reliability, backend resilience, API stability, and UI/UX integration.

---

## 2. Definitive Architecture Selection & Operational Roles

| Tier | Architecture Mode | Model Identifier | Primary Metric (Test F1) | Precision | Recall | Pixel Mask IoU | Mean Latency | Unit API Cost | Verification Status |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|:---:|
| **Primary Production** | **`baseline`** (Meta SAM 3.1) | `sam-3.1` (prompt: `"crack"`) | **0.7732** | **80.3%** | 74.6% | **0.5248** | 1,454 ms | $0.0025 USD | **VERIFIED** |
| **Secondary High-Recall** | **`hybrid`** (SAM + U-Net) | `sam-3.1` + `crack_unet` | **0.7374** | 71.4% | **76.3%** | 0.1282 | 1,550 ms | $0.0025 USD | **VERIFIED** |
| **Offline Edge Fallback** | **`unet_standalone`** | `crack_unet` (v1) | **0.3902** | 63.7% | 28.1% | 0.0844 | **124 ms** | **$0.0000 (Free)** | **VERIFIED** |
| **Semantic Verifier** | **`muse-wonder`** (Shadow) | `muse-wonder-1.0` | N/A (Advisory) | 100.0% Spec. | 100.0% Sens. | N/A | 11.88 s | ~$0.028 USD / call | **VERIFIED** |

### A. Primary Production Architecture: Live Meta SAM 3.1 (`baseline`)
- **Role:** Default detection and pixel segmentation engine for all cloud-connected visual inspections.
- **Performance Rationale:** Achieved **80.3% Precision** and **74.6% Recall** (F1: **0.7732**), capturing 883 true positive cracks across 683 holdout test images.
- **Segmentation Quality:** Delivered **0.5248 Pixel Mask IoU** and **0.6559 Dice**, accurately delineating fine fracture contours.
- **False Alarm Suppression:** Exceptional specificity on pristine concrete walls, generating only **3 false positives across all clean facade test images** (0.32 FP/image overall).
- **Configuration:** Endpoint `https://api.meta.ai/v1/responses`, text prompt `"crack"`, wire grammar decoded via `meta-sam-parser` with canvas coordinate re-projection.

### B. Secondary Architecture: SAM + U-Net Hybrid (`hybrid`)
- **Role:** Optional high-recall inspection mode for critical infrastructure pre-screening.
- **Performance Rationale:** Achieved the highest absolute crack recovery in the benchmark (**76.3% Recall**, 903 true positive cracks), capturing localized micro-discontinuities missed by macro SAM.
- **Trade-off:** Precision dropped from 80.3% to **71.4%**, and false alarms increased to 0.53 FP/image.
- **Operational Rule:** Must remain an opt-in mode; never replaces SAM as the system default.

### C. Offline Edge Fallback: Standalone U-Net (`unet_standalone`)
- **Role:** Autonomous local execution when network connectivity is severed or drone edge processors operate offline.
- **Performance Profile:** Runs at **124 ms latency** with **$0.00 API cost**. Demonstrates robust precision (**63.7%**) and clean negative control (only 1 false alarm on clean walls).
- **Limitation:** Lower recall (**28.1%**) due to fixed receptive field. Inspection reports explicitly flag reduced sensitivity in offline mode.
- **Transparency Constraint:** Fallback must be explicitly logged. The system never silently executes U-Net and misrepresents the output as a successful SAM run.

---

## 3. Rationale for Architecture Rejections

The remaining 11 experimental configurations evaluated during earlier phases are permanently retired from the primary production path:

1. **Standalone SAHI (`sahi_standalone` - Ranked #14, F1: 0.0621):**
   - *Reason for Rejection:* Catastrophic false positive explosion (**9.41 FP/image**, Precision 3.7%, 15 false alarms on clean walls). Un-gated slicing hallucinates cracks across ordinary concrete texture.
2. **DSP Frangi / Top-Hat Proposals (`dsp_frangi_sam` - Ranked #5, F1: 0.7729):**
   - *Reason for Rejection:* Classical Hessian eigenvalue filters fail on rough aggregate concrete and add computational overhead without improving metrics over base SAM (Precision 80.2% vs 80.3%).
3. **Computer Vision Illumination Enhancement (`cv_enhanced_sam` - Ranked #6, F1: 0.7729):**
   - *Reason for Rejection:* Multi-scale Retinex and CLAHE normalization amplify surface pores and concrete aggregate boundaries, increasing false positives in shadowed scenes.
4. **Specialist SegFormer Transformer (`multidefect_segformer` - Ranked #11, F1: 0.1628):**
   - *Reason for Rejection:* High sensitivity (33.5% recall) but unacceptable false positive burden (**4.81 FP/image**, Precision 10.8%, 26 clean wall alarms).
5. **Cascaded SAHI (`sahi_cascaded` - Ranked #12, F1: 0.1390):**
   - *Reason for Rejection:* Gating stages over-suppressed candidate proposals (Recall dropped to 9.3%) while adding ~1,073 ms processing latency.
6. **Adaptive Precision-Gated Hybrid (`hybrid_precision_gated` - Ranked #10, F1: 0.2754):**
   - *Reason for Rejection:* Tight aspect ratio and area filters discarded true branching cracks, depressing recall to 18.1%.

> [!NOTE]
> All rejected architecture implementations are retained in `neev/detection/engine.py` and `neev/inference/` for historical auditability and regression verification, but are excluded from active user-facing workflows.

---

## 4. Architectural Transparency & Provenance Invariants

To eliminate any ambiguity between cloud foundation models and local edge fallbacks, every inspection execution strictly enforces the following audit invariants:

1. **Immutable Execution Records:**
   - Every inspection output records:
     - `selected_architecture`: The requested engine mode.
     - `actual_architecture_executed`: The exact code path and model invoked.
     - `model_id` and `model_version`: e.g. `sam-3.1` / `v3.1`.
     - `inference_status`: `"success"`, `"failed"`, or `"fallback"`.
     - `fallback_reason`: Documented reason when fallback triggers.
     - `cache_usage`: `"cached"`, `"live_api"`, or `"offline_local"`.
     - `api_request_id`: Meta request ID (`resp_...`) where available.
2. **Failure Integrity:**
   - Network timeouts, rate limits, or API errors are recorded as `status="failed"`. A failed API request is NEVER represented as an empty successful detection.
   - Fallback to `unet_standalone` requires explicit configuration (`allow_offline_fallback=True`) and records an explicit audit trail.

