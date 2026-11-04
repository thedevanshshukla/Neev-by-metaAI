# SAM 3.1 Live Pilot Report: CUBIT-Seg 30-Image Evaluation

> **Evaluation Mode:** Live Meta Model API (`sam-3.1`, text prompt: `"crack"`)  
> **Evaluation Protocol:** Strict Cardinality-First Bipartite Matching ($	ext{IoU} \ge 0.50$, `cov50`)  
> **Official Meta API Rate:** **$2.50 / 1,000 images** ($0.0025 USD / image call)  
> **Pilot Sample Size:** **30 representative CUBIT-Seg images** (DEV & VAL partitions; TEST partition strictly preserved)  
> **Pilot Actual Expenditure:** **$0.0750 USD** (30 live requests, 0 cache hits)  
> **Mean Live Latency:** **1.78 seconds / request**  

---

## 1. Executive Summary & Verification Findings

We executed actual, live SAM 3.1 inference through the Meta Model API Responses endpoint (`https://api.meta.ai/v1/responses`) across a 30-image diverse pilot sample.

### Key Verification Checkpoints
1. **API Endpoint & Connectivity:** Functional and verified. The HTTPS endpoint responded with HTTP 200 for all requests.
2. **Model Identifier:** Official Meta model identifier `"sam-3.1"` is verified and active.
3. **Resolution Acceptance:** CUBIT-Seg images at native $512 \times 512$ resolution were accepted and processed without scaling rejections.
4. **Wire Grammar & Parsing:** Meta token grammar (`<0f>0<|box;x1=...;y1=...;w=512;h=512|><|mask;x=0;y=0;data=...|>`) parsed cleanly into `BoundingBox` and decoded via `meta-sam-parser` with zero coordinate shifts.
5. **Mask Decoding & Coordinate Transform:** Decoded mask rasters placed accurately onto the $512 \times 512$ canvas via `place_box_mask`, enabling true pixel-level mask IoU and Dice evaluation.
6. **Billing & Telemetry:** Exact request IDs (`resp_...`) and usage objects (`{"images_processed": 1}`) were captured and logged.
7. **Cache & Deduplication:** RunStore (`runs/<run_id>/`) and persistent CUBIT cache (`data/cubit_sam31_cache.json`) prevented duplicate billing.

---

## 2. Overall Pilot Performance vs Baseline Local Architectures

| Metric | Live SAM 3.1 Pilot (30 Images) | U-Net Standalone / Hybrid (CUBIT Full) | Multi-Defect SegFormer (CUBIT Full) |
|:---|:---:|:---:|:---:|
| **Detection Precision** | **86.8%** | 63.7% | 10.8% |
| **Detection Recall** | **40.7%** (33/81 GTs) | 28.1% | 33.5% |
| **Detection F1 Score** | **0.5546** | 0.3902 | 0.1628 |
| **Pixel Mask IoU** | **0.3032** | 0.0844 | 0.1453 |
| **Pixel Mask Dice** | **0.4168** | 0.1120 | 0.1840 |
| **False Positives / Image** | **0.17** (5 FPs / 30 imgs) | 0.28 | 4.81 |
| **Negative Facade False Alarms**| **0** (4 clean facades tested) | 1 | 26 |
| **Mean Inference Latency** | **1.78 s** | 122.9 ms | 62.0 ms |
| **API Cost / Image** | **$0.0025 USD** | $0.0000 (Free) | $0.0000 (Free) |

---

## 3. Performance Across Challenge Categories (30 Images)

| Category | Images | GT Cracks | Preds | TP | FP | FN | Precision | Recall | F1 | Mask IoU | FP / Image |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Negative Clean Facade** | 4 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% | **0.0000** | 0.0000 | 0.00 |
| **Thin Hairline Crack** | 5 | 9 | 3 | 3 | 0 | 6 | 100.0% | 33.3% | **0.5000** | 0.1701 | 0.00 |
| **Complex Multi Crack** | 5 | 54 | 13 | 13 | 0 | 41 | 100.0% | 24.1% | **0.3881** | 0.5164 | 0.00 |
| **Shadowed Surface** | 5 | 6 | 6 | 5 | 1 | 1 | 83.3% | 83.3% | **0.8333** | 0.3920 | 0.20 |
| **Difficult Texture** | 5 | 5 | 8 | 5 | 3 | 0 | 62.5% | 100.0% | **0.7692** | 0.4413 | 0.60 |
| **Medium Wide Crack** | 6 | 7 | 8 | 7 | 1 | 0 | 87.5% | 100.0% | **0.9333** | 0.2495 | 0.17 |

---

## 4. Cost and Approval Gate (Mandatory Stop)

The official Meta Model API billing rate for SAM 3.1 is verified at **$2.50 per 1,000 images** ($0.0025 per image request).

### Projected Full Evaluation Cost Matrix

| Dataset Partition | Total Images | Cached | Uncached Requests | Cost Rate | Estimated Total Cost (USD) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **30-Image Pilot (Completed)** | 30 | 0 | 30 | $0.0025 / img | **$0.0750** |
| **DEV Partition** | 264 | 18 | 246 | $0.0025 / img | **$0.6150** |
| **VAL Partition** | 373 | 12 | 361 | $0.0025 / img | **$0.9025** |
| **TEST Partition (Holdout)** | 683 | 0 | 683 | $0.0025 / img | **$1.7075** |
| **Total Full CUBIT-Seg Benchmark** | **1,320** | **30** | **1,290** | $0.0025 / img | **$3.2250 USD** |

> [!IMPORTANT]
> **Total Remaining Spend to Evaluate All 1,320 Images:** **$3.2250 USD** (~$3.23).  
> The entire remaining dataset can be evaluated for less than three dollars and twenty-three cents.
> Per instructions, execution is stopped at this gate awaiting explicit user authorization.

---

## 5. Architectural Comparison Insights from Pilot Data

1. **Precision & False Alarms:**
   - SAM 3.1 exhibits high precision (86.8%) on clear fracture lines, matching its performance on historical DEV-10 / VAL-19.
   - On **clean negative facades**, SAM 3.1 produced **0 false alarms** across 4 intact concrete walls, confirming that its text prompt baseline does not hallucinate cracks on plain concrete.
2. **Hairline & Low-Contrast Sensitivity:**
   - On hairline micro-cracks ($w \le 3.5\text{px}$), SAM 3.1 misses subtle hairline fractures (recall 33.3%), whereas local U-Net edge models capture them more reliably.
3. **Complex Networks:**
   - On branching/complex multi-crack concrete scenes, SAM 3.1 often groups adjacent crack branches into larger individual bounding boxes, achieving solid coverage but lower one-to-one cardinality under strict $\text{IoU} \ge 0.50$ matching.
4. **Latency Comparison:**
   - SAM 3.1 cloud API latency averaged **1.78 seconds per image**, whereas local U-Net runs on-device in **0.12 seconds** (over 25× faster).

---

## 6. Artifact Links

- Machine-Readable Pilot JSON: [`data/cubit_sam_pilot_results.json`](file:///d:/projects/metaglobal/data/cubit_sam_pilot_results.json)
- Per-Image Metrics CSV: [`data/cubit_sam_pilot_per_image.csv`](file:///d:/projects/metaglobal/data/cubit_sam_pilot_per_image.csv)
- Visual Overlay Samples: [`CUBIT_PILOT_VISUALS/`](file:///d:/projects/metaglobal/CUBIT_PILOT_VISUALS)
- Persistent Inference Cache: [`data/cubit_sam31_cache.json`](file:///d:/projects/metaglobal/data/cubit_sam31_cache.json)
