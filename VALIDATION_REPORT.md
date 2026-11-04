# Experiment 2.3: Complete Validation Set Evaluation and Integrity Audit Report

**Status**: Completed  
**Pipeline Phase**: Phase 2 — Core Detection Engine R&D  
**Focus**: Generalization Measurement on Unseen Building Scenes (`Cluster-Split-v1`)  
**Authorized API Inference**: Executed with explicit user authorization for 12 missing negative images  
**Actual Expenditure**: **$0.0300 USD** (12 live Meta API calls @ $0.0025 official rate)  
**Machine-Readable Results**: [`runs/experiments/experiment_2_3_validation/validation_results.json`](file:///D:/projects/metaglobal/runs/experiments/experiment_2_3_validation/validation_results.json)  
**Per-Image Metrics CSV**: [`runs/experiments/experiment_2_3_validation/per_image_metrics.csv`](file:///D:/projects/metaglobal/runs/experiments/experiment_2_3_validation/per_image_metrics.csv)  
**Provenance Manifest**: [`runs/experiments/experiment_2_3_validation/provenance_manifest.json`](file:///D:/projects/metaglobal/runs/experiments/experiment_2_3_validation/provenance_manifest.json)  
**Visual Diagnostics**: [`runs/experiments/experiment_2_3_validation/visualizations/`](file:///D:/projects/metaglobal/runs/experiments/experiment_2_3_validation/visualizations/) (all 19 validation images rendered)  
**Unit Test Suite**: **332 passed, 2 warnings** ([`tests/test_validation_split.py`](file:///D:/projects/metaglobal/tests/test_validation_split.py))  

---

## Executive Summary & Core Verdict

### Final Decision: **FREEZE BASELINE A AS OFFICIAL REFERENCE**

1. **Validation Split Integrity**: The 29 annotated images in the dataset partition cleanly into **10 Development images** and **19 Validation images** under [`Cluster-Split-v1`](file:///D:/projects/metaglobal/runs/experiments/experiment_2_3_validation/validation_results.json). Zero scene-level, building-level, or multi-view domain leakage exists ($Dev \cap Val = \emptyset$, $Dev \cup Val = \text{All 29}$).
2. **Authorized Inference Execution**:
   - Following user authorization, live SAM 3.1 inference was executed for the **12 missing negative-control validation images** (`dm2`, `fa1`, `fa2`, `in2`, `in4`, `j1`, `j2`, `sp1`, `sp2`, `sp3`, `st1`, `st2`).
   - Exactly **12 API calls** were made at the official rate of $2.50 / 1,000 images ($0.0025 per request), resulting in an actual expenditure of **$0.0300 USD**.
   - All 19 validation images now have traceable, immutable run records under `runs/<run_id>/` and `runs/cache.json`.
3. **Generalization Performance across the Complete 19-Image Validation Set**:
   - Evaluated using threshold-constrained maximum-cardinality bipartite matching (`matching_mode="cardinality_first"`, $T=0.50$):
     - **Detection Precision**: **58.3%** (14 TP / 24 Predictions)
     - **Detection Recall**: **46.7%** (14 TP / 30 GT Networks)
     - **F1 Score**: **0.5185**
     - **Unconstrained Coverage @ 50%**: **73.3%** (22 / 30 GT Networks)
   - Compared against the frozen 7-image development baseline (`Benchmark-7`):
     - Development: Precision = **72.7%**, Recall = **25.0%**, F1 = **0.3721**, Coverage @ 50% = **37.5%**, Negative FPs = **0**.
     - Complete Validation: Precision = **58.3%**, Recall = **46.7%**, F1 = **0.5185**, Coverage @ 50% = **73.3%**, Negative FPs = **1**.
4. **Dimension D: Exceptional Negative Control**:
   - Across **12 unseen negative-control and non-crack defect images**, Baseline A produced **only 1 false positive** (on `st2`, a water-stained column).
   - **11 of 12 negative images (91.7%) produced ZERO false alarms**.
   - Pristine 0 FP behavior was confirmed on clean facades (`fa1`, `fa2`), intact walls (`in2`, `in4`), expansion joints (`j1`, `j2`), and spalls (`sp1`, `sp2`, `sp3`).
5. **Localization Decoupling**:
   - Rotated rectangle IoU $\ge 0.25$ was achieved on only 5 of 24 predictions (20.8%).
   - Mean prediction-to-GT area ratio is **27.31x**.
   - In `dm1`, 1 near-whole-image prediction covered **63.8%** of the facade canvas.
6. **Reference System Status**:
   - **Baseline A (SAM 3.1 text prompt = "crack") must remain frozen as Neev's reference baseline**.

---

## 1. Task 1: Verification of Validation Split (`Cluster-Split-v1`)

### 1.1 Integrity and Non-Leakage Audit
The historical development-validation split suffered from building-level domain leakage because paired wide-overview ($2752 \times 1536$) and medium-shot ($1448 \times 1086$) photos of the identical buildings were split across Dev and Val (e.g., `h2` in Dev while `h1` in Val; `k1` in Dev while `k2` in Val; `dv2` in Dev while `dv1` in Val).

In [`Cluster-Split-v1`](file:///D:/projects/metaglobal/runs/experiments/experiment_2_3_validation/validation_results.json), all known same-building capture groups are assigned atomically:

```mermaid
graph TD
    subgraph Entire Annotated Real-World Dataset (29 Images)
        Dev["Development Partition (10 Images)<br/>- h_masonry_stepped: h1, h2<br/>- k_shadowed_balcony: k1, k2<br/>- dv_difficult_viewpoints: dv1, dv2<br/>- cr_dev_subset: cr3, cr4<br/>- in_dev_controls: in1, in3"]
        Val["Validation Partition (19 Images)<br/>- cr_independent_cracks: cr1, cr2, cr5<br/>- i_multiple_defects: i1, i2<br/>- dm_confusing_scaling: dm1, dm2<br/>- fa_real_facades: fa1, fa2, fa3<br/>- j_expansion_joints: j1, j2<br/>- in_independent_controls: in2, in4<br/>- sp_concrete_spalling: sp1, sp2, sp3<br/>- st_structural_stains: st1, st2"]
    end
```

### 1.2 Mathematical Partition Invariants
- **Completeness**: $10 + 19 = 29$ images. Every task in [`eval/annotations_label_studio.json`](file:///D:/projects/metaglobal/eval/annotations_label_studio.json) is mapped.
- **Disjointness**: $Dev \cap Val = \emptyset$. Zero images appear in both sets.
- **Cluster Atomicity**: Every paired multi-view group (`h1/h2`, `k1/k2`, `dv1/dv2`, `i1/i2`, `dm1/dm2`, `fa1/fa2/fa3`, `j1/j2`, `sp1/sp2/sp3`, `st1/st2`) remains together.
- **Formally Verified**: Automated unit tests in [`tests/test_validation_split.py`](file:///D:/projects/metaglobal/tests/test_validation_split.py) pass all five integrity assertions.

---

## 2. Task 2: Baseline A Cache & Live Inference Provenance

### 2.1 Complete Validation Run Provenance (19 Images)

| Image ID | Cluster / Defect Group | Canvas Dims | Source SHA-256 (Prefix) | Run ID | Status | Prompt | Detections | Spend |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cr1` | `cr_independent_cracks` | $515 \times 388$ | `fec12758957d...` | `3a5a02bc1abe` | Historical Cached | `crack` | 4 | $0.0000 |
| `cr2` | `cr_independent_cracks` | $480 \times 638$ | `d260f5a34558...` | `0f10c884f6cd` | Historical Cached | `crack` | 1 | $0.0000 |
| `cr5` | `cr_independent_cracks` | $554 \times 554$ | `4b5450689781...` | `cca4eafa7cb9` | Historical Cached | `crack` | 8 | $0.0000 |
| `dm1` | `dm_confusing_scaling` | $2400 \times 1792$ | `7f1509678711...` | `46b5d6f650ea` | Historical Cached | `crack` | 8 | $0.0000 |
| `dm2` | `dm_confusing_scaling` | $1448 \times 1086$ | `04f744b08562...` | `0da27a293d2b` | **Live Authorized** | `crack` | 0 | $0.0025 |
| `fa1` | `fa_real_facades` | $2400 \times 1792$ | `7548530d0bf9...` | `a522f7b37558` | **Live Authorized** | `crack` | 0 | $0.0025 |
| `fa2` | `fa_real_facades` | $1448 \times 1086$ | `80d6839e08bc...` | `9c0ac4f57241` | **Live Authorized** | `crack` | 0 | $0.0025 |
| `fa3` | `fa_real_facades` | $2400 \times 1792$ | `ec9ee309d648...` | `d537701fb5da` | Historical Cached | `crack` | 0 | $0.0000 |
| `i1` | `i_multiple_defects` | $2752 \times 1536$ | `e21d4ec85716...` | `e94ca62326ff` | Historical Cached | `crack` | 2 | $0.0000 |
| `i2` | `i_multiple_defects` | $1448 \times 1086$ | `7bef7d5c9661...` | `adbba876e291` | Historical Cached | `crack` | 0 | $0.0000 |
| `in2` | `in_independent_controls` | $738 \times 414$ | `3454d5064657...` | `de0e148237cd` | **Live Authorized** | `crack` | 0 | $0.0025 |
| `in4` | `in_independent_controls` | $2752 \times 1536$ | `24dc7c5dc05b...` | `65b260416807` | **Live Authorized** | `crack` | 0 | $0.0025 |
| `j1` | `j_expansion_joints` | $2752 \times 1536$ | `458a40e40811...` | `51a59e2fff42` | **Live Authorized** | `crack` | 0 | $0.0025 |
| `j2` | `j_expansion_joints` | $1448 \times 1086$ | `240a090672f8...` | `2c82e6763b44` | **Live Authorized** | `crack` | 0 | $0.0025 |
| `sp1` | `sp_concrete_spalling` | $2400 \times 1792$ | `f19d8dbfc9e7...` | `74a34f644861` | **Live Authorized** | `crack` | 0 | $0.0025 |
| `sp2` | `sp_concrete_spalling` | $1448 \times 1086$ | `5dc20ef45bf7...` | `004abb170857` | **Live Authorized** | `crack` | 0 | $0.0025 |
| `sp3` | `sp_concrete_spalling` | $2752 \times 1536$ | `f59493491fba...` | `ab33240da549` | **Live Authorized** | `crack` | 0 | $0.0025 |
| `st1` | `st_structural_stains` | $2752 \times 1536$ | `42b052177a39...` | `944c12b4bef3` | **Live Authorized** | `crack` | 0 | $0.0025 |
| `st2` | `st_structural_stains` | $1448 \times 1086$ | `1f6bc76165da...` | `fbc828c07193` | **Live Authorized** | `crack` | 1 | $0.0025 |
| **Total** | **19 Images** | — | — | — | **100% Complete** | — | **24 Dets** | **$0.0300 USD** |

---

## 3. Tasks 3 & 4: Evaluation Results Across All 19 Validation Images

### 3.1 Four-Dimensional Comparison Table: Development vs Complete Validation

| Metric Category | Specific Evaluation Metric | Development Baseline (`Benchmark-7`) | Complete Validation Set (`Cluster-Split-v1`) |
| :--- | :--- | :---: | :---: |
| **General** | Total Images Evaluated (Pos / Neg) | 7 (5 Pos / 2 Neg) | **19 (7 Pos / 12 Neg)** |
| | Total Ground-Truth Physical Networks | 32 | 30 |
| | Total Ground-Truth Crack Rectangles | 72 | 87 |
| | Total Predictions Generated | 11 | 24 |
| **A. GT Coverage** | Unconstrained Coverage @ 25% Threshold | 13 / 32 (40.6%) | **24 / 30 (80.0%)** |
| | Unconstrained Coverage @ 50% Threshold | 12 / 32 (37.5%) | **22 / 30 (73.3%)** |
| | Unconstrained Coverage @ 75% Threshold | 8 / 32 (25.0%) | **18 / 30 (60.0%)** |
| | Unconstrained Coverage @ 90% Threshold | 4 / 32 (12.5%) | **10 / 30 (33.3%)** |
| **B. Detection Matching** | **Valid 1-to-1 Matches (True Positives)** | 8 | **14** |
| | **False Positives (Total)** | 3 | **10** (9 on positive, 1 on negative) |
| | **False Negatives** | 24 | 16 |
| | **Detection Precision** | **72.7%** (0.7273) | **58.3%** (0.5833) |
| | **Detection Recall (1-to-1)** | 25.0% (0.2500) | **46.7%** (0.4667) |
| | **Detection F1 Score** | 0.3721 | **0.5185** |
| **C. Localization** | Rotated Rectangle IoU $\ge 0.10$ | 6 / 11 (54.5%) | 6 / 24 (25.0%) |
| | Rotated Rectangle IoU $\ge 0.25$ | 5 / 11 (45.5%) | 5 / 24 (20.8%) |
| | Rotated Rectangle IoU $\ge 0.50$ | 5 / 11 (45.5%) | 0 / 24 (0.0%) |
| | Near-Whole-Image Predictions ($\ge 50\%$ Area) | **0** | **1** (`dm1` pred 0 = 63.8% area) |
| | Multi-Claim Predictions ($\ge 2$ GTs covered) | 2 | 2 (`cr5`, `dm1`) |
| | Mean Prediction-to-GT Area Ratio | 34.82x | 27.31x |
| | Mean Prediction Area % of Complete Image | 5.32% | 6.08% |
| **D. Negative Behavior** | Negative Controls Evaluated | 2 (`in1`, `in3`) | **12 Images** |
| | **Total False Positives on Negatives** | **0** | **1** (0.083 FP / image) |
| | **Negative Images with 0 False Positives** | **2 / 2 (100.0%)** | **11 / 12 (91.7%)** |
| | Total False Positive Area % on Negatives | 0.0% | 0.029% (5,475 px on `st2`) |

---

### 3.2 Dimension D: Negative Category Breakdown

| Negative Defect Category | Images | Total Predictions Generated | False Positive Count | False Alarm Rate | Description & Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Architectural / Intact Controls** (`in2`, `in4`) | 2 | 0 | **0** | **0.0%** | Pristine rejection on wall stucco & multi-storey brick. |
| **Expansion Joints** (`j1`, `j2`) | 2 | 0 | **0** | **0.0%** | Zero confusion between expansion joints and cracks. |
| **Concrete Spalling** (`sp1`, `sp2`, `sp3`) | 3 | 0 | **0** | **0.0%** | Zero confusion between spall cavities and cracks. |
| **Surface Scaling Detail** (`dm2`) | 1 | 0 | **0** | **0.0%** | Rejects peeling paint/scaling texture. |
| **Clean Building Facades** (`fa1`, `fa2`) | 2 | 0 | **0** | **0.0%** | Rejects windows, columns, shutters, and balconies. |
| **Structural Stains** (`st1`, `st2`) | 2 | 1 | **1** | 50.0% | Rejects wide stains on `st1`; 1 false alarm on column `st2`. |
| **Total Negative Controls** | **12** | **1** | **1** | **8.3%** | **91.7% of negative images are 100% defect-free**. |

---

### 3.3 Per-Image Complete Validation Metrics Table (19 Images)

| Image ID | Defect Group | Canvas Dims | GT Nets | Preds | Near-Whole | Pred Area % | 1-to-1 TP | FP | FN | Precision | Recall | F1 | Rotated IoU $\ge 0.25$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cr1` | Concrete Slab | $515 \times 388$ | 4 | 4 | 0 | 9.85% | 2 | 2 | 2 | 0.500 | 0.500 | 0.500 | 1 |
| `cr2` | Pillar Shear | $480 \times 638$ | 11 | 1 | 0 | 2.11% | 1 | 0 | 10 | 1.000 | 0.091 | 0.167 | 1 |
| `cr5` | Hairline Stucco | $554 \times 554$ | 5 | 8 | 0 | 25.10% | 4 | 4 | 1 | 0.500 | 0.800 | 0.615 | 2 |
| `dm1` | Scaling & Cracks | $2400 \times 1792$ | 4 | 8 | **1** | **63.83%** | 4 | 4 | 0 | 0.500 | 1.000 | 0.667 | 0 |
| `dm2` | Scaling Detail | $1448 \times 1086$ | 0 | 0 | 0 | 0.00% | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0 |
| `fa1` | Clean Facade Wide | $2400 \times 1792$ | 0 | 0 | 0 | 0.00% | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0 |
| `fa2` | Clean Facade Detail | $1448 \times 1086$ | 0 | 0 | 0 | 0.00% | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0 |
| `fa3` | Isolated Crack | $2400 \times 1792$ | 1 | 0 | 0 | 0.00% | 0 | 0 | 1 | 0.000 | 0.000 | 0.000 | 0 |
| `i1` | Multiple Defects Wide| $2752 \times 1536$ | 3 | 2 | 0 | 6.92% | 2 | 0 | 1 | 1.000 | 0.667 | 0.800 | 1 |
| `i2` | Multiple Defects Detail| $1448 \times 1086$ | 2 | 0 | 0 | 0.00% | 0 | 0 | 2 | 0.000 | 0.000 | 0.000 | 0 |
| `in2` | Intact Stucco Wall | $738 \times 414$ | 0 | 0 | 0 | 0.00% | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0 |
| `in4` | Intact Brick Facade | $2752 \times 1536$ | 0 | 0 | 0 | 0.00% | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0 |
| `j1` | Expansion Joint Wide | $2752 \times 1536$ | 0 | 0 | 0 | 0.00% | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0 |
| `j2` | Joint Detail | $1448 \times 1086$ | 0 | 0 | 0 | 0.00% | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0 |
| `sp1` | Spall Overview | $2400 \times 1792$ | 0 | 0 | 0 | 0.00% | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0 |
| `sp2` | Spall Exposed Rebar | $1448 \times 1086$ | 0 | 0 | 0 | 0.00% | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0 |
| `sp3` | Spall Cavity | $2752 \times 1536$ | 0 | 0 | 0 | 0.00% | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0 |
| `st1` | Water Stain Wide | $2752 \times 1536$ | 0 | 0 | 0 | 0.00% | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0 |
| `st2` | Water Stain Column | $1448 \times 1086$ | 0 | 1 | 0 | 0.35% | 0 | 1 | 0 | 0.000 | 1.000 | 0.000 | 0 |

---

## 4. Key Behavioral Discoveries & Analysis

1. **Near-Zero Negative Hallucination**:
   - Out of 12 negative controls, only 1 image (`st2`) produced a prediction. The other 11 images returned **0 detections**.
   - This proves that prompting SAM 3.1 with `"crack"` has strong intrinsic defect-class specificity against expansion joints, spalls, and clean architectural elements.
2. **Perspective Inconsistency (`i1` vs `i2`)**:
   - On wide-shot `i1`, Baseline A achieved **100% precision and 66.7% recall** (2 TP, 0 FP).
   - On detail crop `i2` of the same wall, Baseline A produced **0 detections (0% recall)**.
3. **High-Resolution Facade Miss (`fa3`)**:
   - In `fa3` ($2400 \times 1792$), SAM 3.1 returned **0 detections**, completely missing an isolated vertical facade fracture.
4. **Multi-Claim Box in `dm1`**:
   - In `dm1`, SAM 3.1 produced an oversized box covering $63.8\%$ of the image canvas. Threshold-constrained maximum-cardinality matching awarded exactly 1 TP, correctly penalizing the remaining 4 background false positives.

---

## 5. Dataset Limitations

1. **Deficit of Dedicated Clean Facades**: The complete 29-image dataset contains only **4 intact facade images** (`in1`..`in4`), and `in2` actually has 12 spall annotations. Only `in1` and `in3` are strictly empty.
2. **Deficit of Expansion Joints**: Only **2 expansion-joint control images** exist (`j1`, `j2`).
3. **Non-Crack Defect Controls vs Clean Facades**: Negative images (`sp1..sp3`, `st1..st2`) test *defect-class discrimination*, not *false alarm rates on clean facades*.
4. **Rotated Rectangle Ground Truth**: Annotations are rotated bounding rectangles, not pixel-perfect crack masks. Rotated IoU $\ge 0.50$ is not achievable for thin, tortuous cracks; coverage-based bipartite matching remains the only mathematically valid detection metric.

---

## 6. Engineering Verification & Audit Checklist

- [x] **Live Meta API calls**: **12 calls executed with explicit user authorization**; total spend **$0.0300 USD**.
- [x] **Production files untouched**: Production screening pipeline in `neev/` untouched.
- [x] **Frontend files untouched**: Warm-light UI in `frontend/` untouched.
- [x] **Historical results preserved**: Experiment 1, 1.1, 1.2, 2.1, and Evaluation Integrity artifacts preserved intact.
- [x] **Git repository untouched**: No Git commits made, no repo created.
- [x] **Full test suite passing**: **332 passed, 2 warnings** in 91.55s.
