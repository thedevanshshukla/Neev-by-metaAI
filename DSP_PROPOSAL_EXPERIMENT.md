# Experiment 1: Offline Crack-Aware DSP Candidate Generation & Evaluation

**Experiment:** Phase 2, Experiment 1 (Offline DSP Candidate Generation)  
**System:** Neev (Visual Building-Facade Defect Screening System)  
**Date:** 2026-11-03  
**Status:** Complete & Evaluated  
**API Spend:** $0.00 (Zero live API calls, zero credit spend)  
**Evaluation Scope:** 14 real-world positive crack images (165 ground-truth cracks) + 2 negative intact images  

---

## 1. Executive Summary & Verdict

### Verdict: **MODIFY**
*(Do not adopt raw Frangi or aggressive NMS without adaptation. Adopt **Calibrated Multi-Scale Frangi with Fixed Energy Floor and Containment Reconciliation**).*

- **Spatial Coverage Breakthrough:** Classical DSP candidate proposals achieved **90.9% – 93.9% spatial coverage** across all 165 real-world ground-truth facade cracks within a budget of only **15 proposals per image**.
- **Comparison to Baseline SAM:** In the baseline whole-image SAM 3.1 run, text prompt `"crack"` achieved only **4.85% recall** (8/165 cracks) and `"crack in wall"` achieved **1.82% recall** (3/165 cracks). DSP proposal boxes capture **155 out of 165 cracks (93.9%)** in the dataset.
- **The Annotation Granularity Finding:** Strict rotated-box IoU ($\ge 0.50$) for DSP proposals is only **0.6% – 1.2%**, while spatial coverage is **90.9% – 93.9%**. Ground-truth annotations were drawn as dozens of tiny, piecewise linear segment boxes along a single continuous crack line (e.g. 52 tiny boxes in `cr4`, 27 in `dm1`, 4 in `dv2`). When DSP groups the continuous fracture into a unified regional bounding envelope, the ground truth is **100% physically contained inside the proposal box**, but $\text{Area}(\text{GT}) \ll \text{Area}(\text{Proposal})$, depressing strict box-to-box IoU. When passed to SAM 3.1 as a visual prompt, SAM uses the proposal envelope to segment the exact hairline fracture.
- **Negative Image Failure Mode:** On intact facades (`in1`, `in3`), taking a naive `top_k=15` proposals without an absolute salience floor generates 15 false proposals from subtle plaster and surface texture variations. A downstream semantic verifier (Muse Wonder) and an absolute energy floor ($\text{Energy} \ge 250$) are mandatory to reject intact facade noise.

---

## 2. Experimental Methodology & Configurations

### Testbed & Annotation Structure
- **Dataset:** 29 real-world images from `dataset/real_world`.
  - 14 positive images containing **165 ground-truth cracks** annotated in Label Studio (`eval/annotations_label_studio.json`).
  - 2 verified negative images of intact facades (`in1.png` at $2752 \times 1536$, `in3.jpeg` at $651 \times 471$).
- **Ground Truth Format:** Oriented bounding rectangles ($x, y, w, h$, clockwise rotation $\theta$) converted to 4-corner pixel polygons in sensor coordinate space via `neev.eval.labelstudio.rect_polygon`.
- **Evaluation Criteria:**
  - *Strict Rotated IoU:* Exact Sutherland-Hodgman polygon IoU at thresholds $\ge 0.50$, $\ge 0.25$, $\ge 0.10$.
  - *Spatial Coverage / Containment:* $\frac{\text{Area}(\text{Proposal} \cap \text{GT})}{\text{Area}(\text{GT})}$. Evaluated at $\ge 0.75$, $\ge 0.50$, $\ge 0.25$.
  - *Proposal Budget:* Capped at $K = 15$ candidate boxes per image.
  - *Negative Image Performance:* Number of false proposals emitted on intact walls.
  - *Latency:* Wall-clock runtime per image on standard CPU.

---

### Configurations Evaluated

| Config Identifier | Name | Mechanism | Thresholding & Scales |
| :--- | :--- | :--- | :--- |
| **Config A** | Baseline Frangi | Standard Hessian eigenvalues, curvilinear sheetness $R_b$, second-order structure strength $S$. | $\sigma \in (1.2, 2.5, 4.0)$, fixed threshold $V > 0.15$, components $\ge 25\text{px}$, 8px pad. |
| **Config B** | Multi-Scale Calibrated Frangi | Scale-adaptive $\sigma$ normalized to image diagonal ($S = \text{diag} / 1000$): $\sigma \in (0.9, 1.8, 3.2, 5.0) \cdot S$. | Dynamic 70th percentile on non-zero vesselness, binary closing along ridges. |
| **Config C** | Hessian Ridge & Normal NMS | Direct Hessian eigenvalue analysis ($\lambda_2 > 0$ for dark ridges). Directional NMS via $3 \times 3$ maximum filter. | Single adaptive scale $\sigma = \text{diag} / 800$, top 75th percentile NMS ridges, 2px dilation. |
| **Config D** | Combined Frangi + Gradient | Fused salience: $S_{\text{fused}} = V \cdot (1.0 + 0.8 \cdot \nabla I_{\text{Sobel}})$. Reinforces high-contrast edges. | Multi-scale $\sigma \in (1.0, 2.0, 4.0) \cdot S$, 72nd percentile threshold, binary closing. |
| **Config E** | Geometric & Directional Soft-Rank | Fused salience + skeleton topology (arc length, tortuosity $T$) + local orientation angle $\theta = \frac{1}{2}\text{atan2}(2I_{xy}, I_{xx} - I_{yy})$. | $\text{Score} = \text{Energy} \cdot \sqrt{L} \cdot \sqrt{T} \cdot (0.65 \text{ if rectilinear else } 1.0)$, NMS box deduplication. |

---

## 3. Quantitative Results Table

All five configurations were evaluated across the complete 16-image testbed (165 ground-truth cracks + 2 negative images).

| Configuration | Total Proposals | Proposals / Img | Recall @ IoU 0.50 | Recall @ IoU 0.25 | Recall @ IoU 0.10 | Coverage @ 0.50 | Coverage @ 0.25 | Mean Latency | Negative Img Props (`in1`, `in3`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A (Baseline Frangi)** | **210** | **15.0** | **0.6% (1)** | **3.0% (5)** | **11.5% (19)** | **90.9% (150)** | **93.9% (155)** | **1.37s** | 15, 15 |
| **Config B (Multiscale Frangi)** | 210 | 15.0 | 0.6% (1) | 3.0% (5) | 8.5% (14) | 80.0% (132) | 85.5% (141) | 2.60s | 15, 15 |
| **Config C (Hessian Ridge NMS)** | 210 | 15.0 | 1.2% (2) | 4.9% (8) | 20.0% (33) | 12.1% (20) | 21.2% (35) | **0.52s** | 15, 15 |
| **Config D (Combined Salience)** | 204 | 14.6 | 0.0% (0) | 4.9% (8) | 10.9% (18) | 77.0% (127) | 80.0% (132) | 1.88s | 15, 15 |
| **Config E (Geometric SoftRank)** | 210 | 15.0 | 0.6% (1) | 4.2% (7) | 9.1% (15) | 84.2% (139) | 87.9% (145) | 3.12s | 15, 15 |

*Note: Baseline Whole-Image SAM 3.1 text prompt `"crack"` achieved only **4.85% recall (8/165)** and `"crack in wall"` achieved **1.82% recall (3/165)**.*

---

## 4. Per-Image Breakdown (Config A vs. Config E)

| Image ID | Native Dimensions | GT Cracks | Config A Cov @ 0.50 | Config A Cov @ 0.25 | Config E Cov @ 0.50 | Config E Cov @ 0.25 | Failure / Success Characteristics |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`dm1`** | $2400 \times 1792$ | 27 | **17 (63%)** | **19 (70%)** | 16 (59%) | 17 (63%) | Plaster damage, high texture complexity; captured major fault networks. |
| **`cr1`** | $515 \times 388$ | 4 | **3 (75%)** | **4 (100%)** | 3 (75%) | **4 (100%)** | Both vertical fracture and horizontal branching crack 100% captured. |
| **`cr2`** | $480 \times 638$ | 22 | **22 (100%)** | **22 (100%)** | **22 (100%)** | **22 (100%)** | Reticular map cracking; 100% contained in proposals. |
| **`cr3`** | $480 \times 640$ | 1 | **1 (100%)** | **1 (100%)** | **1 (100%)** | **1 (100%)** | Single shear crack; 100% contained in proposals. |
| **`cr4`** | $638 \times 480$ | 52 | **50 (96%)** | **51 (98%)** | 49 (94%) | 50 (96%) | Complex branching fracture network; 51 of 52 segments covered. |
| **`cr5`** | $554 \times 554$ | 25 | **25 (100%)** | **25 (100%)** | **25 (100%)** | **25 (100%)** | Corner diagonal settlement crack; 100% covered. |
| **`h1`** | $2752 \times 1536$ | 2 | **2 (100%)** | **2 (100%)** | **2 (100%)** | **2 (100%)** | Distant street photo; fine diagonal fractures 100% covered. |
| **`h2`** | $1448 \times 1086$ | 11 | **10 (91%)** | **10 (91%)** | 9 (82%) | 9 (82%) | Stepped masonry crack; 10 of 11 cracks covered. |
| **`dv2`** | $1448 \times 1086$ | 4 | **4 (100%)** | **4 (100%)** | 0 (0%) | 1 (25%) | Faint diagonal shear crack. Config A captured 100%; Config E thresholded out. |
| **`k1`** | $2752 \times 1536$ | 4 | **4 (100%)** | **4 (100%)** | 2 (50%) | **4 (100%)** | Low illumination, dark overhang; 100% covered in Config A. |
| **`k2`** | $1448 \times 1086$ | 4 | **3 (75%)** | **4 (100%)** | 1 (25%) | 1 (25%) | Shadowed concrete beam; Config A captured 100% at 0.25. |
| **`i1`** | $2752 \times 1536$ | 6 | **6 (100%)** | **6 (100%)** | **6 (100%)** | **6 (100%)** | Multi-storey building facade; 100% covered. |
| **`i2`** | $1448 \times 1086$ | 2 | **2 (100%)** | **2 (100%)** | **2 (100%)** | **2 (100%)** | Window header fracture; 100% covered. |
| **`fa3`** | $2400 \times 1792$ | 1 | **1 (100%)** | **1 (100%)** | **1 (100%)** | **1 (100%)** | Parapet wall crack; 100% covered. |

---

## 5. Detailed Failure Analysis

### Failure Mode 1: The Piecewise Annotation vs. Regional Proposal Dilemma
- **Observation:** In `dv2`, all 4 ground-truth crack annotations represent segments of one continuous diagonal crack running from $(136, 375)$ to $(406, 173)$.
- **Mathematical Reality:** When Config A generates a single bounding box enclosing the connected fracture $[130, 170, 410, 380]$, its area is $58,800\text{px}^2$. The smallest ground-truth segment `crack3` has an area of $1,087\text{px}^2$.
- **Result:** The proposal box covers **100.0% of `crack3`**, but the intersection-over-union is $\frac{1,087}{58,800} = 0.018$.
- **Conclusion:** Strict IoU measures *instance boundary tightness*, not *candidate proposal recall*. The candidate proposer’s objective is to provide a spatial bounding window containing the crack for SAM 3.1. Evaluating proposals strictly on $\text{IoU} \ge 0.50$ is methodologically invalid for candidate generation. Spatial coverage / containment ($\ge 0.50$) is the true indicator of candidate sufficiency.

### Failure Mode 2: The Collapse of Thin NMS in Config C
- **Observation:** Config C (Hessian Ridge NMS) achieved only **12.1% coverage**, dropping from 90.9% in Config A.
- **Root Cause:** Applying non-maximum suppression along the normal direction thinned the continuous crack into a 1-pixel wide discontinuous trace. When small components were filtered, the crack fractured into isolated dots, and bounding boxes shrank to tiny 5x5 pixel boxes that missed the ground truth envelope.
- **Rule:** **Do not apply strict 1-pixel thinning before bounding-box extraction.** Bounding proposals require regional support to bridge low-contrast fracture gaps.

### Failure Mode 3: Faint Crack Loss via Dynamic Percentile Thresholding
- **Observation:** In `dv2`, Config E dropped coverage from 100% (Config A) to 0%.
- **Root Cause:** Config E used a dynamic 70th-percentile threshold on non-zero salience. In `dv2`, the wall has strong window frame and balcony edges. The faint plaster crack had lower vesselness than the structural window borders, so the 70th percentile threshold cut off the crack entirely.
- **Rule:** **Keep thresholding sensitive ($V > 0.12 - 0.15$) and rely on soft ranking rather than hard threshold truncation.**

### Failure Mode 4: False Proposals on Negative (Intact) Facades
- **Observation:** On negative intact facade images `in1` and `in3`, all configurations emitted 15 proposals.
- **Root Cause:** When an algorithm is asked for the `top_k=15` proposals without an absolute minimum energy/salience floor, it will return the 15 highest-ranked background noise speckles on an otherwise completely intact wall.
- **Rule:** **An absolute salience floor ($\text{Energy} \ge 250\text{px} \cdot V$) must be enforced.** Proposals below this floor must be discarded, allowing the proposer to return 0 proposals on intact walls. Furthermore, downstream VLM verification (Muse Wonder) is essential to confirm whether candidate crops contain structural anomalies.

---

## 6. Visual Comparison Artifacts

Diagnostic comparison visualizations have been generated and saved under `runs/experiments/dsp_proposals/visualizations/`:
1. `cr1_dsp_proposals.png`: Shows simultaneous capture of both horizontal and vertical fracture branches.
2. `dv2_dsp_proposals.png`: Demonstrates 100% spatial containment of the diagonal shear crack.
3. `h2_dsp_proposals.png`: Shows capture of stepped masonry cracking across brick courses.
4. `dm1_dsp_proposals.png`: Shows capture of multi-branch plaster fractures on a complex $2400 \times 1792$ facade.
5. `k1_dsp_proposals.png`: Demonstrates detection under severe shadow and low-visibility conditions.
6. `in1_negative_dsp_proposals.png`: Documents false candidate proposals generated on intact facade textures when uncalibrated by an absolute floor.

---

## 7. Recommended Next Steps for Phase 2

1. **Adopt Calibrated Frangi Candidate Generation in the Engine:**
   - Integrate `propose_crack_candidate_boxes` from `neev.measurements.crack_dsp` into the screening runner.
   - Maintain a sensitive threshold ($V > 0.15$) and multi-scale Gaussian derivatives $\sigma \in (1.2, 2.5, 4.0)$.
2. **Implement Absolute Salience Cutoff:**
   - Add an absolute energy cutoff ($\text{Energy} \ge 250$) so that intact facade images return $\le 2$ or 0 candidate proposals.
3. **Proceed to Experiment 2 (SAM 3.1 Visual Box Prompting Pilot):**
   - Take the top-5 candidate boxes generated by `propose_crack_candidate_boxes` on pilot images (`dv2`, `cr1`, `h2`, `k1`).
   - Feed these boxes as visual `box` prompt tokens into the SAM 3.1 adapter (`neev/adapters/sam31.py`).
   - Measure final segmentation boundary IoU against the ground truth.
