# Experiment 1.1: Proposal Quality & Energy Calibration

**Status**: Completed  
**Pipeline Phase**: Phase 2 — Core Detection Engine R&D (Offline Evaluation & Calibration)  
**Execution Environment**: Pure Python / NumPy / SciPy (Offline, zero live API calls, zero credit spend: $0.00)  
**Machine-Readable Results**: [`runs/experiments/dsp_calibration/calibration_results.json`](file:///D:/projects/metaglobal/runs/experiments/dsp_calibration/calibration_results.json)  
**Visual Diagnostics**: [`runs/experiments/dsp_calibration/visualizations/`](file:///D:/projects/metaglobal/runs/experiments/dsp_calibration/visualizations/)  

---

## Executive Summary & Verdict

### Final Decision: **MODIFY**

The classical Digital Signal Processing (DSP) candidate proposal generator based on multi-scale Frangi vesselness, skeleton topology, and salience ranking achieves **89.4% physical crack recall** (59 of 66 physical crack networks covered $\ge 50\%$) and **91.5% annotation-box coverage** (151 of 165 Label Studio boxes covered $\ge 50\%$) across the full positive dataset under a budget of $K = 15$ proposals per image.

However, **unconditional adoption (ADOPT) is rejected** due to three critical architectural findings:
1. **The $E = 250$ scalar energy cutoff hypothesis is falsified**: A global scalar energy threshold does not eliminate false proposals on large intact building facades. On high-resolution intact facade `in1` ($2752 \times 1536$), long architectural window mullions and facade panel seams accumulate energies exceeding $169,746$, retaining all 15 false proposals even at $E = 400$. Conversely, $E \ge 200$ prematurely truncates genuine faint cracks in smaller images (`cr4`, `cr1`).
2. **Extreme Area Disparity (330x Proposal-to-GT Ratio)**: Proposal bounding boxes are on average $330\times$ larger than the ground-truth crack pixels they enclose. Average proposal precision is $4.36\%$, meaning $>95\%$ of each proposal box consists of background wall.
3. **Negative Facade False Proposals**: On negative (intact) building facades, DSP generates candidate boxes along architectural grid lines. A downstream semantic verifier (such as Muse Wonder or SAM prompt-mask scoring) is strictly required to filter non-crack candidates.

The candidate generator is **READY** for a controlled SAM 3.1 visual-box prompting experiment, provided the **MODIFY** directives specified in Section 8 are implemented.

---

## 1. Audit of the Existing Proposal Implementation

Before calibration, an audit of `neev/measurements/crack_dsp.py`, `runs/experiments/dsp_proposals/run_dsp_experiment.py`, and `DSP_PROPOSAL_EXPERIMENT.md` was conducted to verify the exact mathematical definitions of all metrics.

| Component / Metric | Existing Implementation / Historical Description | Audit Finding & Mathematical Reality |
| :--- | :--- | :--- |
| **Candidate Energy** | Described as "component energy" | Calculated as $\sum_{(x,y) \in \text{comp}} V(x,y)$, where $V(x,y) \in [0, 1]$ is the Frangi vesselness response. Energy is mathematically an integral of vesselness over component pixels. Consequently, energy scales with both line contrast and line length. Long architectural edges on high-resolution images naturally yield massive energies ($>10^5$). |
| **Ground-Truth Coverage** | Termed "GT recall" in Experiment 1 | Measures the fraction of an individual Label Studio annotation box covered by the single best candidate proposal box. It is **not** physical crack recall; a single continuous crack annotated as 4 piecewise segments counted as 4 separate recall events. |
| **Proposal Count** | Fixed at top $K = 15$ | Proposals are ranked by $S = \text{Energy} \times \sqrt{\text{Length}} \times \text{rect\_mult}$. Truncation at $K$ is uncalibrated against image scale and defect complexity. |
| **Proposal Deduplication** | None in baseline Config A | Connected components are isolated via 8-connected labeling on thresholded Frangi vesselness ($V > 0.15$). No spatial NMS or bounding-box overlap merging was performed. |
| **Negative Image Behavior** | Hypothesized that $E \ge 250$ suppresses negative noise | Evaluated on only 2 negative images. While $E \ge 250$ effectively suppresses fine pebble-dash texture on `in3`, it fails completely on `in1`, where architectural seams yield energies orders of magnitude above 250. |

---

## 2. Annotation Grouping Analysis

### 2.1 Why Grouping is Required
In Label Studio, human annotators decompose long curvilinear, tortuous, or branching cracks into multiple short straight bounding boxes. Across the 14 positive images, there are **165 individual annotation rectangles**. 

Describing the percentage of these 165 rectangles covered as "crack recall" is misleading:
- An inspection engineer cares whether the *physical crack* was identified, not whether 10 sub-segments of the same crack were individually covered.
- When an algorithm generates a single tight bounding box covering an entire continuous crack, strict IoU against each 20-pixel constituent sub-box is artificially penalized (often yielding IoU $< 0.05$), despite 100% true physical coverage.

### 2.2 Grouping Methodology & Geometry
We constructed an undirected graph $G = (V, E)$ for each image:
- Each vertex $v_i \in V$ is an annotated crack rectangle polygon.
- An edge $e_{ij}$ connects $v_i$ and $v_j$ if and only if:
  1. **Spatial Proximity**: $\min_{p \in v_i, q \in v_j} \|p - q\|_2 \le \max(15\text{px}, 0.012 \times \text{diagonal})$.
  2. **Orientation Alignment**: Angular difference $|\theta_i - \theta_j| \pmod{180^\circ} \le 35.0^\circ$.
- Physical crack systems are the connected components of $G$. The polygon union is rasterized to compute exact ground-truth pixel area without double-counting overlapping endpoints.

### 2.3 Grouping Results by Image

| Image ID | Image Dimensions | Ungrouped Boxes | Grouped Physical Cracks | Largest Group Size | Structural Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `cr1` | $515 \times 388$ | 13 | 7 | 4 | Step-shear cracks on concrete |
| `cr2` | $515 \times 449$ | 10 | 5 | 3 | Vertical settling fractures |
| `cr3` | $688 \times 490$ | 3 | 1 | 3 | Single vertical continuous fracture |
| `cr4` | $515 \times 388$ | 52 | 18 | 9 | Complex branching network |
| `cr5` | $515 \times 388$ | 25 | 6 | 7 | Interconnected facade crack system |
| `dv1` | $1536 \times 2048$ | 3 | 2 | 2 | Plaster corner cracks |
| `dv2` | $1536 \times 2048$ | 4 | 1 | 4 | Long diagonal plaster shear crack |
| `h1` | $2048 \times 1536$ | 3 | 2 | 2 | Vertical mortar joint fracture |
| `h2` | $1536 \times 2048$ | 20 | 9 | 4 | Stepped masonry mortar crack |
| `h3` | $2048 \times 1536$ | 5 | 3 | 3 | Brick wall settlement cracks |
| `k1` | $3024 \times 4032$ | 7 | 4 | 3 | High-res shadowed overhang crack |
| `k2` | $3024 \times 4032$ | 8 | 4 | 3 | High-res vertical pillar crack |
| `s1` | $1536 \times 2048$ | 4 | 2 | 3 | Spalling-adjacent shear crack |
| `s2` | $1536 \times 2048$ | 8 | 2 | 6 | Multi-segment horizontal fracture |
| **Total** | — | **165** | **66** | — | **2.5:1 Reduction Ratio** |

### 2.4 Ambiguous Cases Documented
- **T-Junctions and Orthogonal Branches (`cr4`)**: In `cr4`, secondary horizontal cracks branch off the primary vertical fracture at near $90^\circ$ angles. Because $|\theta_1 - \theta_2| \approx 90^\circ > 35^\circ$, our geometric rule kept these branches as distinct physical sub-cracks. This is advantageous for visual prompting: prompting SAM with separate boxes per branch prevents SAM from hallucinating a massive diagonal envelope.
- **Masonry Stepped Fractures (`h2`)**: The crack steps through alternating horizontal bed joints and vertical head joints. When adjacent sub-rectangles alternate between $0^\circ$ and $90^\circ$, they are preserved as distinct connected segments rather than forced into a single amorphous cluster.

---

## 3. Improved Evaluation Metrics

We evaluated candidate proposals across the 14 positive images (165 ungrouped boxes, 66 physical crack groups) using exact pixel-level rasterization:

1. **Ground-Truth Coverage**: For each GT polygon, $\text{Coverage} = \max_{p \in \text{Props}} \frac{|\text{GT} \cap \text{Prop}_p|}{|\text{GT}|}$. Evaluated at thresholds 25%, 50%, 75%, and 90%.
2. **Proposal Precision**: Fraction of proposal bounding-box area overlapping the union mask of annotated crack pixels: $\text{Prec} = \frac{|\text{Prop} \cap \bigcup \text{GT}|}{|\text{Prop}|}$.
3. **Proposal-to-GT Area Ratio**: How much larger the proposal box is compared to the GT region it covers: $\frac{\text{Area}(\text{Prop})}{\text{Area}(\text{GT})}$.
4. **Strict Rotated Polygon IoU**: Rotated polygon IoU at 0.10, 0.25, and 0.50 for historical benchmarking.
5. **Negative Facade Performance**: Proposal counts, maximum energy, mean energy, and mean area on verified intact negative facades `in1` and `in3`.

---

## 4. Energy Calibration Experiments

We evaluated absolute energy filtering thresholds $E \in [0, 100, 150, 200, 250, 300, 400]$ at fixed budget $K = 15$.

### 4.1 Calibration Results Table

| Energy Cutoff ($E \ge$) | Total Props | Props / Image | Mean Precision | Ungrouped Cov @ 50% | Grouped Cov @ 50% | Grouped Cov @ 90% | Faint Crack Losses | `in1` Props (Negative) | `in3` Props (Negative) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 (No Filter)** | 210 | 15.0 | 4.36% | **151/165 (91.5%)** | **59/66 (89.4%)** | **52/66 (78.8%)** | **0** | 15 | 15 |
| **100** | 199 | 14.2 | 5.92% | 151/165 (91.5%) | 59/66 (89.4%) | 52/66 (78.8%) | 0 | 15 | 13 |
| **150** | 190 | 13.6 | 6.27% | 150/165 (90.9%) | 59/66 (89.4%) | 51/66 (77.3%) | 0 | 15 | 2 |
| **200** | 179 | 12.8 | 6.27% | 149/165 (90.3%) | 58/66 (87.9%) | 51/66 (77.3%) | 1 (`cr4`) | 15 | 1 |
| **250** | 165 | 11.8 | 5.66% | 149/165 (90.3%) | 58/66 (87.9%) | 51/66 (77.3%) | 1 (`cr4`) | 15 | 1 |
| **300** | 157 | 11.2 | 6.91% | 149/165 (90.3%) | 58/66 (87.9%) | 51/66 (77.3%) | 1 (`cr4`) | 15 | 1 |
| **400** | 150 | 10.7 | 6.06% | 146/165 (88.5%) | 57/66 (86.4%) | 51/66 (77.3%) | 2 (`cr4`) | 15 | 1 |

### 4.2 Coverage Threshold Breakdown (at Baseline $E = 0, K = 15$)

| Evaluation Mode | Total Ground Truths | Cov $\ge 25\%$ | Cov $\ge 50\%$ | Cov $\ge 75\%$ | Cov $\ge 90\%$ | Mean Coverage | Mean Area Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ungrouped Rectangles** | 165 | 154 (93.3%) | 151 (91.5%) | 147 (89.1%) | 145 (87.9%) | 88.5% | 330.2x |
| **Grouped Physical Cracks** | 66 | 61 (92.4%) | 59 (89.4%) | 56 (84.8%) | 52 (78.8%) | 85.9% | 330.2x |

### 4.3 Historical Strict IoU Benchmarks

| Metric | IoU $\ge 0.10$ | IoU $\ge 0.25$ | IoU $\ge 0.50$ |
| :--- | :---: | :---: | :---: |
| **Historical Baseline IoU (165 Rectangles)** | 71 / 165 (43.0%) | 28 / 165 (17.0%) | 6 / 165 (3.6%) |

*Insight*: The disparity between high coverage ($91.5\%$) and low IoU ($3.6\%$) is mathematically inevitable when bounding boxes enclose thin diagonal lines. For a 2-pixel wide diagonal crack spanning a $200 \times 200$ box, the crack area is $\approx 560$ px while the bounding box area is $40,000$ px ($\text{IoU} \approx 0.014$). Strict IoU is therefore an inappropriate evaluation metric for visual prompting proposals; GT coverage is the appropriate operational metric.

### 4.4 Falsification of the $E = 250$ Cutoff
Experiment 1 conjectured that filtering candidates at $E \ge 250$ was an optimal operating point. Our empirical sweep disproves this:
1. **Faint Crack Loss**: At $E = 250$, faint crack branches in `cr4` drop below threshold, reducing grouped physical recall from 59 to 58. At $E = 400$, two physical crack systems in `cr4` are dropped. In low-resolution images, real crack segments have integrated vesselness as low as $80$.
2. **Negative Building Failure (`in1`)**: On image `in1` ($2752 \times 1536$), building spandrels, window mullions, and floor slab joints have continuous lengths of hundreds of pixels. The top candidate on `in1` has an energy of **169,746.4**—over $670\times$ higher than the 250 threshold! Even at $E = 400$, `in1` retains all 15 false proposals.
3. **Conclusion**: An absolute scalar energy threshold cannot separate facade cracks from architectural edges across different image scales.

---

## 5. Proposal Budget Experiments ($K = 5, 10, 15$)

We evaluated candidate budgets $K \in [5, 10, 15]$ at $E = 0$.

| Budget ($K$) | Total Props | Grouped Cov @ 50% | Ungrouped Cov @ 50% | Mean Precision | Mean Proposal Area | Runtime / Image | Negative Props (`in1`, `in3`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 5$** | 70 | 56 / 66 (84.8%) | 146 / 165 (88.5%) | **5.80%** | 373,915 px | 3.5s | 5, 5 |
| **$K = 10$** | 140 | 58 / 66 (87.9%) | 149 / 165 (90.3%) | 4.75% | 197,368 px | 3.5s | 10, 10 |
| **$K = 15$** | 210 | **59 / 66 (89.4%)** | **151 / 165 (91.5%)** | 4.36% | 135,712 px | 3.5s | 15, 15 |

### 5.1 Budget Trade-Offs
- **Recall Gain**: Expanding $K$ from 5 to 15 increases grouped physical crack recall from $84.8\%$ to $89.4\%$ (+3 physical crack networks recovered in complex branching scenes like `cr4` and `cr5`).
- **Proposal Box Scale**: Top-ranked proposals ($K=5$) correspond to the largest, longest continuous components ($\text{mean area} \approx 373\text{k px}$). Lower-ranked proposals ($K=10, 15$) capture smaller localized branch fractures, reducing the average proposal area to $135\text{k px}$.
- **Latency Invariance**: Runtime is completely dominated by the 2D multi-scale Gaussian derivative convolutions during Frangi vesselness calculation ($\approx 3.5$s per image). Top-K ranking is sub-millisecond; changing $K$ from 5 to 15 has zero measurable impact on runtime.

---

## 6. Proposal Merging Analysis (Unmerged vs Merged)

We investigated whether merging overlapping candidate proposals ($\text{IoU} \ge 0.25$) into an enclosing bounding box improves prompting precision or creates oversized boxes.

| Mode | Total Proposals | Mean Proposal Area | Proposal Precision | Ungrouped Cov @ 50% | Grouped Cov @ 50% | Mean Area Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unmerged** | 210 | 135,712 px | 4.36% | 151 / 165 (91.5%) | 59 / 66 (89.4%) | 330.2x |
| **Merged ($\text{IoU} \ge 0.25$)** | 210 | 132,551 px | 4.39% | 151 / 165 (91.5%) | 59 / 66 (89.4%) | 332.3x |

### Finding
Because Frangi vesselness components are extracted via 8-connected component labeling (`ndimage.label`), raw components rarely overlap significantly unless their bounding boxes intersect obliquely. Merging at $\text{IoU} \ge 0.25$ produced negligible difference in proposal count or coverage. 

Attempting aggressive merging of non-overlapping nearby boxes was found to create **massive, oversized boxes** that encompass huge swaths of non-defective facade wall, degrading SAM prompt specificity. Proposal merging is therefore **not recommended**.

---

## 7. Visual Diagnostics for Representative Cases

High-resolution visual diagnostics were generated and saved in `runs/experiments/dsp_calibration/visualizations/`. Each visualization displays:
- **Left**: Original image overlay showing Ground-Truth Annotations (Green polygons), Physical Crack Group Envelopes (Cyan boxes), and Candidate Proposals (Magenta boxes with Energy scores).
- **Right**: Normalized Frangi Hessian Vesselness Heatmap with Candidate Proposal Bounding Boxes (Yellow).

```
Visual Diagnostics Inventory:
├── cr3_calibration_diagnostic.png   (Simple Crack - isolated vertical fracture)
├── cr4_calibration_diagnostic.png   (Branching Crack - intricate multi-branch network)
├── dv2_calibration_diagnostic.png   (Faint Crack - subtle diagonal plaster shear fracture)
├── k1_calibration_diagnostic.png    (Shadowed Crack - high-contrast illumination gradient)
├── h2_calibration_diagnostic.png    (Complex Masonry - stepped mortar joint fracture)
├── in1_calibration_diagnostic.png   (Intact Negative 1 - multi-storey building with grid seams)
└── in3_calibration_diagnostic.png   (Intact Negative 2 - textured stucco facade)
```

### Case-by-Case Diagnostic Analysis

1. **Simple Crack (`cr3`)**:
   - *Observation*: A continuous vertical crack across concrete. The 3 Label Studio boxes form 1 physical crack group.
   - *DSP Response*: Top candidate proposal $P_0$ ($E = 1297$) aligns with the crack with 100% coverage. Vesselness cleanly suppresses the surrounding concrete texture.
2. **Branching Crack (`cr4`)**:
   - *Observation*: 52 Label Studio rectangles forming 18 distinct branches.
   - *DSP Response*: Proposals capture all major vertical and diagonal branches ($88.9\%$ grouped coverage). Micro-hairline branches with $E < 150$ require $K \ge 15$ to be captured.
3. **Faint Crack (`dv2`)**:
   - *Observation*: Subtle diagonal shear crack traversing a plaster facade. 4 collinear annotations form 1 continuous physical crack.
   - *DSP Response*: Vesselness values are low ($V \in [0.18, 0.32]$), but because the crack is continuous over 600+ pixels, integrated energy reaches $E = 624.5$. Top proposal $P_0$ encloses the crack completely ($100\%$ coverage).
4. **Shadowed Crack (`k1`)**:
   - *Observation*: High-resolution ($3024 \times 4032$) image with harsh shadow under a balcony overhang.
   - *DSP Response*: The multi-scale Hessian eigenvalues cleanly reject the step edge of the illumination shadow and isolate the crack ridge. Candidate boxes cover all 4 physical crack systems ($100\%$ grouped coverage).
5. **Complex Masonry Facade (`h2`)**:
   - *Observation*: 20 Label Studio boxes following stepped mortar joints in a brick facade.
   - *DSP Response*: Mortar joints exhibit significant linear ridge contrast. The top proposals cover 8 of the 9 physical crack systems ($88.9\%$ coverage), but 3 brick mortar joints without cracks are also proposed.
6. **Intact Negative Facade 1 (`in1`)**:
   - *Observation*: Multi-storey building with curtain wall window frames, horizontal spandrel panels, and architectural grid lines. No cracks exist.
   - *DSP Response*: Frangi filter generates strong responses along architectural lines ($E_{\max} = 169,746$). Proposals cluster on window mullions and floor transitions. A downstream semantic verifier is mandatory.
7. **Intact Negative Facade 2 (`in3`)**:
   - *Observation*: Uniform stucco surface with fine aggregate pebble texture. No cracks exist.
   - *DSP Response*: Pebble texture generates isolated points rather than continuous ridges. At $E \ge 150$, proposals drop from 15 down to 2, and at $E \ge 250$ only 1 proposal (a plaster construction seam) remains.

---

## 8. Limitations & Required Modifications for SAM 3.1 Prompting

### 8.1 Measured Limitations
1. **Low Proposal Precision ($4.36\%$)**: Bounding boxes inherently contain large amounts of background wall. Passing these boxes directly to SAM will require SAM to identify the medial linear defect within a broad bounding context.
2. **Scale Invariance Failure of Scalar Energy**: A fixed scalar energy cutoff fails across varied camera distances and sensor resolutions.
3. **Architectural Ridge Confusion on Intact Negatives**: The Hessian ridge detector cannot semantically distinguish an architectural metal window frame from a straight concrete crack.

### 8.2 Recommended Modifications Before SAM 3.1 Integration

To transition the candidate generator into an effective visual prompting module for SAM 3.1, the following four modifications are required:

1. **Relative Salience Ranking Over Hard Truncation**:
   - Do **not** use a hard cutoff like $E \ge 250$. Keep $E \ge 0$ (or a minimal noise gate $E \ge 25$) and rank candidates using the calibrated composite salience score:
     $$S = \text{Energy} \times \sqrt{\text{Length}} \times \text{rect\_mult}$$
   - On positive images, this ensures faint cracks are retained.
2. **Downstream VLM Verification (The Muse Wonder Filter)**:
   - Because DSP proposals on intact negative facades (`in1`) cannot be eliminated via classical filtering without dropping faint real cracks, false alarm suppression must be delegated to downstream semantic verification.
   - For every candidate proposal where SAM produces a mask, extract a localized patch and query the VLM (or prompt-based feature head) to confirm: "Is this linear feature a structural crack or an architectural joint/window frame?"
3. **Bounding-Box Aspect Ratio Clamping**:
   - For sprawling diagonal cracks that span $500 \times 500$ pixel areas, decompose the connected component along its skeleton medial axis into 2–3 localized sub-boxes. This prevents sending an enormous $500 \times 500$ box containing $99\%$ background to SAM.
4. **Adaptive Multi-Scale Scale Parameterization**:
   - Current scales $(1.2, 2.5, 4.0)$ are well-tuned for standard focal lengths. Ensure GSD (Ground Sampling Distance) is accounted for when available.

---

## 9. Verification & Audit Sign-Off

- **Baseline Codebase Preserved**: The production inspection pipeline, SAM single-view/multi-view adapters, and existing experiment outputs (`runs/crack_exp_eval.json`, `dsp_proposal_results.json`) remain intact and unaltered.
- **Regression Tests Executed**: 
  - `tests/test_crack_dsp_proposals.py`: 4 passed (including new `min_energy` filter regression test).
  - `tests/test_app.py` & `tests/test_multiview.py`: 58 passed.
  - Total test suite status: 100% green.
- **Live API & Financial Audit**:
  - Live Meta API calls made: **0**
  - Paid API credits spent: **$0.00**
  - Model weights downloaded or fine-tuned: **None**
