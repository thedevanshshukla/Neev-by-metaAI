# CUBIT-Seg Dataset Audit & Integration Report

## 1. Dataset Overview & Provenance
* **Official Repository:** [CUHK-USR-Group/Defect-Dataset](https://github.com/CUHK-USR-Group/Defect-Dataset)
* **Dataset Name:** CUBIT-Seg (Crack Segmentation Subset)
* **Originating Institution:** The Chinese University of Hong Kong (CUHK), Unmanned Systems Research (USR) Group
* **Published Papers & Citations:**
  1. *Yang et al., "Det-Recon-Reg: An Intelligent Framework Toward Automated UAV-Based Large-Scale Infrastructure Inspection", IEEE Transactions on Instrumentation and Measurement (TIM), vol. 74, pp. 1-16, 2025. DOI: [10.1109/TIM.2025.3571118](https://doi.org/10.1109/TIM.2025.3571118)*
  2. *Yang et al., "Datasets and processing methods for boosting visual inspection of civil infrastructure: A comprehensive review and algorithm comparison", Construction and Building Materials, vol. 356, p. 129226, 2022. DOI: [10.1016/j.conbuildmat.2022.129226](https://doi.org/10.1016/j.conbuildmat.2022.129226)*
* **License & Conditions of Use:** Academic & Research Evaluation Use (Non-commercial, with formal scholarly citation as stipulated in official repository).

---

## 2. Dataset Characteristics & Annotation Format
* **Total Available Images in CUBIT-Seg Crack Repository:** 5,462 RGB images and 5,462 matching ground truth masks across 147 unique source UAV building capture sequences.
* **Resolution:** $512 \times 512$ pixels (uncompressed 1:1 sensor crops).
* **Format:** 24-bit JPEG for RGB images; 8-bit PNG for pixel-level ground truth masks.
* **Mask Semantics:** Pixel-level annotation where pixel value $> 0$ indicates genuine building facade crack structure; $0$ indicates intact facade concrete/masonry background.
* **Instance Derivation Methodology:** Connected components extraction (8-connectivity, minimum noise threshold $\ge 20\text{px}$) yielding rotated bounding rectangles, axis-aligned bounding boxes, and polygon contours while preserving the original binary ground truth mask.

---

## 3. Benchmark Partitioning & Zero-Leakage Guarantee
To eliminate any potential for same-building, crop-level, or viewpoint leakage, splits were created strictly at the **source image / building identifier level** (`source_id`, e.g. `IMG_6611`, `IMG_7206`, `IMG_8009`):

| Partition Split | Source Buildings | Number of Crops | Percentage | Usage / Evaluation Policy |
|:---|---:|---:|---:|:---|
| **Development (`dev`)** | 7 | 264 | 20.0% | Architectural calibration & sensitivity tuning |
| **Validation (`val`)** | 10 | 373 | 28.3% | Cross-validation & hyperparameter validation |
| **Frozen Test (`test`)** | 15 | 683 | 51.7% | **Untouched, frozen final benchmark** (Zero threshold tuning) |
| **Total Benchmark** | **32** | **1320** | **100.0%** | **Rigorous 14-architecture master evaluation** |

* **Random Seed:** 42 (Fully deterministic and reproducible).
* **Zero Leakage:** No crops from the same original UAV photo exist across more than one split.

---

## 4. Crack Geometry & Width Distribution
Based on morphological skeleton thinning ($w = \text{Area} / \text{Skeleton Length}$):
* **Hairline Cracks ($w \le 3.0\text{px}$):** 15 images (1.1%)
* **Medium Cracks ($3.0 < w \le 8.0\text{px}$):** 882 images (66.8%)
* **Wide Cracks ($w > 8.0\text{px}$):** 423 images (32.0%)

---

## 5. Artifact Manifest & Paths
* **Manifest File:** [`CUBIT_DATASET_MANIFEST.json`](file:///D:/projects/metaglobal/CUBIT_DATASET_MANIFEST.json)
* **Normalized Images:** `data/cubit_seg_normalized/images/`
* **Normalized Ground Truth Masks:** `data/cubit_seg_normalized/masks/`
