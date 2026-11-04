# CUBIT-Seg Benchmark Reproduction Guide

> **Dataset:** CUHK CUBIT-Seg Crack Segmentation Benchmark  
> **Source Repository:** [https://github.com/CUHK-USR-Group/Defect-Dataset](https://github.com/CUHK-USR-Group/Defect-Dataset)  
> **Evaluation Standard:** Strict Cardinality-First Bipartite Matching with $\text{IoU} \ge 0.50$ (`cov50`)  
> **API Cost Target:** **$0.00** (Zero external paid Meta API calls)  

---

## 1. System & Hardware Environment

* **Operating System:** Windows 11 Home / Pro (64-bit)
* **Python Runtime:** Python 3.13 / 3.11
* **Deep Learning Framework:** PyTorch 2.7.1+cu118
* **Acceleration Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB dedicated VRAM)
* **Host CPU:** 13th Gen Intel Core i7 (24 logical cores)

---

## 2. Dataset Ingestion & Zero-Leakage Normalization

The ingestion script discovers Google Drive file IDs, downloads raw uncompressed crops, normalizes resolutions to $512 \times 512$, computes SHA-256 content hashes, classifies crack widths via morphological skeleton thinning, and partitions by source building ID:

```bash
# Ingest and normalize 1,320 image-mask pairs
python scripts/ingest_cubit_dataset.py
```

### Partitioning Parameters:
* **Partitioning Seed:** `42` (Fixed, fully deterministic)
* **Split Hierarchy:** By source building identifier (`source_id`), strictly preventing same-building, crop-level, or viewpoint leakage.
* **Partitions Generated:**
  * **Development (`dev`):** 7 source buildings, 264 images
  * **Validation (`val`):** 10 source buildings, 373 images
  * **Frozen Test (`test`):** 15 source buildings, 683 images (untouched final holdout)
  * **Total:** 32 source buildings, 1,320 images (2,640 total normalized files)

### Verification:
```bash
python -c "import json; m = json.load(open('CUBIT_DATASET_MANIFEST.json')); print('Images:', len(m['images']), 'Files Verified:', len(m['images'])*2)"
```

---

## 3. Evaluation Integrity & Unit Testing

Validate the cardinality-first bipartite matching protocol, duplicate prediction suppression, and negative image evaluation before benchmarking:

```bash
python -m pytest -q -p no:asyncio tests/test_cubit_evaluation.py
```
Expected output:
```
6 passed in ~1.3s
```

Run full regression test suite across all Neev components:
```bash
python -m pytest -q -p no:asyncio tests/
```
Expected output:
```
362 passed in ~90s
```

---

## 4. Master Benchmark Execution

Execute the full 14-architecture benchmark across all 1,320 images (18,480 total image inspections):

```bash
python -u scripts/run_cubit_master_benchmark.py
```

### Fast Smoke Mode (Verification Run):
To run a fast verification pass on 9 sample images across all 14 architectures:
```bash
python -u scripts/run_cubit_master_benchmark.py --smoke
```

---

## 5. Output Artifacts Inventory

Running the master benchmark produces the following deliverables:

1. **[`CUBIT_DATASET_AUDIT.md`](file:///d:/projects/metaglobal/CUBIT_DATASET_AUDIT.md):** Complete dataset audit, provenance, license, geometry, and partition metadata.
2. **[`CUBIT_DATASET_MANIFEST.json`](file:///d:/projects/metaglobal/CUBIT_DATASET_MANIFEST.json):** Machine-readable manifest tracking all 1,320 images, SHA-256 hashes, source building IDs, and split assignments.
3. **[`CUBIT_BENCHMARK_RESULTS.json`](file:///d:/projects/metaglobal/CUBIT_BENCHMARK_RESULTS.json):** Complete machine-readable performance metrics (TP, FP, FN, Precision, Recall, F1, Mask IoU, Latency, Category Recalls) for all 14 architectures across DEV, VAL, and Frozen TEST.
4. **[`CUBIT_BENCHMARK_PER_IMAGE.csv`](file:///d:/projects/metaglobal/CUBIT_BENCHMARK_PER_IMAGE.csv):** Per-image, per-architecture tabular metrics across all 18,480 evaluations.
5. **[`CUBIT_BENCHMARK_ARCHITECTURE_COMPARISON.md`](file:///d:/projects/metaglobal/CUBIT_BENCHMARK_ARCHITECTURE_COMPARISON.md):** Main comparative report containing master tables, cross-split analysis, historical DEV-10 / VAL-19 comparisons, and multi-criteria rankings.
6. **[`CUBIT_BENCHMARK_FAILURE_ANALYSIS.md`](file:///d:/projects/metaglobal/CUBIT_BENCHMARK_FAILURE_ANALYSIS.md):** Detailed diagnostic breakdown of false alarms, missed cracks, and architectural failure patterns.
7. **[`CUBIT_BENCHMARK_VISUALS/`](file:///d:/projects/metaglobal/CUBIT_BENCHMARK_VISUALS):** 70 side-by-side 3-panel visual overlay images comparing model predictions (TP in cyan, FP in red, FN in orange) against ground truth masks.
