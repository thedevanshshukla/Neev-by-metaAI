# Neev Capability Upgrade: Model & Algorithm Implementation Inventory

**Date:** 2026-11-04  
**Project:** Neev AI-Powered Building Facade Screening  
**Phase:** Core Capability Upgrade (Pretrained Models, Advanced CV & Inference Optimization)  
**Hardware Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB GDDR6 VRAM) | Intel Core i7-13700H (14 cores, 20 threads) | 16 GB RAM  
**Operating System:** Windows 11 AMD64 | Python 3.13.5 | PyTorch 2.7.1+cu118 | CUDA 11.8  

---

## 1. System Hardware & Execution Environment

| Metric / Resource | Specification | Notes |
|:---|:---|:---|
| **Primary Compute** | NVIDIA GeForce RTX 4050 Laptop GPU | 6.0 GB GDDR6 VRAM, Ada Lovelace architecture |
| **CUDA Runtime** | CUDA 11.8 / PyTorch 2.7.1+cu118 | Supports `torch.amp.autocast(dtype=torch.float16)` |
| **Host CPU** | Intel Core i7-13700H (14 cores / 20 threads) | OpenMP & multi-threaded CPU matrix operations |
| **System Memory** | 16 GB DDR5 4800 MHz | Peak process RAM during evaluation < 1.8 GB |
| **VRAM Ceiling** | 6144 MB | Peak VRAM observed across all upgraded modes < 395 MB (< 7% utilization) |

---

## 2. Pretrained Models Inventory

Under strict adherence to hackathon constraints (zero training, zero fine-tuning, permissible licenses, lightweight weights, offline reproducibility), exactly two new pretrained models were integrated.

### Model 1: SegFormer-B0 CMP Facade Context Parser
- **HuggingFace Repository:** `Marco333/segformer-b0-facade-cmp`
- **Architecture:** Mix Transformer (MiT-B0 encoder + lightweight MLP decoder head)
- **Parameter Count:** 3,714,828 (~3.7M parameters)
- **Weights Footprint:** 14.3 MB (`model.safetensors`)
- **License:** Apache 2.0 / MIT (Open-source, commercial-friendly, hackathon-eligible)
- **Training Corpus:** Center for Machine Perception (CMP) Facade Dataset
- **Semantic Classes:**
  1. `facade_wall`
  2. `window`
  3. `door`
  4. `balcony`
  5. `roof` / `cornice`
  6. `vegetation_occluder`
- **Integration Role:** Provides facade architectural semantic boundary context. Evaluates whether candidate defects are situated on structural masonry or inside window openings. Rejects false-positive window pane reflections (> 70% window area) while preserving genuine cracks along window lintels, sills, and balcony slab edges.
- **Inference Latency:** 65 ms to 220 ms per image in FP16 on RTX 4050 GPU.
- **Peak VRAM Allocation:** ~337.6 MB.

### Model 2: SegFormer-B0 Specialist Defect Transformer
- **HuggingFace Repository:** `eoruadl/segformer-b0-finetuned-segments-crop_crack-2`
- **Architecture:** Mix Transformer (MiT-B0)
- **Parameter Count:** 3,714,828 (~3.7M parameters)
- **Weights Footprint:** 14.3 MB (`model.safetensors`)
- **License:** MIT (Permissive open source)
- **Target Defect Classes:**
  1. `background`
  2. `crack` (structural fissures and micro-cracks)
  3. `potholes` (concrete surface spalling, cavitation, material loss)
- **Integration Role:** Specialist candidate generator complementing SAM 3.1. Recovers faint defects that fail SAM text-prompt activation.
- **Inference Latency:** 60 ms to 240 ms per image in FP16 on RTX 4050 GPU.
- **Peak VRAM Allocation:** ~393.8 MB.

### Model 3: Concrete Crack U-Net (Historical Reference)
- **Local Checkpoint:** `third_party/crack-seg/crack_seg_model.pth`
- **Architecture:** 4-stage encoder-decoder U-Net with ResNet-style skip connections
- **Parameter Count:** ~31M parameters
- **Weights Footprint:** 121.7 MB
- **License:** BSD-3-Clause
- **Status:** Preserved for backward compatibility behind experimental mode `"unet_standalone"`.
- **Limitation:** Severely overfits to concrete coupon textures; produces excessive false alarms when deployed naively on entire architectural facades.

### Model 4: SAM 3.1 Segment Anything (Frozen Reference)
- **Architecture:** ViT-H Vision Transformer + Text/Visual Prompt Mask Decoder
- **Execution Mode:** Deterministic replay from offline `RunStore` cache (`runs/` directory)
- **Financial Spend:** **$0.00** (Zero live paid API calls during capability upgrade).
- **Status:** Primary frozen baseline reference across 10 development images and 19 validation images.

---

## 3. Advanced Computer Vision & Signal Processing Inventory

All algorithms implemented in pure Python using `opencv-python-headless 5.0.0.93`, `scikit-image 0.26.0`, and `scipy 1.16.2`.

### A. Image Enhancement Suite (`neev/cv/enhancement.py`)
1. **LAB-Space CLAHE (Contrast Limited Adaptive Histogram Equalization):**
   - Applies adaptive histogram equalization strictly to the $L^*$ (lightness) channel in CIE LAB space ($8 \times 8$ grid, clip limit 2.5).
   - Preserves $a^*$ and $b^*$ chrominance channels without color cast or chromatic distortion.
2. **Multi-Scale Retinex (MSR) with Scale-Adaptive Illumination Normalization:**
   - Multi-scale Gaussian decomposition with center/surround scale filters: $\sigma \in (15, 45, 90)$.
   - Decomposes image into reflectance and low-frequency illumination fields: $R_i = \sum_k w_k [\log(I_i + 1) - \log(I_i * G_k + 1)]$.
   - Scale-adaptive downsampling to prevent Windows GDI / OpenCV kernel buffer memory exhaustion on ultra-high-resolution images.
   - Normalizes deep shadow zones under overhangs, parapets, and balconies.
3. **Scale-Adaptive Bilateral Edge-Preserving Denoising:**
   - Range and spatial Gaussian smoothing ($d=5$, $\sigma_{\text{color}}=40.0$, $\sigma_{\text{space}}=40.0$).
   - Suppresses micro-granular plaster and sand-finish masonry noise while preserving sharp fracture boundaries.
4. **Shadow Background Field Equalization:**
   - Low-frequency background illumination division ($61 \times 61$ kernel) to flatten horizontal and vertical sun glare across facade elevations.

### B. Physics-Based Crack Signal Processing (`neev/cv/crack_filters.py`)
1. **Multi-Scale Frangi Vesselness Filter:**
   - Computes Hessian matrix second-order partial derivatives across scales $\sigma \in (1.0, 2.0, 3.0)$.
   - Evaluates eigenvalue ratios: $\lambda_2 \gg \lambda_1 \approx 0$ with $R_B = |\lambda_1 / \lambda_2|$ and Frobenius matrix norm $S = \sqrt{\lambda_1^2 + \lambda_2^2}$.
   - High sensitivity to dark continuous linear fissures.
2. **Sato Continuous Ridge / Tubularity Filter:**
   - Second-order directional curvature across scales $\sigma \in (1.0, 2.0)$.
   - Complements Frangi by detecting faint continuous ridge profiles with low contrast.
3. **Multi-Scale Black Top-Hat (Bottom-Hat) Morphology:**
   - Dual-scale morphological closing subtraction: $\text{BottomHat}(I) = \text{Closing}(I, K) - I$ with kernel sizes $5 \times 5$ and $11 \times 11$.
   - Directly isolates narrow fissures darker than surrounding masonry.
4. **Directional Gabor Filter Bank:**
   - 8-angle directional filter bank ($\theta \in [0, \pi]$) with spatial wavelengths $\lambda \in (4.0, 8.0)$.
   - Resolves diagonal shear and settlement crack orientations.
5. **Gated Candidate Recovery Engine:**
   - Scale-adaptive proposal extraction with energy peak gating ($\max(\text{energy}) \ge 0.15$).
   - Geometric aspect ratio gate ($\text{aspect} \ge 2.0$, $\max(W, H) \ge 20$).
   - Top-3 candidate per-image quota to strictly prevent false-positive inflation on clean masonry negative controls.

### C. Segmentation & Morphological Reconstruction (`neev/cv/reconstruction.py`)
1. **Morphological Island Cleanup & Fissure Bridging:**
   - $3 \times 3$ morphological closing connects micro-discontinuities along fracture paths.
   - Small island suppression eliminates isolated 1-2 pixel noise specks.
2. **Zhang-Suen Medial Axis Skeletonization:**
   - Computes 1-pixel-wide topological medial axis of segmented defects.
3. **Topological Node Degree Analysis:**
   - 8-connected neighbor degree calculation:
     * Degree = 1: Endpoint of crack segment.
     * Degree = 2: Continuous crack path.
     * Degree $\ge$ 3: Branch / bifurcation node.
4. **Euclidean Distance-Transform Width Profiling:**
   - Measures defect width in native sensor pixels along skeleton coordinates without inventing uncalibrated metric dimensions.
5. **Region Defect Morphometrics:**
   - Area ($A$), perimeter ($P$), isoperimetric compactness ($4\pi A / P^2$), bounding box extent ($A / (W \times H)$), and boundary irregularity index ($P / \text{ConvexHull}(P)$).

### D. Multi-Defect Physical & Spectral Classifier (`neev/detection/multidefect.py`)
Classifies candidates into verified structural damage categories using combined physical and colorimetric evidence:
- **`crack`**: Linear medial axis skeleton $\ge 25$ px or aspect ratio $\ge 2.0$.
- **`corrosion_stain`**: Elevated $a^* \ge 138$ and $b^* \ge 145$ in LAB space, hue in $[5^\circ, 28^\circ]$ in HSV, surface rust ratio $\ge 0.12$.
- **`efflorescence`**: High lightness $L \ge 205$, low saturation $S \le 35$, diffuse non-linear shape.
- **`water_stain`**: Low lightness $L \le 85$, low saturation, vertical gravity elongation ($H > W$).
- **`spalling`**: High compactness $\ge 0.35$, high boundary irregularity $\ge 1.25$, area $\ge 150$ px.
- **`unknown_anomaly`**: Defect detected but geometry and color signature lack specificity; flagged with `requires_human_review = True`.

---

## 4. Software Dependencies Added

| Package | Version | Purpose | License | Hackathon Eligibility |
|:---|:---|:---|:---|:---|
| `opencv-python-headless` | `5.0.0.93` | CLAHE, morphological ops, connected components, Gaussian & bilateral filters | Apache 2.0 | Verified |
| `scikit-image` | `0.26.0` | Frangi vesselness, Sato tubularity, Zhang-Suen skeletonization | BSD 3-Clause | Verified |
| `scipy` | `1.16.2` | Bipartite matching (Hungarian algorithm), spatial distance transforms | BSD 3-Clause | Verified |
| `transformers` | `4.57.0` | SegFormer vision transformer model loading and FP16 inference | Apache 2.0 | Verified |
| `torch` / `torchvision` | `2.7.1+cu118` / `0.22.1` | CUDA tensor acceleration and FP16 automatic mixed precision | BSD-style | Verified |

---

## 5. Summary of Architecture Modes

| Mode Name | Components Executed | Target Role | Mean Latency (ms) | Peak VRAM (MB) |
|:---|:---|:---|:---:|:---:|
| `baseline` | Frozen SAM 3.1 text prompt `"crack"` (whole image) | Benchmark Reference | 7.3 ms (cached) | 8.1 MB |
| `advanced_sam` | SAM Whole + Adaptive Tiling + CCS + Guardrails | High-precision crack detection | 7.9 ms (cached) | 8.1 MB |
| `cv_enhanced_sam` | Advanced SAM + CLAHE/Bilateral + Gated Ridge Filters | Weak/faint defect recovery | 4760.2 ms | 0.0 MB |
| `facade_context` | Advanced SAM + SegFormer CMP Facade parser | Window reflection suppression | 360.0 ms | 337.6 MB |
| `multidefect_segformer` | Advanced SAM + SegFormer Crack & Spall Transformer | Multi-defect candidate generation | 461.4 ms | 393.8 MB |
| `sahi_cascaded` | SAHI 1:1 Native Slices + Two-Stage Cascade + Shadow Dual-Path | 4-Strategy High-Recall Inspection | 3090.2 ms | 251.5 MB |
| `full_inspection` | 4 Strategies + SegFormer Facade + Specialist Transformer + Multi-Defect Classifier | Complete Production Inspection Pipeline | 3146.5 ms | 251.5 MB |
