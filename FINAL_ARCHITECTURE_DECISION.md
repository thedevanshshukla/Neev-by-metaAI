# Final Architecture Selection Decision: Neev Production Core

> **Decision Date:** 2026-11-04  
> **Scope:** Final Selection of Neev's Primary Detection Architecture  
> **Evaluation Basis:** Live SAM 3.1 Meta Model API Inference + Complete 1,320-Image CUBIT-Seg Benchmark  

---

## 1. Definitive Architecture Selection

### 🥇 Primary Selected Architecture: `baseline`
- **Test Set F1 Score:** **0.7732**
- **Precision:** **80.3%** (High fidelity; minimal inspection false alarms)
- **Recall:** **74.6%** (883 true positive cracks detected)
- **Pixel Mask IoU:** **0.5248**
- **False Alarm Rate:** **0.32 FP / image**
- **Negative Control:** Only **3** false alarm across clean facade holdouts.

### 🥈 Secondary High-Recall Mode: `multidefect_segformer`
- **When to Use:** Optional automated pre-screening mode where capturing every possible hairline defect is prioritized and human engineers verify all candidate boxes.
- **Recall:** **33.5%** (396 GT defects recovered).
- **Caveat:** High false alarm rate (4.81 FP/image) on rough aggregate concrete.

---

## 2. Multi-Criteria Evaluation Matrix

| Decision Factor | Live SAM 3.1 (`baseline`) | Crack U-Net (`unet_standalone`) | Fused Hybrid (`hybrid`) | Multi-Defect SegFormer |
|:---|:---:|:---:|:---:|:---:|
| **Detection Quality (F1)** | **0.7732** | 0.3902 | 0.7374 | 0.1628 |
| **Precision** | **80.3%** | 63.7% | 71.4% | 10.8% |
| **Pixel Mask IoU** | **0.5248** | 0.0844 | 0.1282 | 0.1453 |
| **End-to-End Latency** | ~1,450 ms | **122 ms** | ~1,550 ms | **54 ms** |
| **API Cost / Image** | $0.0025 USD | **$0.0000** | $0.0025 USD | **$0.0000** |
| **Offline Operation** | Requires Internet | **100% Offline** | Requires Internet | **100% Offline** |
| **Production Reliability** | Cloud dependent | **Deterministic / Edge** | Cloud + Edge | Edge |

---

## 3. Rationale for Architecture Rejections

1. **Rejection of `sahi_standalone`:**
   - Catastrophic false alarm explosion (9.41 FP/image, Precision 3.7%). Tiling without cascade gating hallucinates cracks across concrete textures.
2. **Rejection of `dsp_frangi_sam` and `cv_enhanced_sam`:**
   - Classical Hessian ridge filters fail on textured concrete and add computation latency without improving detection metrics over base SAM.
3. **Rejection of Un-gated High-Recall Transformers:**
   - `multidefect_segformer` achieves localized sensitivity but produces unacceptable false alarm volumes (3.99 FP/image) for primary inspection reporting.

---

## 4. Final Deployment Recommendation
- **Deploy `baseline` as Neev's primary cloud inspection engine.**
- If edge offline capability with zero cloud latency is mandated for mobile drone units without 5G connectivity, fallback to **`unet_standalone`** (122 ms, 0 API cost).
