# NEEV: UI/UX & ENGINEERING KNOWN LIMITATIONS
**Milestone:** Complete Frontend Product Experience  
**Date:** November 4, 2026  
**Document Status:** Official Production Engineering Record  
**Repository Working Directory:** `d:\projects\metaglobal`  

---

## 1. Structural Engineering & Physical Boundaries

> [!WARNING]
> **CRITICAL LEGAL & STRUCTURAL DISCLAIMER**  
> Neev is strictly an **automated visual screening instrument**. It is **NOT** a certified structural health monitoring system, nor does it replace the statutory responsibilities of a licensed professional structural engineer.

1. **Depth Blindness of 2D Optical Photography:**
   - Conventional 2D photographs capture only surface-visible reflectance. They cannot measure crack penetration depth through concrete cover, delamination, rebar debonding, or subsurface voiding.
   - A fine visual surface crack may reflect a deep structural shear fracture, while a wide visual mark might only be superficial plaster crazing or paint peeling. Physical examination (hammer sounding, ultrasonic pulse velocity, or core sampling) is always mandatory.
2. **Absence of Proof vs. Proof of Absence:**
   - **Zero detected candidates does NOT mean the building is defect-free.**
   - Cracks obscured by shadows, vegetation, signage, grime, or inadequate photograph resolution will not be detected.
3. **Physical Scale Calibration (`mm` vs `px`):**
   - In uncalibrated photographs (lacking a fiducial scale target or known physical reference dimensions), crack lengths and apertures are reported strictly in **pixels (`px`)**.
   - Estimated millimeter values (`mm`) are only provided when explicit camera sensor metadata, ground-sampling distance (GSD), or reference markers are supplied.

---

## 2. Neural Model & Architecture Limitations

1. **Zero-Shot Segmentation Boundaries (Meta SAM 3.1):**
   - SAM 3.1 is frozen as the primary engine due to its superior precision (80.3%) and low false alarm rate (0.32 FP/image).
   - However, in scenes with extreme architectural textures (e.g., rustic stone masonry, exposed aggregate, deep mortar joints), SAM 3.1 may segment masonry joints as cracks unless multi-view filtering is active.
2. **High-Recall Hybrid Mode Tradeoff:**
   - The secondary `hybrid` mode (SAM + U-Net) boosts crack recall to 76.3% but drops precision to 71.4%. Evaluators must expect a higher volume of candidate false positives requiring human triage.
3. **Offline U-Net Standalone Limitations:**
   - The local U-Net model runs completely offline with 124ms latency and $0.00 cost, but has an F1 score of 0.3900 (Precision 63.7%, Recall 28.1%). It serves as a resilience fallback when cloud APIs are disabled, not as a replacement for SAM 3.1.
4. **Multimodal Vision-Language Advisory (Muse Wonder 1.0):**
   - Wonder 1.0 acts purely as an advisory cross-check. Because it operates on downsampled visual representations, it cannot replace pixel-level mask segmentations produced by SAM.

---

## 3. Image Capture & Environmental Constraints

1. **Illumination & Glare:**
   - Strong direct sunlight causing harsh specular glare or deep shadows impairs crack edge visibility.
   - Nighttime captures or low-light conditions with sensor noise (high ISO grain) trigger quality warnings and reduce model confidence.
2. **Oblique Perspective & Extreme Pitch Angles:**
   - Photographs taken from ground level aiming upwards at tall facade elevations suffer from severe perspective foreshortening.
   - While multi-view tiling mitigates this, severe distortion distorts true orientation angles and relative aperture measurements.
3. **File Size & Dimension Guards:**
   - Maximum upload size is strictly capped at **15 MiB** and image dimensions at **12,000 pixels** on any side to prevent server denial-of-service and memory exhaustion.

---

## 4. Frontend & Browser Compatibility

1. **Canvas Hardware Acceleration:**
   - The interactive viewport relies on HTML5 Canvas 2D rendering. On machines with disabled hardware acceleration, panning very large images (> 4000×4000 px) may exhibit slight stuttering.
2. **Offline Local Operation:**
   - When running completely offline without an active internet connection, external web fonts (`Inter`) gracefully fall back to local system sans-serif fonts (`-apple-system`, `BlinkMacSystemFont`, `Segoe UI`).
3. **No Database Requirement:**
   - The frontend consumes file-based records stored in `runs/inspections/`. Deleting or modifying files inside `runs/inspections/` manually on disk while the server is running without taking the `io_lock` may cause cache inconsistencies until refreshed.

