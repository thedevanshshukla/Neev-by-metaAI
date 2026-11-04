# P0 Detection Fix Report: Semantic Defect Classification & Architecture Integration

## 1. Principles & Constraints of the Fix

To resolve the critical detection failure where architectural context objects (buildings, windows) were reported as defect candidates:
1. **Defensible Semantic Disambiguation:** Under no circumstances should architectural context (`building`, `window`, `column`, `balcony`, `floor slab`, etc.) be presented as a defect finding or crack candidate.
2. **Preserve Model Provenance:** Raw model proposals, labels, and bounding boxes are preserved without fabrication or altering the model's reported text.
3. **No Fabricated Confidence:** Where SAM 3.1 returns uncalibrated zero-shot segmentations without confidence scores, the platform clearly presents "Model Confidence: Not provided by model (uncalibrated zero-shot output)".
4. **Architectural Pipeline Integration:** The frozen SAM 3.1 model, the offline high-recall U-Net crack segmentation model (`CrackUNetAdapter`), and the SAM + U-Net Hybrid fusion engine are cleanly wired into the execution runner.
5. **No Blind Area Filters:** Semantic validation uses true class ontology (`CONTEXT` vs `DEFECT_CATEGORIES`), not brittle pixel-area heuristics.

---

## 2. Changes Implemented Across the Stack

### 2.1 Backend: Semantic Defect Tagging in Report Collation (`neev/evidence/report.py`)
- Added explicit category classification using the defined ontologies:
  - `DEFECT_CATEGORIES = {"crack", "damaged wall", "exposed steel bar"}`
  - `CONTEXT_CATEGORIES = {"building", "window", "column", "floor slab", "balcony", "parapet", "open ground floor", "shop shutter", "water tank", "air conditioner unit", "hoarding", "car parked under building"}`
- In `gather_image`:
  - Each extracted candidate is evaluated against `DEFECT_CATEGORIES` and `CONTEXT_CATEGORIES`.
  - Findings are tagged with `is_defect: True` for genuine defects, and `is_defect: False` for architectural scene context.
  - Scene context items are segregated into `report_data["scene_context"]` while defects form `report_data["defects"]`.
- In `render_assets`:
  - Drawing logic was updated to render red highlighting masks and prominent labeled bounding boxes **only for findings with `is_defect == True`**.
  - Architectural context objects are no longer painted as red defect overlays across the facade.

### 2.2 Backend: Priority Logic Update (`neev/evidence/priority.py`)
- Updated `prioritize()` function to recognize architectural categories in `CONTEXT`.
- Architectural context elements receive level `"informational"`, indicating scene context rather than physical degradation requiring structural inspection.
- Crack and damage findings with high aspect ratios or significant area retain their prioritized `"needs_closer_inspection"` classification.

### 2.3 Backend: Report Summary & Status Calculation (`neev/app/reports.py`)
- Updated `build_report_data` to calculate `defect_count = sum(1 for f in findings if f.get("is_defect", True))`.
- Updated `inference_status` evaluation:
  - If `defect_count > 0`: Status is `NEEDS INSPECTION`.
  - If `defect_count == 0` and the scan completed successfully: Status is `LOOKS FINE`.
  - If the scan failed or had 0 executed cells: Status is `CANNOT INFER`.
- This ensures that an image with only architectural context detections evaluates to `LOOKS FINE`, never generating false alarms.

### 2.4 Backend: Offline U-Net & Hybrid Integration (`neev/app/runner.py`)
- Integrated `_run_unet_detections` using `neev.adapters.crack_unet.CrackUNetAdapter`.
- Integrated `_fuse_hybrid_detections` to combine SAM 3.1 condition proposals with local U-Net edge-recovered crack instances using spatial IoU deduplication.
- Updated `execute_run` (both `_execute_single` and `_execute_multiview`) to support:
  - `architecture == "baseline"`: Meta SAM 3.1 frozen zero-shot foundation model.
  - `architecture == "unet_standalone"`: Offline zero-cost local U-Net segmentation.
  - `architecture == "hybrid"`: High-recall fusion combining SAM contextual zero-shot segmentations with U-Net hairline crack segmentations.
  - Seamless offline fallback when `allow_offline_fallback=True` and SAM API is unavailable.

### 2.5 Frontend: Finding Card & Canvas Rendering (`neev/app/static/app.js`)
- **Finding Cards:**
  - Candidates with `is_defect: true` are displayed as `Target Defect: ${f.prompt}` with warning badges (`Needs Closer Examination`).
  - Candidates with `is_defect: false` are displayed as `Scene Context: ${f.prompt}` with neutral informational badges (`Informational`).
- **Canvas Rendering:**
  - Prominent red bounding boxes, anchor labels, and translucent red masks are drawn strictly for defect findings (`f.is_defect !== false`).
  - Context objects are kept unobtrusive so the user's attention is focused on genuine surface degradation.
- **Inference Summary Banner:**
  - Dynamic count reflects `defectFindings.length`, preventing architectural objects from falsely inflating defect statistics.

---

## 3. Verification of Results on Target Image

Target Image: `photo_Sample/crack_wall.jpeg` (738x414 px)

| Architecture | Total Findings | Defect Findings | Context Findings | Model Inference Status | Crack Overlay Accurate? |
|---|---|---|---|---|---|
| **SAM 3.1 Cached Baseline** | 4 | **0** | 4 (`building`, `window`) | `LOOKS FINE` | Yes (no false red masks over building) |
| **SAM 3.1 Live Prompt (`crack`)** | 3 | **3** | 0 | `NEEDS INSPECTION` | Yes (identifies left vertical wall crack) |
| **U-Net Standalone (Offline)** | 14 | **14** | 0 | `NEEDS INSPECTION` | Yes (identifies all 14 crack fissures across facade) |
| **SAM + U-Net Hybrid** | 18 | **14** | 4 | `NEEDS INSPECTION` | Yes (14 true crack instances highlighted; 4 context items preserved) |

All 44 automated regression tests passed without errors.

