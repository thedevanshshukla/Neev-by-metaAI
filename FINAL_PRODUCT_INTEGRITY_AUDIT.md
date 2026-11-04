# NEEV FINAL PRODUCT INTEGRITY AUDIT
## Comprehensive Scientific Claim Verification, Engineering Terminology Standardization & Discrepancy Reconciliation

**Date:** November 4, 2026  
**Auditor:** Quality & Scientific Integrity Subsystem  
**Application Scope:** All UI/UX Interfaces (`index.html`, `app.js`, `styles.css`), Static Reports (`report.html`), and API Response Payloads  
**Status:** ALL 9 INTEGRITY RULES AUDITED & ENFORCED (`VERIFIED`)

---

## 1. Context & Objectives

Building facade safety assessment is a high-stakes engineering discipline where inaccurate AI claims can introduce severe liability. An AI model analyzing a 2D optical photograph cannot determine concrete depth, internal rebar corrosion, rebar yield stress, foundation settlement, or structural safety code compliance.

This audit establishes strict scientific boundaries for Neev. Every visual label, badge, tooltip, report section, and demo card has been audited to ensure that optical visual observations are never conflated with physical engineering conclusions.

---

## 2. Nine-Point Scientific Integrity Audit

### Rule 1: Seismic Multipliers & Structural Risk Scores
- **Audit Requirement:** Eliminate all uncalibrated multipliers (e.g., "1.5x Seismic Multiplier") or synthetic "Structural Risk Scores" (e.g., "High Risk: 8.7/10") that lack physical sensor calibration or finite element modeling.
- **Previous Implementation State:** The finding card and report previously displayed `Seismic Factor: 1.5x` for diagonal cracks.
- **Correction Applied:**
  - Removed all multiplier formulas and arbitrary risk scores from `neev/app/static/index.html` and `neev/app/static/app.js`.
  - Replaced with standard structural screening metadata: `Orientation Pattern: In-plane diagonal pattern ~36.1° (physical cause undetermined)`.
- **Audit Verification:** Grepped codebase for `seismic`, `risk_multiplier`, and `risk_score`. Confirmed zero uncalibrated risk scoring in active UI templates.
- **Status:** `VERIFIED`

---

### Rule 2: 2D Image Geometry vs Definitive Structural Failure Mechanisms
- **Audit Requirement:** Optical 2D crack traces must not be labeled as definitive failure mechanisms (e.g., claiming a diagonal crack is definitively "Shear Failure" or "Settlement Cracking").
- **Previous Implementation State:** Finding cards labeled diagonal features as `Mechanism: Shear Failure`.
- **Correction Applied:**
  - Standardized terminology across the application to: `Visual Geometric Pattern: In-plane diagonal pattern ~36.1°; physical cause undetermined`.
  - Added engineering disclaimer on workbench: *"2D visual pattern recognition does not determine internal stress states or structural failure mechanisms. Physical intrusive testing and licensed engineering analysis are required."*
- **Audit Verification:** Confirmed that finding inspection cards render visual geometric orientation without diagnostic assertion.
- **Status:** `VERIFIED`

---

### Rule 3: 2D Pixel Measurements vs Physical Aperture & Depth
- **Audit Requirement:** Pixel measurements must be explicitly labeled as uncalibrated 2D pixel counts (`px`). Under no circumstances may pixel counts be converted to physical crack aperture (mm) or crack depth without calibrated photogrammetric targets.
- **Previous Implementation State:** Finding metrics labeled crack width as `Aperture (Width): 184 px`.
- **Correction Applied:**
  - Replaced label with: `Image Trace Width (Max): 184 px (uncalibrated 2D pixel count; cannot determine physical aperture or depth)`.
  - Clarified that optical images cannot resolve crack depth beneath the facade surface.
- **Audit Verification:** Confirmed across both web UI finding cards and static printable inspection reports.
- **Status:** `VERIFIED`

---

### Rule 4: Adaptive Multi-View Tiling & Hairline Cracks
- **Audit Requirement:** High-resolution tile cropping must not claim to guarantee detection of all sub-millimeter hairline cracks.
- **Previous Implementation State:** Tile feature description implied "Guaranteed hairline crack detection via adaptive zoom".
- **Correction Applied:**
  - Updated description in `#card-multiview` to: *"Evaluates 2×2 overlapping tiles at higher effective pixel resolution; does not guarantee detection of all hairline cracks."*
  - Documented physical sensor limits (ground sampling distance, optical blur, and contrast limits).
- **Audit Verification:** Verified in `neev/app/static/index.html` line 258 and `app.js`.
- **Status:** `VERIFIED`

---

### Rule 5: Structural Safety, Habitability & Certification Disclaimers
- **Audit Requirement:** Neev must never present an inspection outcome as a structural safety certification, building habitability clearance, or municipal code sign-off.
- **Previous Implementation State:** Missing prominent legal/engineering disclaimer in the main header and report view.
- **Correction Applied:**
  - Added persistent global engineering disclaimer in the header and footer:
    > *"Neev is an AI-assisted visual screening tool for preliminary facade inspection. It is not a structural safety certification, habitability assessment, or replacement for a licensed structural engineer."*
  - Added dedicated compliance notice on all generated printable reports (`/reports/:id`).
- **Audit Verification:** Verified presence in DOM (`#app-disclaimer-banner`), CSS styling, and test coverage in `tests/test_app.py`.
- **Status:** `VERIFIED`

---

### Rule 6: Negative Control Representation (`fa1.png`)
- **Audit Requirement:** Never claim "Zero false alarms on clean facade" unless the ground truth actually represents a clean facade.
- **Previous Implementation State:** `fa1.png` was described as a "Clean facade / 0 defects / pristine building".
- **Ground Truth Reality:** Expert annotations in `eval/annotations_label_studio.json` reveal that `fa1.png` contains **15 spalling annotations** and over 40 building components.
- **Correction Applied:**
  - Accurately re-framed the demonstration:
    - Card Title: `Crack Screening Specificity (Negative Control)`
    - Description: *"Evaluates prompt specificity against complex facade textures with no crack features. Yields 0 candidate crack detections. (Note: Contains concrete spalling not targeted by the crack-specific model)."*
  - Inspection banner clearly informs evaluators of the distinction between defect classes.
- **Audit Verification:** Inspected showcase card rendering in Playwright browser tests (`10_showcase_fa1_negative_control.png`).
- **Status:** `VERIFIED`

---

### Rule 7: Visual Prominence vs Structural Severity
- **Audit Requirement:** Distinguish visual prominence (apparent pixel area and contrast on the 2D photograph) from structural severity (criticality to building load-bearing capacity).
- **Previous Implementation State:** Large crack findings were labeled `Severity: structural_severe`.
- **Correction Applied:**
  - Renamed metric to: `Visual Prominence: prominent (based on 2D pixel extent; does not indicate structural severity)`.
  - Added tooltip explaining that a thin hairline crack at a critical beam-column joint may have extreme structural severity while possessing low visual prominence, whereas wide surface stucco cracking may have high visual prominence with minimal structural consequence.
- **Audit Verification:** Verified across all finding cards in `app.js`.
- **Status:** `VERIFIED`

---

### Rule 8: Candidate Findings vs Confirmed Defects
- **Audit Requirement:** AI detections must be identified as candidate findings rather than confirmed defects.
- **Previous Implementation State:** Showcase cards and header counters labeled results as `Confirmed Defects: 14`.
- **Correction Applied:**
  - Replaced all instances of `Confirmed Defects` with `Candidate Cracks` or `Candidate Findings`.
  - Updated showcase gallery cards:
    - `dm1.png`: `14 Candidate Cracks`
    - `h2.png`: `10 Candidate Cracks`
    - `dv2.png`: `3 Candidate Cracks`
    - `fa1.png`: `0 Candidate Cracks (Negative Control)`
- **Audit Verification:** Verified DOM node text contents and automated test assertions.
- **Status:** `VERIFIED`

---

### Rule 9: Model Confidence vs Calibrated Probability
- **Audit Requirement:** Uncalibrated neural network output logits or heuristic confidence scores must not be represented as the true statistical probability of a defect.
- **Previous Implementation State:** Label was formatted as `Probability of Failure: 94%`.
- **Correction Applied:**
  - Replaced with: `Model Confidence: 0.94 (uncalibrated neural network detection score)`.
  - Preserved raw confidence values without deceptive probability claims.
- **Audit Verification:** Verified in finding details drawer and API serialization schema.
- **Status:** `VERIFIED`

---

## 3. Comprehensive Scientific Claim Audit Table

| Scientific Integrity Rule | Target Surface | Previous Vulnerability | Current Implementation | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Rule 1: Seismic Multipliers** | Finding Drawer & Reports | Uncalibrated `1.5x` multiplier | Removed; replaced with visual geometric orientation | `VERIFIED` |
| **Rule 2: Failure Mechanisms** | Finding Cards | Asserted "Shear Failure" | Labeled "Visual Geometric Pattern; physical cause undetermined" | `VERIFIED` |
| **Rule 3: Pixel Measurements** | Metrics Badges | Labeled "Aperture: 184 px" | Labeled "Image Trace Width (Max): 184 px (uncalibrated 2D pixel count)" | `VERIFIED` |
| **Rule 4: Multi-View Tiling** | Dashboard Form | Implied guaranteed hairline detection | Explicit disclaimer: "does not guarantee detection of all hairline cracks" | `VERIFIED` |
| **Rule 5: Structural Safety** | Global Header & Reports | Absent legal disclaimer | Prominent disclaimer: preliminary screening tool; not safety certification | `VERIFIED` |
| **Rule 6: Negative Control** | Showcase `fa1.png` | Falsely claimed "clean facade" | Accurately framed as crack-specificity negative control (15 spalling in GT) | `VERIFIED` |
| **Rule 7: Visual Prominence** | Findings List | Conflated pixel size with structural risk | Explicit distinction between 2D visual prominence and structural severity | `VERIFIED` |
| **Rule 8: Candidate Findings** | Summary Counts | Labeled "Confirmed Defects" | Standardized as "Candidate Cracks" / "Candidate Findings" | `VERIFIED` |
| **Rule 9: Model Confidence** | Score Readouts | Labeled "Probability of Failure" | Standardized as "Model Confidence (uncalibrated detection score)" | `VERIFIED` |

---

## 4. Conclusion

Through these 9 rigorous corrections, Neev maintains the highest standards of scientific and product integrity. The platform communicates its preliminary AI screening capabilities with technical precision while strictly preventing deceptive or uncalibrated structural engineering claims.

