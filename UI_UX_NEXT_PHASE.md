# UI/UX Transition Plan & Next Phase Specification

> **Milestone Status:** Backend Frozen & Verified | UI/UX Redesign Deferred  
> **Prepared For:** Next Phase Dedicated UI/UX Sprint  
> **Target Date:** November 5, 2026  

---

## 1. Deferral Confirmation & Current Frontend Baseline

As mandated by engineering milestone rules, **all cosmetic and visual redesigns were strictly deferred** during this milestone to protect backend stabilization, API integration, and mathematical benchmark integrity. The existing UI codebase (`neev/app/static/index.html`, `styles.css`, `app.js`) remains functional, responsive, and aligned with backend contracts.

### Current Visual & Architectural Baseline:
- **Typography:** Inter (400, 500, 600, 700).
- **Color Palette:** Warm neutral stone (`#FFFEFC`, `#F0EEE9`, `#E8E4DE`), charcoal text (`#292735`, `#5C5865`), brand purple (`#6558C9`), amber warning banner (`#FEF3C7`).
- **Layout:** Single-page responsive workflow comprising:
  1. Health & capability check bar.
  2. Mandatory engineering disclaimer banner.
  3. Image upload dropzone with client-side file validation.
  4. Inspection configuration panel.
  5. Interactive defect findings card deck with asset previews.

---

## 2. Known Frontend Limitations & Technical Debt

1. **Architecture Mode Selection Absent:**
   - The current UI exposes only execution strategy (`single` vs `multiview`), without exposing the 3 official frozen architecture modes: `baseline` (SAM 3.1), `hybrid` (SAM + U-Net), and `unet_standalone` (Offline Edge).
2. **Wonder Semantic Verification Controls Missing:**
   - The user currently cannot toggle Muse Wonder modes (`disabled`, `shadow`, `advisory`) from the web interface.
3. **Execution Telemetry Not Surfaced:**
   - The runner records model provenance, request IDs, token usage, and live spend, but the UI currently only displays `status: complete`.
4. **Coarse Progress Feedback:**
   - Long-running inference (~1.5s for SAM, ~12s for Wonder) displays a basic text spinner without pipeline step tracking.
5. **Static Finding Cards:**
   - Finding cards display bounding boxes and crop thumbnails, but lack interactive overlay opacity sliders and high-resolution zoom/pan capability.

---

## 3. Required Frontend Integration Changes

The backend now fully supports explicit configuration parameters in `POST /inspections/{id}/run`. The frontend must be updated to pass these fields:

```javascript
// Updated RunRequest payload to be wired in app.js
const runPayload = {
  confirm_live: true,
  phrases: ["crack"],
  strategy: "single",
  architecture: selectedArchitecture, // "baseline" | "hybrid" | "unet_standalone"
  allow_offline_fallback: allowOfflineFallbackCheckbox.checked,
  wonder_mode: selectedWonderMode,   // "disabled" | "shadow" | "advisory"
};
```

And in `report-data` rendering:
- Display `selected_architecture` and `actual_architecture_executed`.
- Render advisory badges on findings:
  - `<span class="badge badge-supports">Wonder Verified</span>`
  - `<span class="badge badge-doubts">Wonder Disputed (Human Review Req.)</span>`

---

## 4. Priority Roadmap for Next UI/UX Milestone

### Phase 1: High-Priority Functional Controls (P0)
- **Architecture Selector Card Group:**
  - Card A: **Live Meta SAM 3.1 (Default)** - High-fidelity cloud foundation screening.
  - Card B: **SAM + U-Net Hybrid** - Maximum crack recall for critical audits.
  - Card C: **Crack U-Net Offline Edge** - Zero cost, 124 ms local on-device screening.
- **Muse Wonder Verification Toggle:**
  - Toggle switch between `Shadow Mode` (backend telemetry only), `Advisory Mode` (flags disputes in report), and `Off`.
- **Wire Audit Telemetry:**
  - Display actual API model version, request ID, and billed cost in the inspection metadata header.

### Phase 2: Inspection Experience & Visual Viewer (P1)
- **Interactive Canvas Overlay:**
  - Canvas overlay with layer toggles (original photo, SAM segmentation mask, U-Net micro-cracks, bounding boxes).
  - Opacity slider (0% to 100%) for seamless defect comparison against concrete substrate.
- **Pipeline Progress Stepper:**
  - Multi-step timeline: `Upload` $\rightarrow$ `SAM Segmentation` $\rightarrow$ `Geometry Extraction` $\rightarrow$ `Semantic Verification` $\rightarrow$ `Report Ready`.

### Phase 3: Export & Deliverable Polish (P2)
- **Professional PDF Export:**
  - Clean `@media print` CSS stylesheet formatting findings into an executive structural screening memo with photo proof, disclaimer banners, and geometric summaries.
- **Inspection History Dashboard:**
  - Enhanced `/reports` overview table showing past inspections, date, finding counts, architecture used, and quick report links.

