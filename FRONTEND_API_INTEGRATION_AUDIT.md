# NEEV: FRONTEND-API INTEGRATION AUDIT
**Milestone:** Complete Frontend Integration Verification  
**Date:** November 4, 2026  
**Status:** 100% Passed & Verified  
**Repository Working Directory:** `d:\projects\metaglobal`  

---

## 1. Executive Summary

This audit validates the complete contractual alignment between the frontend user interface (`neev/app/static/`) and the backend REST endpoints implemented in FastAPI (`neev/app/main.py`, `neev/app/runner.py`, `neev/app/store.py`, and `neev/app/reports.py`).

Every endpoint consumed by the frontend was audited for:
1. HTTP method, request payload schema, and content type.
2. Response schema, JSON structure, and binary asset streaming.
3. Concurrency safety under the global `io_lock`.
4. Cost guardrails and zero unintentional billable API calls.

---

## 2. Endpoint Contract Verification Matrix

| Endpoint | Method | Frontend Consumer | Payload / Parameters | Response Contract | Status |
| :--- | :---: | :--- | :--- | :--- | :---: |
| **`/health`** | `GET` | Navigation bar status pill & startup check | None | `{"status": "ok", "app": "neev-inspection", "masks_available": bool, "live_configured": bool}` | **PASS** |
| **`/`** | `GET` | Root application URL | None | Static HTML5 document (`index.html`), 200 OK | **PASS** |
| **`/styles.css`** | `GET` | Styling stylesheet | None | CSS stylesheet (`styles.css`), `text/css` | **PASS** |
| **`/app.js`** | `GET` | Pure vanilla script | None | JavaScript application (`app.js`), `application/javascript` | **PASS** |
| **`/reports`** | `GET` | Dashboard & History tables | None | `{"reports": [InspectionSummary, ...], "count": int}` | **PASS** |
| **`/inspections`** | `POST` | New Inspection upload dropzone | `multipart/form-data` with `file: bytes` | 201 Created with full inspection record metadata (no inference triggered) | **PASS** |
| **`/inspections/{id}`** | `GET` | Router & deep link validation | Path parameter `id: str` | Inspection record JSON, 200 OK | **PASS** |
| **`/inspections/{id}/run`** | `POST` | Execute Screening action | `RunRequest` (see Section 3) | Updated inspection record with run metadata and audit trace | **PASS** |
| **`/inspections/{id}/report-data`** | `GET` | Evidence Explorer & Finding cards | Path parameter `id: str` | Comprehensive report payload: findings, boxes, physics, assets, audit | **PASS** |
| **`/inspections/{id}/image`** | `GET` | Canvas viewport base layer | Path parameter `id: str` | Original image bytes with magic-sniffed MIME type | **PASS** |
| **`/inspections/{id}/assets/{name}`**| `GET` | Canvas viewport overlay & crop images | `name: str` (`{id}_overlay.png`, `{id}_{finding}.png`) | PNG image bytes, 200 OK | **PASS** |
| **`/inspections/{id}/report`** | `GET` | "Printable HTML Report" button | Path parameter `id: str` | Standalone offline HTML report page | **PASS** |

---

## 3. Detailed Request & Response Schemas

### 3.1 Screening Execution Contract (`POST /inspections/{id}/run`)
The frontend submits the following validated payload:
```json
{
  "confirm_live": false,
  "phrases": ["crack", "crack in wall"],
  "strategy": "multiview",
  "architecture": "baseline",
  "allow_offline_fallback": true,
  "wonder_mode": "shadow"
}
```

#### Field Semantics:
- `confirm_live` (boolean, default: `false`): When `false`, inference operates strictly over existing cached benchmark traces or local models, guaranteeing **$0.00 billable API spend**.
- `phrases` (list of strings, 1 to 32 items): Defect categories to screen.
- `strategy` (`"single"` or `"multiview"`): Controls whether the image is processed as a single full frame or tiled with intelligent 2×2 overlaps for hairline crack recovery.
- `architecture` (`"baseline"` | `"hybrid"` | `"unet_standalone"`): Selects between Meta SAM 3.1 (Primary), SAM+U-Net Hybrid, or local PyTorch U-Net.
- `allow_offline_fallback` (boolean): When enabled, transient network failures fall back gracefully to local U-Net.
- `wonder_mode` (`"disabled"` | `"shadow"` | `"advisory"`): Configures secondary multimodal vision-language verification.

### 3.2 Report Data Contract (`GET /inspections/{id}/report-data`)
The frontend receives a normalized, JSON-safe data structure:
```json
{
  "inspection_id": "e254a194d2b7",
  "image_name": "h2.png",
  "width": 1024,
  "height": 1024,
  "sha256": "713e717e089225725f0adfae36ae257dae5927ad74eb8f72a50a36fcad25a4c9",
  "status": "complete",
  "has_run": true,
  "quality_warnings": [],
  "findings": [
    {
      "finding_id": "crack#0",
      "prompt": "crack",
      "display_label": "possible crack",
      "category": "crack",
      "box": {"x0": 575, "y0": 436, "x1": 827, "y1": 620},
      "area_px": 46368,
      "area_frac": 0.0442,
      "crop_url": "/inspections/e254a194d2b7/assets/e254a194d2b7_crack#0.png",
      "priority": "needs_closer_inspection",
      "physics": {
        "length_px": 312.0,
        "max_width_px": 184.0,
        "mean_width_px": 147.2,
        "orientation_deg": 36.1,
        "orientation_type": "diagonal_shear",
        "severity": "structural_severe",
        "seismic_risk_factor": 1.5,
        "failure_mechanism": "Observed Diagonal Linear Fracture (~45° in-plane angle, Seismic In-Plane Shear Pattern)",
        "engineering_action": "Visual screening flag: diagonal crack pattern observed. Recommend physical examination by a qualified structural engineer.",
        "code_reference": "Visual screening benchmark (IS 13935 / FEMA P-154 screening guide)",
        "is_calibrated": false
      }
    }
  ],
  "assets": {
    "image": "/inspections/e254a194d2b7/image",
    "overlay": "/inspections/e254a194d2b7/assets/e254a194d2b7_overlay.png",
    "report": "/inspections/e254a194d2b7/report"
  },
  "audit": {
    "selected_architecture": "baseline",
    "actual_architecture_executed": "baseline",
    "model_id": "sam-3.1",
    "inference_status": "complete",
    "wonder_mode": "shadow"
  }
}
```

---

## 4. Concurrency & Locking Safety

The application uses FastAPI serving multiple concurrent browser requests:
- When a user views an inspection, the frontend triggers requests for `/report-data`, `/image`, `/assets/..._overlay.png`, and multiple `/assets/..._crack#*.png` thumbnails concurrently.
- The backend protects all directory reads and asset generation with `neev.app.store.io_lock` (a re-entrant threading lock).
- This prevents race conditions where asset rendering could collide with inspection invalidation or partial file writes on Windows OS.

---

## 5. Cost & Safety Guardrails Audit

1. **Zero Uncontrolled Network Calls:** The frontend default setting is `confirm_live: false`. All showcase demo inspections replay pre-cached run traces directly from `RunStore`.
2. **Explicit UI Confirmation:** When a user checks "Confirm Live API Inference", an explicit warning banner appears highlighting that uncached cells will incur Meta API charges ($0.0025 per image).
3. **MIME Verification:** All uploads are validated via binary magic bytes (`sniff_image_mime`) rather than client-provided file extensions, preventing malicious uploads or malformed image processing.

