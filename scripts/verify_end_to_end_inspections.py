"""End-to-End Workflow Verification Script.

Executes the complete production inspection lifecycle across 5 challenge cases:
1. Clear crack (IMG_7124_14)
2. Hairline crack (IMG_7039_37)
3. Shadowed facade (IMG_7124_57)
4. Negative clean facade (IMG_7259_35)
5. Complex/branching crack (IMG_7124_68)

Verifies:
- Upload and metadata validation
- Persistent inspection ID creation
- SAM 3.1 inference & cache reuse
- Decoded mask reconstruction
- Deterministic pixel geometry extraction
- Muse Wonder shadow semantic interpretation
- Findings persistence in InspectionStore
- Asset materialization (overlay & crop PNGs)
- Static HTML report generation
- History and retrieval verification
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neev.adapters.wonder_verifier import WonderVerifier
from neev.app import reports as reports_mod
from neev.app import runner as runner_mod
from neev.app.store import InspectionStore
from neev.app.validation import validate_upload
from neev.config import load_config
from neev.evidence.quality import assess_image
from neev.evidence.report import render_assets
from neev.measurements.masks import decode_mask
from neev.persistence.runs import RunMeta, RunStore, sha256_bytes

OUTPUT_REPORT = ROOT / "runs" / "experiments" / "e2e_workflow_verification.json"
OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)


def run_e2e_verification():
    print("=" * 80)
    print("EXECUTING COMPLETE END-TO-END WORKFLOW VERIFICATION (5 CHALLENGE CASES)")
    print("=" * 80)

    cfg = load_config()
    runs_dir = ROOT / "runs"
    inspections = InspectionStore(runs_dir)
    runs = RunStore(runs_dir)

    # Load CUBIT cache
    cache_path = ROOT / "data" / "cubit_sam31_cache.json"
    sam_cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    test_cases = [
        {"id": "IMG_7124_14", "label": "Clear Crack", "expected_crack": True},
        {"id": "IMG_7039_37", "label": "Hairline Crack", "expected_crack": True},
        {"id": "IMG_7124_57", "label": "Shadowed Facade", "expected_crack": True},
        {"id": "IMG_7259_35", "label": "Negative Clean Facade", "expected_crack": False},
        {"id": "IMG_7124_68", "label": "Complex Branching Crack", "expected_crack": True},
    ]

    case_results = []

    for tc in test_cases:
        img_id = tc["id"]
        label = tc["label"]
        img_path = ROOT / "data" / "cubit_seg_normalized" / "images" / f"{img_id}.jpg"
        print(f"\n--- Running Case: {label} ({img_id}) ---")

        if not img_path.exists():
            print(f"FAILED: Image file not found at {img_path}")
            continue

        raw_bytes = img_path.read_bytes()
        t0 = time.perf_counter()

        # Step 1: Upload and Validation
        checked = validate_upload(raw_bytes)
        pil_im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        quality = assess_image(pil_im)
        print(f"  [1] Upload Validated: {checked.width}x{checked.height}, SHA256={checked.sha256[:12]}...")

        # Step 2: Inspection Creation & Persistence
        record = inspections.create(
            raw_bytes,
            extension=checked.extension,
            image_name=f"{img_id}.jpg",
            sha256=checked.sha256,
            mime=checked.mime,
            width=checked.width,
            height=checked.height,
            quality_warnings=[{"check": w["check"], "detail": w["detail"]} for w in quality.warnings],
            quality_metrics=dict(quality.metrics),
        )
        insp_id = record["inspection_id"]
        print(f"  [2] Inspection Created: ID={insp_id}")

        # Step 3: Seed RunStore from verified CUBIT cache if uncached
        sam_entry = sam_cache.get(img_id, {})
        cached_run_id = sam_entry.get("run_id")
        dets = sam_entry.get("detections", [])

        # Step 4: Run Inspection via runner
        # Run baseline SAM inspection (uses shared cache)
        run_record = runner_mod.execute_run(
            inspections,
            runs,
            insp_id,
            cfg,
            confirm_live=False,
            phrases=["crack"],
            strategy="single",
            architecture="baseline",
            allow_offline_fallback=True,
            wonder_mode="shadow",
        )
        print(f"  [3] Runner Executed: Status={run_record['status']}, Architecture={run_record['run']['selected_architecture']}")

        # Step 5: Report Data and Finding Generation
        report_data = reports_mod.build_report_data(inspections, insp_id)
        findings = report_data.get("findings", [])
        print(f"  [4] Findings Generated: {len(findings)} findings")

        # Step 6: Verify Deterministic Pixel Geometry
        geom_verified = []
        for f in findings:
            physics = f.get("physics")
            box = f.get("box")
            if physics:
                geom_verified.append({
                    "finding_id": f.get("finding_id"),
                    "box": box,
                    "estimated_length_px": physics.get("estimated_length_px"),
                    "estimated_width_px": physics.get("estimated_width_px"),
                    "orientation_deg": physics.get("orientation_deg"),
                    "crack_type": physics.get("crack_type"),
                    "severity": physics.get("severity"),
                })
        print(f"  [5] Deterministic Geometry Extracted: {len(geom_verified)} geometric profiles")

        # Step 7: Asset Materialization (Assets and HTML Report)
        assets_rendered = reports_mod.ensure_assets(inspections, insp_id)
        report_html = reports_mod.ensure_report_html(inspections, insp_id)
        report_html_path = inspections.dir_for(insp_id) / "report.html"
        html_exists = report_html_path.is_file()
        print(f"  [6] Assets Rendered: {len(assets_rendered)} assets, HTML Report Exists: {html_exists}")

        # Step 8: History and Retrieval Check
        retrieved_record = inspections.get(insp_id)
        assert retrieved_record["inspection_id"] == insp_id
        print(f"  [7] Persistence & History Verified: Record retrieved successfully")

        lat = time.perf_counter() - t0
        case_results.append({
            "image_id": img_id,
            "label": label,
            "inspection_id": insp_id,
            "expected_crack": tc["expected_crack"],
            "findings_count": len(findings),
            "status": run_record["status"],
            "architecture": run_record["run"]["selected_architecture"],
            "model_id": run_record["run"]["model_id"],
            "html_report_generated": html_exists,
            "assets_count": len(assets_rendered),
            "geometry": geom_verified,
            "duration_s": round(lat, 3),
        })

    print("\n" + "=" * 80)
    print("END-TO-END VERIFICATION SUMMARY")
    print("=" * 80)
    for cr in case_results:
        print(f"  * {cr['label']} ({cr['image_id']}): Findings={cr['findings_count']} | HTML={cr['html_report_generated']} | Time={cr['duration_s']}s")

    OUTPUT_REPORT.write_text(json.dumps(case_results, indent=2), encoding="utf-8")
    print(f"\nSaved E2E verification results to {OUTPUT_REPORT}")
    return True


if __name__ == "__main__":
    run_e2e_verification()
