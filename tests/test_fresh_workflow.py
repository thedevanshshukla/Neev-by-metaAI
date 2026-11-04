"""Integration test: Fresh image workflow with mocked deterministic inference.

Verifies end-to-end lifecycle for a brand-new image without using pre-existing
cached data:
- Upload fresh image through /inspections
- Verify storage and dimensions
- Trigger adaptive multi-view inspection via /inspections/{id}/run with mocked transport
- Verify coordinate transformations, prediction aggregation, and findings persistence
- Verify report-data contract, overlays, and evidence crops
- Verify static HTML report rendering
- Verify asset loading (original image, overlay, crops)
- Verify appearance in inspection history (/reports)
"""
import io
from pathlib import Path
from PIL import Image, ImageDraw
import pytest

from neev.config import Config
from neev.app.main import create_app
from fastapi.testclient import TestClient


class DeterministicMockTransport:
    """Mock transport returning predictable SAM 3.1 detection strings."""

    def __init__(self):
        self.calls = []

    def post_json(self, url, headers, payload, timeout=0):
        self.calls.append({"url": url, "payload": payload})
        # Generate predictable bounding boxes:
        # Whole view gets one box; tiles get another box
        # Box format: <0f>0<|box;x1=100;y1=120;x2=350;y2=280;w=1200;h=900|>
        call_idx = len(self.calls)
        if call_idx == 1:
            # Whole view detection
            box_str = "<0f>0<|box;x1=150;y1=180;x2=320;y2=310;w=1200;h=900|>"
        else:
            # Tile view detection
            box_str = f"<0f>0<|box;x1=50;y1=60;x2=200;y2=180;w=685;h=514|>"
        return {
            "output_text": box_str,
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": box_str}],
                }
            ],
        }


def make_fresh_synthetic_image(width=1200, height=900) -> bytes:
    """Generate a unique synthetic facade image (1200x900) with geometric patterns."""
    img = Image.new("RGB", (width, height), color=(220, 215, 205))
    draw = ImageDraw.Draw(img)
    # Draw facade-like windows and brick courses
    for y in range(50, height - 100, 150):
        for x in range(60, width - 100, 180):
            draw.rectangle([x, y, x + 100, y + 90], fill=(80, 100, 120), outline=(50, 60, 70))
    # Draw a simulated crack line
    draw.line([(200, 200), (250, 260), (270, 310)], fill=(40, 30, 30), width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_fresh_image_mocked_end_to_end_workflow(tmp_path):
    # 1. Setup isolated application with mock transport
    transport = DeterministicMockTransport()
    config = Config(
        model_api_key="mock-submission-key",
        mapillary_token=None,
        runs_dir=str(tmp_path),
    )
    app = create_app(config=config, transport=transport, runs_dir=str(tmp_path))
    client = TestClient(app)

    # 2. Upload fresh image through actual application API
    image_bytes = make_fresh_synthetic_image(1200, 900)
    upload_res = client.post(
        "/inspections",
        files={"file": ("facade_fresh_test.png", image_bytes, "image/png")},
    )
    assert upload_res.status_code == 201, f"Upload failed: {upload_res.text}"
    upload_data = upload_res.json()
    inspection_id = upload_data["inspection_id"]
    assert inspection_id
    assert upload_data["width"] == 1200
    assert upload_data["height"] == 900
    assert upload_data["mime"] == "image/png"
    assert upload_data["status"] == "uploaded"

    # Verify physical storage isolation
    stored_dir = tmp_path / "inspections" / inspection_id
    assert stored_dir.is_dir()
    assert (stored_dir / "original.png").exists()
    assert (stored_dir / "inspection.json").exists()

    # 3. Create and execute inspection with adaptive multi-view strategy
    run_payload = {
        "strategy": "multiview",
        "phrases": ["crack"],
        "confirm_live": True,
    }
    run_res = client.post(f"/inspections/{inspection_id}/run", json=run_payload)
    assert run_res.status_code == 200, f"Run failed: {run_res.text}"
    run_data = run_res.json()

    # Verify execution outcome and strategy persistence
    assert run_data["status"] == "complete"
    assert "run" in run_data
    run_info = run_data["run"]
    assert run_info["strategy"] == "multiview"
    assert run_info["model"] == "sam-3.1"
    assert len(transport.calls) > 0, "Mock transport was not called"

    # Verify cell plan has whole view and tile views
    cells = run_info["cells"]
    view_kinds = {c["view_kind"] for c in cells}
    assert "whole" in view_kinds
    assert "tile" in view_kinds

    # 4. Verify report-data endpoint returns structured candidate findings
    report_data_res = client.get(f"/inspections/{inspection_id}/report-data")
    assert report_data_res.status_code == 200
    report_data = report_data_res.json()

    assert report_data["inspection_id"] == inspection_id
    assert report_data["run"]["strategy"] == "multiview"
    findings = report_data["findings"]
    assert len(findings) > 0, "Expected at least one merged finding"

    # Verify finding schema and coordinate bounds
    first_finding = findings[0]
    assert first_finding["prompt"] == "crack"
    assert "crack" in first_finding["display_label"].lower()
    assert "box" in first_finding
    box = first_finding["box"]
    assert 0 <= box["x0"] < box["x1"] <= 1200
    assert 0 <= box["y0"] < box["y1"] <= 900
    assert "crop_url" in first_finding
    assert "assets" in report_data and "overlay" in report_data["assets"]

    # 5. Open report endpoint and verify HTML generation
    report_res = client.get(f"/inspections/{inspection_id}/report")
    assert report_res.status_code == 200
    assert "text/html" in report_res.headers.get("content-type", "")
    html_content = report_res.text
    assert "Neev inspection screening" in html_content
    assert inspection_id in html_content
    assert "crack" in html_content.lower()

    # 6. Verify original image loads via API
    orig_img_res = client.get(f"/inspections/{inspection_id}/image")
    assert orig_img_res.status_code == 200
    assert orig_img_res.content == image_bytes

    # 7. Verify overlay asset loads and has valid image dimensions
    overlay_path = report_data["assets"]["overlay"]
    overlay_res = client.get(overlay_path)
    assert overlay_res.status_code == 200
    overlay_img = Image.open(io.BytesIO(overlay_res.content))
    assert overlay_img.size == (1200, 900)

    # 8. Verify finding crop assets load
    for finding in findings:
        crop_url = finding["crop_url"]
        crop_res = client.get(crop_url)
        assert crop_res.status_code == 200, f"Crop {crop_url} failed to load: {crop_res.status_code}"
        crop_img = Image.open(io.BytesIO(crop_res.content))
        assert crop_img.width > 0 and crop_img.height > 0

    # 9. Verify inspection appears in history (/reports)
    history_res = client.get("/reports")
    assert history_res.status_code == 200
    history = history_res.json()["reports"]
    matching_records = [rec for rec in history if rec["inspection_id"] == inspection_id]
    assert len(matching_records) == 1
    rec = matching_records[0]
    assert rec["image_name"] == "facade_fresh_test.png"
    assert rec["report_exists"] is True
    assert rec["status"] == "complete"
