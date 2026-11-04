"""Unit tests: neev/app Phase 7 web application. No network, no billing.

Pure helper tests (validation, store, runner prompt rules) run without
the web extra. Endpoint tests use FastAPI's TestClient with an isolated
tmp runs dir and are skipped when FastAPI is unavailable.
"""
import base64
import hashlib
import io
import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from neev.adapters.sam31 import (  # noqa: E402
    build_image_request,
    sanitized_request,
)
from neev.app import reports as reports_mod  # noqa: E402
from neev.app import runner as runner_mod  # noqa: E402
from neev.app.store import InspectionStore  # noqa: E402
from neev.app.validation import (  # noqa: E402
    MAX_UPLOAD_BYTES,
    UploadRejected,
    validate_upload,
)
from neev.config import Config  # noqa: E402
from neev.persistence.runs import RunMeta, RunStore, sha256_bytes  # noqa: E402

try:
    from fastapi.testclient import TestClient

    from neev.app.main import create_app

    HAS_WEB = True
except ImportError:
    HAS_WEB = False

needs_web = pytest.mark.skipif(not HAS_WEB, reason="FastAPI extra not installed")

OFFICIAL_LINE = (
    "<0f>0<|box;x1=10;y1=10;x2=50;y2=60;w=100;h=100|>"
    "<|mask;x=0;y=0;data=2,2,!AB|>"
)
BOX_ONLY_LINE = "<0f>0<|box;x1=10;y1=10;x2=50;y2=60;w=100;h=100|>"


class ExplodingTransport:
    def __init__(self):
        self.calls = []

    def post_json(self, url, headers, payload, timeout=0):
        self.calls.append(payload)
        raise AssertionError("network request attempted")


class FakeTransport:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def post_json(self, url, headers, payload, timeout=0):
        self.calls.append(payload)
        response = self.responses[min(len(self.calls) - 1,
                                      len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


def make_image_bytes(fmt="PNG", size=(120, 100), color=(200, 30, 30)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def seed_cache(runs: RunStore, raw: bytes, mime: str, phrase: str,
               output_text: str, purpose="sam31-sweep") -> str:
    """Save a CLI-shaped trace so the app can hit the shared cache."""
    trace = sanitized_request(
        build_image_request(
            phrase, base64.b64encode(raw).decode("ascii"), mime),
        sha256_bytes(raw))
    cached = runs.find_cached(trace)
    if cached is not None:
        return cached
    return runs.save(
        trace,
        {"output_text": output_text},
        RunMeta(
            purpose=purpose,
            models={"segment": "sam-3.1"},
            input_hashes={"image": sha256_bytes(raw)},
            notes={"phrase": phrase, "mime": mime,
                   "detections": 1 if output_text.strip() else 0},
        ),
    )


def make_client(tmp_path, transport=None, api_key=None):
    config = Config(model_api_key=api_key, mapillary_token=None,
                    runs_dir=str(tmp_path))
    app = create_app(config=config, transport=transport,
                     runs_dir=str(tmp_path))
    return TestClient(app), config


def upload(client, raw: bytes, filename="photo.png"):
    return client.post("/inspections",
                       files={"file": (filename, raw, "application/octet-stream")})


# --- Step 1: validation (no web needed) ----------------------------------


def test_validate_upload_accepts_png_and_jpeg():
    for fmt, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg")):
        raw = make_image_bytes(fmt)
        checked = validate_upload(raw)
        assert checked.mime == mime
        assert checked.width == 120 and checked.height == 100
        assert checked.sha256 == hashlib.sha256(raw).hexdigest()


def test_validate_upload_rejects_text_and_wrong_extension():
    with pytest.raises(UploadRejected) as exc:
        validate_upload(b"not an image at all")
    assert exc.value.status == 415
    # Text renamed to .png: magic bytes, not the extension, decide.
    with pytest.raises(UploadRejected) as exc2:
        validate_upload(b"GIF00 not really")
    assert exc2.value.status == 415


def test_validate_upload_rejects_empty_oversize_and_garbage():
    with pytest.raises(UploadRejected) as exc:
        validate_upload(b"")
    assert exc.value.status == 400
    with pytest.raises(UploadRejected) as exc:
        validate_upload(b"\x00" * (MAX_UPLOAD_BYTES + 1))
    assert exc.value.status == 413
    # JPEG magic but undecodable pixels.
    with pytest.raises(UploadRejected) as exc:
        validate_upload(b"\xff\xd8\xff" + b"\x00" * 100)
    assert exc.value.status == 400


def test_inspection_ids_are_unique_and_isolated(tmp_path):
    store = InspectionStore(tmp_path)
    raw = make_image_bytes()
    first = store.create(raw, extension="png", image_name="a.png",
                         sha256="s", mime="image/png", width=1, height=1,
                         quality_warnings=[], quality_metrics={})
    second = store.create(raw, extension="png", image_name="a.png",
                          sha256="s", mime="image/png", width=1, height=1,
                          quality_warnings=[], quality_metrics={})
    assert first["inspection_id"] != second["inspection_id"]
    assert (store.dir_for(first["inspection_id"]) /
            "original.png").read_bytes() == raw
    with pytest.raises(KeyError):
        store.dir_for("../escape")
    with pytest.raises(KeyError):
        store.dir_for("nope-missing-id")


def test_runner_default_prompts_are_the_docs_sixteen():
    prompts = runner_mod.default_prompts()
    assert len(prompts) == 16
    assert prompts[0] == "building"
    assert "open ground floor" in prompts
    assert "crack in wall" in prompts
    assert runner_mod.normalize_phrases(None) == prompts
    with pytest.raises(runner_mod.BadPhrases):
        runner_mod.normalize_phrases([])
    with pytest.raises(runner_mod.BadPhrases):
        runner_mod.normalize_phrases(["ok", "  "])


# --- Step 1: upload endpoints ---------------------------------------------


@needs_web
def test_health(tmp_path):
    client, _ = make_client(tmp_path)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert "masks_available" in res.json()


@needs_web
def test_upload_returns_metadata_and_hash(tmp_path):
    client, _ = make_client(tmp_path, transport=ExplodingTransport())
    raw = make_image_bytes("PNG")
    res = upload(client, raw, "facade.png")
    assert res.status_code == 201
    body = res.json()
    assert body["image_name"] == "facade.png"
    assert body["sha256"] == hashlib.sha256(raw).hexdigest()
    assert body["width"] == 120 and body["height"] == 100
    assert body["mime"] == "image/png"
    assert body["status"] == "uploaded"
    assert body["run"] is None
    assert isinstance(body["quality_warnings"], list)
    assert body["links"]["report_data"].endswith("/report-data")
    stored = (tmp_path / "inspections" / body["inspection_id"] /
              "original.png")
    assert stored.read_bytes() == raw


@needs_web
def test_upload_rejects_invalid_files(tmp_path):
    client, _ = make_client(tmp_path)
    assert upload(client, b"plain text", "note.txt").status_code == 415
    assert upload(client, b"plain text", "fake.png").status_code == 415
    assert upload(client, b"", "empty.png").status_code == 400
    big = b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_UPLOAD_BYTES
    assert upload(client, big, "big.png").status_code == 413


@needs_web
def test_upload_never_triggers_inference(tmp_path):
    transport = ExplodingTransport()
    client, _ = make_client(tmp_path, transport=transport)
    raw = make_image_bytes("JPEG")
    res = upload(client, raw, "evil.jpg")
    assert res.status_code == 201
    iid = res.json()["inspection_id"]
    assert client.get(f"/inspections/{iid}").status_code == 200
    assert client.get(f"/inspections/{iid}/report-data").status_code == 200
    assert client.get(f"/inspections/{iid}/report").status_code == 200
    assert client.get(f"/inspections/{iid}/image").status_code == 200
    assert client.get("/reports").status_code == 200
    assert transport.calls == []


@needs_web
def test_inspections_do_not_share_dirs(tmp_path):
    client, _ = make_client(tmp_path)
    raw = make_image_bytes()
    first = upload(client, raw, "same.png").json()
    second = upload(client, raw, "same.png").json()
    assert first["inspection_id"] != second["inspection_id"]
    assert first["sha256"] == second["sha256"]


@needs_web
def test_unknown_and_traversal_ids_404(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/inspections/doesnotexist").status_code == 404
    assert client.get("/inspections/bad%20id!").status_code == 404
    assert client.get("/inspections/a/b").status_code == 404
    assert client.post("/inspections/doesnotexist/run",
                       json={}).status_code == 404
    assert client.get("/inspections/doesnotexist/report-data").status_code == 404


# --- Step 2: cached-first execution ---------------------------------------


@needs_web
def test_run_cache_hits_make_zero_transport_calls(tmp_path):
    transport = ExplodingTransport()
    client, _ = make_client(tmp_path, transport=transport)
    raw = make_image_bytes()
    runs = RunStore(tmp_path)
    seed_cache(runs, raw, "image/png", "building", OFFICIAL_LINE)
    seed_cache(runs, raw, "image/png", "window", "  ")
    iid = upload(client, raw).json()["inspection_id"]
    res = client.post(f"/inspections/{iid}/run",
                      json={"phrases": ["building", "window"]})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "complete"
    assert body["run"]["cache_hits"] == 2
    assert body["run"]["live_calls"] == 0
    assert [c["status"] for c in body["run"]["cells"]] == ["cached", "cached"]
    assert transport.calls == []


@needs_web
def test_run_without_confirmation_leaves_cells_uncached(tmp_path):
    transport = ExplodingTransport()
    client, _ = make_client(tmp_path, transport=transport)
    iid = upload(client, make_image_bytes()).json()["inspection_id"]
    res = client.post(f"/inspections/{iid}/run",
                      json={"phrases": ["building", "window"]})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "uploaded"  # nothing executed
    assert [c["status"] for c in body["run"]["cells"]] == ["uncached"] * 2
    assert transport.calls == []


@needs_web
def test_run_confirmed_without_key_is_rejected_before_any_call(tmp_path):
    transport = ExplodingTransport()
    client, _ = make_client(tmp_path, transport=transport, api_key=None)
    iid = upload(client, make_image_bytes()).json()["inspection_id"]
    res = client.post(f"/inspections/{iid}/run",
                      json={"confirm_live": True, "phrases": ["building"]})
    assert res.status_code == 422
    assert "MODEL_API_KEY" in res.json()["detail"]
    assert transport.calls == []
    assert client.get(f"/inspections/{iid}").json()["run"] is None


@needs_web
def test_run_mocked_live_then_cached(tmp_path):
    fake = FakeTransport([{"output_text": OFFICIAL_LINE}])
    client, _ = make_client(tmp_path, transport=fake, api_key="fake-key")
    raw = make_image_bytes()
    iid = upload(client, raw).json()["inspection_id"]
    res = client.post(f"/inspections/{iid}/run",
                      json={"confirm_live": True, "phrases": ["building"]})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "complete"
    assert body["run"]["cells"][0]["status"] == "live"
    assert len(fake.calls) == 1
    # The live payload went through the SAM adapter request flow.
    sent = fake.calls[0]
    assert sent["model"] == "sam-3.1"
    assert sent["input"][0]["content"][0]["text"] == "building"
    live_run_id = body["run"]["cells"][0]["run_id"]
    assert RunStore(tmp_path).load(live_run_id)["meta"]["purpose"] == \
        "app-inspection"
    # Second run is a pure cache hit: zero additional transport calls.
    res2 = client.post(f"/inspections/{iid}/run",
                       json={"phrases": ["building"]})
    assert res2.json()["run"]["cells"][0]["status"] == "cached"
    assert res2.json()["run"]["cells"][0]["run_id"] == live_run_id
    assert len(fake.calls) == 1
    assert "fake-key" not in json.dumps(res2.json())


@needs_web
def test_run_partial_failure_preserves_cached_results(tmp_path):
    from neev.adapters.base import AdapterError

    runs = RunStore(tmp_path)
    raw = make_image_bytes()
    cached_id = seed_cache(runs, raw, "image/png", "building", OFFICIAL_LINE)
    fake = FakeTransport([AdapterError("boom")])
    client, _ = make_client(tmp_path, transport=fake, api_key="fake-key")
    iid = upload(client, raw).json()["inspection_id"]
    res = client.post(
        f"/inspections/{iid}/run",
        json={"confirm_live": True, "phrases": ["building", "window"]})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "partial"
    by_phrase = {c["phrase"]: c for c in body["run"]["cells"]}
    assert by_phrase["building"]["status"] == "cached"
    assert by_phrase["building"]["run_id"] == cached_id
    assert by_phrase["window"]["status"] == "failed"
    assert "AdapterError" in by_phrase["window"]["error"]
    # Evidence still renders from the surviving cached cell.
    data = client.get(f"/inspections/{iid}/report-data").json()
    assert data["finding_count"] == 1
    assert data["failures"][0]["phrase"] == "window"


@needs_web
def test_run_does_not_confuse_images_or_phrases(tmp_path):
    transport = ExplodingTransport()
    client, _ = make_client(tmp_path, transport=transport)
    runs = RunStore(tmp_path)
    raw_a = make_image_bytes(color=(10, 10, 10))
    raw_b = make_image_bytes(color=(200, 200, 200))
    seed_cache(runs, raw_a, "image/png", "building", OFFICIAL_LINE)
    iid = upload(client, raw_b).json()["inspection_id"]
    res = client.post(f"/inspections/{iid}/run",
                      json={"phrases": ["building", "window"]})
    cells = res.json()["run"]["cells"]
    assert [c["status"] for c in cells] == ["uncached"] * 2
    assert transport.calls == []


@needs_web
def test_run_reuses_cli_shaped_traces(tmp_path):
    """A sweep-style saved run is reused verbatim (CLI compatibility)."""
    client, _ = make_client(tmp_path, transport=ExplodingTransport())
    runs = RunStore(tmp_path)
    raw = make_image_bytes()
    cli_run_id = seed_cache(runs, raw, "image/png", "window", BOX_ONLY_LINE,
                            purpose="sam31-sweep")
    iid = upload(client, raw).json()["inspection_id"]
    res = client.post(f"/inspections/{iid}/run",
                      json={"phrases": ["window"]})
    assert res.json()["run"]["cells"][0]["run_id"] == cli_run_id
    data = client.get(f"/inspections/{iid}/report-data").json()
    assert data["findings"][0]["prompt"] == "window"
    assert data["findings"][0]["run_id"] == cli_run_id


@needs_web
def test_run_rejects_bad_phrase_lists(tmp_path):
    client, _ = make_client(tmp_path, transport=ExplodingTransport())
    iid = upload(client, make_image_bytes()).json()["inspection_id"]
    assert client.post(f"/inspections/{iid}/run",
                       json={"phrases": []}).status_code == 422
    assert client.post(
        f"/inspections/{iid}/run",
        json={"phrases": ["x"] * 33}).status_code == 422


# --- Step 3: report-data, assets, frontend --------------------------------


@needs_web
def test_report_data_schema_and_honesty(tmp_path):
    client, _ = make_client(tmp_path, transport=ExplodingTransport())
    runs = RunStore(tmp_path)
    raw = make_image_bytes()
    seed_cache(runs, raw, "image/png", "building", OFFICIAL_LINE)
    iid = upload(client, raw).json()["inspection_id"]
    client.post(f"/inspections/{iid}/run", json={"phrases": ["building"]})
    data = client.get(f"/inspections/{iid}/report-data").json()
    assert data["status"] == "complete"
    assert data["has_run"] is True
    assert data["finding_count"] == 1
    finding = data["findings"][0]
    assert finding["prompt"] == "building"
    assert finding["category"] == "building"
    assert finding["box"] == {"x0": 10, "y0": 10, "x1": 50, "y1": 60}
    assert finding["area_px"] == 40 * 50
    assert finding["confidence"] is None
    assert finding["confidence_label"] == "not provided by model"
    # If no decoder installed in this env: decoder_unavailable; if installed on dummy payload: decode_failed.
    assert finding["mask_status"] in ("decoder_unavailable", "decode_failed")
    assert data["boxes_only"] is True
    assert finding["priority"] in (
        "needs_closer_inspection", "review", "informational")
    assert finding["reasons"] and finding["next_action"] and finding["run_id"]
    assert finding["crop_url"]
    assert "structural safety" in data["disclaimer"]
    assert data["assets"]["overlay"].endswith("_overlay.png")


@needs_web
def test_report_data_distinguishes_not_run_from_zero_findings(tmp_path):
    client, _ = make_client(tmp_path, transport=ExplodingTransport())
    runs = RunStore(tmp_path)
    raw = make_image_bytes()
    iid = upload(client, raw).json()["inspection_id"]
    fresh = client.get(f"/inspections/{iid}/report-data").json()
    assert fresh["has_run"] is False
    assert fresh["findings"] == []
    assert fresh["status"] == "uploaded"
    seed_cache(runs, raw, "image/png", "building", "  ")
    client.post(f"/inspections/{iid}/run", json={"phrases": ["building"]})
    empty = client.get(f"/inspections/{iid}/report-data").json()
    assert empty["has_run"] is True
    assert empty["findings"] == []
    assert empty["status"] == "complete"


@needs_web
def test_report_data_unmapped_prompt_keeps_raw_prompt(tmp_path):
    client, _ = make_client(tmp_path, transport=ExplodingTransport())
    runs = RunStore(tmp_path)
    raw = make_image_bytes()
    seed_cache(runs, raw, "image/png", "crack", BOX_ONLY_LINE)
    iid = upload(client, raw).json()["inspection_id"]
    client.post(f"/inspections/{iid}/run", json={"phrases": ["crack"]})
    finding = client.get(f"/inspections/{iid}/report-data").json()["findings"][0]
    assert finding["prompt"] == "crack"
    assert finding["category"] is None
    assert "presentation label" in finding["display_label"]
    assert finding["mask_status"] == "missing"


def test_presentation_labels_are_documented_and_raw():
    assert reports_mod.PRESENTATION_LABELS["wall crack"]
    assert reports_mod.PRESENTATION_LABELS["damaged wall"]
    assert reports_mod.display_label("crack in wall", "crack") == "crack"
    assert "no evaluator mapping" in reports_mod.display_label("crack", None)
    assert "presentation label" in reports_mod.display_label("wall crack", None)
    assert "unmapped prompt" in reports_mod.display_label("brand new phrase", None)


@needs_web
def test_report_html_and_assets_are_offline_and_aligned(tmp_path):
    transport = ExplodingTransport()
    client, _ = make_client(tmp_path, transport=transport)
    runs = RunStore(tmp_path)
    raw = make_image_bytes()
    seed_cache(runs, raw, "image/png", "building", BOX_ONLY_LINE)
    iid = upload(client, raw).json()["inspection_id"]
    client.post(f"/inspections/{iid}/run", json={"phrases": ["building"]})
    page = client.get(f"/inspections/{iid}/report")
    assert page.status_code == 200
    assert "not a structural safety" in page.text.replace("\n", " ")
    assert (tmp_path / "inspections" / iid / "report.html").is_file()
    data = client.get(f"/inspections/{iid}/report-data").json()
    overlay = client.get(data["assets"]["overlay"])
    assert overlay.status_code == 200
    assert overlay.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(overlay.content)).size == (120, 100)
    crop = client.get(data["findings"][0]["crop_url"])
    assert crop.status_code == 200
    assert max(Image.open(io.BytesIO(crop.content)).size) <= 420
    assert client.get(f"/inspections/{iid}/assets/../inspection.json"
                      ).status_code in (404, 307, 404)
    assert transport.calls == []


@needs_web
def test_frontend_is_served_without_build_step(tmp_path):
    client, _ = make_client(tmp_path)
    page = client.get("/")
    assert page.status_code == 200
    assert "Neev inspection" in page.text
    assert "confirm-live" in client.get("/app.js").text
    assert "body{" in client.get("/styles.css").text


# --- Concurrency: runs vs on-demand rendering -------------------------------
# A browser fires concurrent requests: POST /run invalidates rendered
# assets while GET /assets renders them. Both must serialize on one
# shared lock or the run 500s (Windows refuses to unlink a file being
# written) and readers can catch partial writes.


def _seeded_inspection(tmp_path, phrases=("building", "window")):
    from neev.app import store as store_mod

    inspections = InspectionStore(tmp_path)
    runs = RunStore(tmp_path)
    raw = make_image_bytes()
    for phrase in phrases:
        seed_cache(runs, raw, "image/png", phrase, BOX_ONLY_LINE)
    record = inspections.create(
        raw, extension="png", image_name="p.png",
        sha256=sha256_bytes(raw), mime="image/png", width=120, height=100,
        quality_warnings=[], quality_metrics={})
    config = Config(model_api_key=None, mapillary_token=None,
                    runs_dir=str(tmp_path))
    return store_mod, inspections, runs, record["inspection_id"], config


def test_run_and_render_serialize_on_shared_lock(tmp_path):
    store_mod, inspections, runs, iid, config = _seeded_inspection(tmp_path)
    lock = store_mod.io_lock  # AttributeError until the fix exists
    lock.acquire()
    try:
        worker = threading.Thread(target=runner_mod.execute_run, kwargs={
            "inspections": inspections, "runs": runs,
            "inspection_id": iid, "config": config,
            "phrases": ["building"]})
        worker.start()
        worker.join(timeout=3)
        assert worker.is_alive(), "execute_run ignored the shared io lock"
    finally:
        lock.release()
    worker.join(timeout=60)
    assert not worker.is_alive(), "execute_run deadlocked"
    assert inspections.get(iid)["status"] == "complete"

    lock.acquire()
    try:
        worker = threading.Thread(target=reports_mod.ensure_assets, kwargs={
            "inspections": inspections, "inspection_id": iid})
        worker.start()
        worker.join(timeout=3)
        assert worker.is_alive(), "ensure_assets ignored the shared io lock"
    finally:
        lock.release()
    worker.join(timeout=60)
    assert not worker.is_alive(), "ensure_assets deadlocked"


def test_concurrent_run_render_and_read_are_safe(tmp_path):
    _, inspections, runs, iid, config = _seeded_inspection(tmp_path)
    errors = []

    def run_loop():
        try:
            for _ in range(5):
                runner_mod.execute_run(
                    inspections, runs, iid, config,
                    phrases=["building", "window"])
        except Exception as exc:  # noqa: BLE001 - collected, asserted below
            errors.append(exc)

    def render_loop():
        try:
            for _ in range(10):
                reports_mod.ensure_assets(inspections, iid)
        except Exception as exc:  # noqa: BLE001 - collected, asserted below
            errors.append(exc)

    def read_loop():
        try:
            for _ in range(10):
                reports_mod.build_report_data(inspections, iid)
        except Exception as exc:  # noqa: BLE001 - collected, asserted below
            errors.append(exc)

    workers = [threading.Thread(target=fn)
               for fn in (run_loop, render_loop, read_loop)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=120)
    assert not any(w.is_alive() for w in workers), "worker deadlocked"
    assert errors == []
    assert inspections.get(iid)["status"] == "complete"


# --- Step 4: report index --------------------------------------------------


@needs_web
def test_reports_index_lists_inspections_with_flags(tmp_path):
    client, _ = make_client(tmp_path, transport=ExplodingTransport())
    first = upload(client, make_image_bytes(), "a.png").json()["inspection_id"]
    second = upload(client, make_image_bytes(), "b.png").json()["inspection_id"]
    index = client.get("/reports").json()
    assert index["count"] == 2
    assert index["reports"][0]["inspection_id"] == second  # newest first
    entry = index["reports"][0]
    assert entry["image_name"] == "b.png"
    assert entry["status"] == "uploaded"
    assert entry["created_utc"]
    assert entry["report_exists"] is False
    assert entry["report_url"] == f"/inspections/{second}/report"
    client.get(f"/inspections/{first}/report")
    again = client.get("/reports").json()
    by_id = {r["inspection_id"]: r for r in again["reports"]}
    assert by_id[first]["report_exists"] is True


@needs_web
def test_reset_history_endpoint(tmp_path):
    client, _ = make_client(tmp_path, transport=ExplodingTransport())
    first = upload(client, make_image_bytes(), "a.png").json()["inspection_id"]
    second = upload(client, make_image_bytes(), "b.png").json()["inspection_id"]
    assert client.get("/reports").json()["count"] == 2

    # Reset with keep_showcase=True (default): since neither is a showcase id, both are deleted
    res = client.post("/history/reset", json={"keep_showcase": True})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["deleted_count"] == 2
    assert body["remaining_count"] == 0
    assert client.get("/reports").json()["count"] == 0


# --- Frontend inspection strategy & phrases integration ---------------------


@needs_web
def test_frontend_ui_elements_and_strategy_controls(tmp_path):
    """Verify that the frontend UI serves the strategy and phrase controls."""
    client, _ = make_client(tmp_path)
    html = client.get("/").text
    assert 'id="strategy-multiview"' in html
    assert 'id="strategy-single"' in html
    assert 'id="card-multiview"' in html
    assert 'id="card-single"' in html
    assert 'Adaptive Multi-View' in html
    assert 'Standard Whole-Image' in html
    assert 'id="preset-crack"' in html
    assert 'id="preset-defects"' in html
    assert 'id="preset-all"' in html
    assert 'id="phrase-checkboxes"' in html
    assert 'id="live-warning"' in html
    assert 'not a structural safety' in html

    js = client.get("/app.js").text
    assert "getSelectedStrategy" in js
    assert "getSelectedPhrases" in js
    assert "updateStrategyUI" in js
    assert "updateLiveWarning" in js
    assert "strategy:" in js
    assert "phrases:" in js
    assert "state.running" in js
    assert "badge-view" in js

    css = client.get("/styles.css").text
    assert "body{" in css
    assert ".strategy-card" in css
    assert ".badge-recommended" in css
    assert ".badge-view" in css


@needs_web
def test_run_endpoint_strategy_multiview(tmp_path):
    """POST /run with strategy='multiview' executes multi-view and records strategy."""
    from neev.multiview.views import tile_views
    from neev.multiview import raster

    client, _ = make_client(tmp_path, transport=ExplodingTransport())
    runs = RunStore(tmp_path)
    raw = make_image_bytes(size=(600, 600))
    # Seed whole image and tiles
    seed_cache(runs, raw, "image/png", "crack", "  ")
    tiles = {t.view_id: t for t in tile_views(600, 600)}
    tbytes = raster.view_bytes_map(raw, list(tiles.values()))
    for tview in tiles.values():
        seed_cache(runs, tbytes[tview.view_id], "image/png", "crack", "  ")

    iid = upload(client, raw).json()["inspection_id"]
    res = client.post(
        f"/inspections/{iid}/run",
        json={"strategy": "multiview", "phrases": ["crack"], "confirm_live": False},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["run"]["strategy"] == "multiview"
    assert body["run"]["phrases"] == ["crack"]

    report_data = client.get(f"/inspections/{iid}/report-data").json()
    assert report_data["run"]["strategy"] == "multiview"


@needs_web
def test_run_endpoint_strategy_single_and_invalid(tmp_path):
    """POST /run with strategy='single' or invalid strategies behaves correctly."""
    client, _ = make_client(tmp_path, transport=ExplodingTransport())
    runs = RunStore(tmp_path)
    raw = make_image_bytes(size=(120, 100))
    seed_cache(runs, raw, "image/png", "crack", "  ")

    iid = upload(client, raw).json()["inspection_id"]
    res = client.post(
        f"/inspections/{iid}/run",
        json={"strategy": "single", "phrases": ["crack"], "confirm_live": False},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["run"]["cells"][0]["phrase"] == "crack"

    # Invalid strategy returns 422
    bad_res = client.post(
        f"/inspections/{iid}/run",
        json={"strategy": "unsupported_xyz", "phrases": ["crack"]},
    )
    assert bad_res.status_code == 422
    assert "unknown strategy" in bad_res.json()["detail"]


@needs_web
def test_run_endpoint_phrase_filtering(tmp_path):
    """POST /run screens only the specifically requested phrases."""
    client, _ = make_client(tmp_path, transport=ExplodingTransport())
    runs = RunStore(tmp_path)
    raw = make_image_bytes(size=(120, 100))
    seed_cache(runs, raw, "image/png", "crack", "  ")
    seed_cache(runs, raw, "image/png", "damaged wall", "  ")

    iid = upload(client, raw).json()["inspection_id"]
    res = client.post(
        f"/inspections/{iid}/run",
        json={"phrases": ["crack", "damaged wall"], "strategy": "single"},
    )
    assert res.status_code == 200
    cells = res.json()["run"]["cells"]
    assert len(cells) == 2
    assert {c["phrase"] for c in cells} == {"crack", "damaged wall"}


@needs_web
def test_frontend_hash_persistence_and_error_guards(tmp_path):
    """Verify frontend code contains hash-based deep linking, URL persistence, and upload guards."""
    client, _ = make_client(tmp_path)
    js = client.get("/app.js").text

    assert "initialHash" in js
    assert "hashchange" in js
    assert 'window.location.hash = "#" + id' in js
    assert "upload-btn" in js
    assert 'target = "_blank"' in js
    assert "Upload or select an inspection first" in js


