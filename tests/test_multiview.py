"""Offline tests: neev/multiview (Phase 8, steps 1-4). No network.

Steps 1-2 are pure geometry + reconciliation over synthetic fixtures.
Step 3 uses seeded RunStore caches with exploding/fake transports.
Step 4 replays the frozen benchmark outputs without new inference.
"""
import base64
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from neev.adapters.sam31 import (  # noqa: E402
    build_image_request,
    sanitized_request,
)
from neev.app import runner as runner_mod  # noqa: E402
from neev.app.store import InspectionStore  # noqa: E402
from neev.config import Config  # noqa: E402
from neev.multiview import provenance, raster, reconcile, views  # noqa: E402
from neev.multiview.provenance import (  # noqa: E402
    STRATEGY_MULTIVIEW,
    STRATEGY_SINGLE,
    finding_provenance,
    merge_manifest,
    view_notes,
)
from neev.multiview.views import (  # noqa: E402
    CRACK_FAMILY,
    MAX_REINSPECT_CROPS,
    reinspect_views,
    should_tile,
    tile_views,
    whole_view,
)
from neev.persistence.runs import RunMeta, RunStore, sha256_bytes  # noqa: E402

try:
    from fastapi.testclient import TestClient

    from neev.app.main import create_app

    HAS_WEB = True
except ImportError:
    HAS_WEB = False

needs_web = pytest.mark.skipif(not HAS_WEB, reason="FastAPI extra not installed")


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


def box_line(x0, y0, x1, y1, w, h):
    return (f"<0f>0<|box;x1={x0};y1={y0};x2={x1};y2={y1};w={w};h={h}|>")


def make_png(size=600, color=(200, 30, 30)):
    # Quadrant pattern (never solid): every tile must hash distinctly,
    # or the cache correctly treats identical tiles as one key.
    from PIL import ImageDraw

    img = Image.new("RGB", (size, size), color)
    draw = ImageDraw.Draw(img)
    half = size // 2
    draw.rectangle([0, 0, half, half], fill=(200, 30, 30))
    draw.rectangle([half, 0, size, half], fill=(30, 200, 30))
    draw.rectangle([0, half, half, size], fill=(30, 30, 200))
    draw.rectangle([half, half, size, size], fill=(200, 200, 30))
    draw.line([0, 0, size, size], fill=(255, 255, 255), width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def seed_cell(runs, raw, mime, phrase, output_text):
    trace = sanitized_request(
        build_image_request(
            phrase, base64.b64encode(raw).decode("ascii"), mime),
        sha256_bytes(raw))
    cached = runs.find_cached(trace)
    if cached is not None:
        return cached
    return runs.save(
        trace, {"output_text": output_text},
        RunMeta(purpose="sam31-sweep", models={"segment": "sam-3.1"},
                input_hashes={"image": sha256_bytes(raw)},
                notes={"phrase": phrase, "mime": mime,
                       "detections": 1 if output_text.strip() else 0}))


def make_inspection(tmp_path, raw, size):
    inspections = InspectionStore(tmp_path)
    record = inspections.create(
        raw, extension="png", image_name="big.png",
        sha256=sha256_bytes(raw), mime="image/png", width=size, height=size,
        quality_warnings=[], quality_metrics={})
    return inspections, record["inspection_id"]


def make_config(tmp_path, api_key=None):
    return Config(model_api_key=api_key, mapillary_token=None,
                  runs_dir=str(tmp_path))


def det(phrase, box, view_id, tile_box, run_id="run1", kind="tile"):
    frame_w = tile_box["x1"] - tile_box["x0"]
    frame_h = tile_box["y1"] - tile_box["y0"]
    return {"phrase": phrase, "box": dict(box), "view": view_id,
            "view_kind": kind, "tile_box": dict(tile_box),
            "frame_width": frame_w, "frame_height": frame_h,
            "run_id": run_id}


# --- Step 1: view planner + provenance ----------------------------------


def test_tile_coverage_overlap_ids_and_edges():
    tiles = tile_views(101, 71)
    assert [t.view_id for t in tiles] == ["r0c0", "r0c1", "r1c0", "r1c1"]
    assert all(t.kind == "tile" for t in tiles)
    for t in tiles:
        assert 0 <= t.box["x0"] < t.box["x1"] <= 101
        assert 0 <= t.box["y0"] < t.box["y1"] <= 71
    # Every border pixel is covered by some tile.
    for x, y in [(0, 0), (100, 0), (0, 70), (100, 70), (50, 0), (0, 35)]:
        assert any(t.box["x0"] <= x < t.box["x1"]
                   and t.box["y0"] <= y < t.box["y1"] for t in tiles), (x, y)
    # Last row/col ends exactly at the frame edge.
    assert max(t.box["x1"] for t in tiles) == 101
    assert max(t.box["y1"] for t in tiles) == 71
    # Neighbors overlap (shared interior pixels exist both ways).
    left = next(t for t in tiles if t.view_id == "r0c0")
    right = next(t for t in tiles if t.view_id == "r0c1")
    assert left.box["x1"] > right.box["x0"]
    assert right.box["x0"] > left.box["x0"]
    # Deterministic: same dims always give the same plan.
    assert tile_views(101, 71) == tiles
    try:
        whole_view(0, 10)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-width frame must be rejected")


def test_should_tile_policy():
    assert "crack in wall" in CRACK_FAMILY
    assert should_tile("crack in wall", 800, 600, 0) is True  # recall rescue
    assert should_tile("crack", 2400, 1792, 5) is True  # large image
    assert should_tile("crack", 800, 600, 5) is False  # small + detections
    assert should_tile("crack", 738, 414, 0) is False  # min side 414 < 480
    assert should_tile("crack", 100, 80, 0) is False  # too small to tile
    assert should_tile("window", 2400, 1792, 0) is False  # not crack family


def test_reinspect_views_capped_and_clamped():
    boxes = [{"x0": 0, "y0": 0, "x1": 200, "y1": 200},
             {"x0": 400, "y0": 400, "x1": 600, "y1": 600},
             {"x0": 10, "y0": 10, "x1": 210, "y1": 210}]
    crops = reinspect_views(boxes, 1000, 1000)
    assert len(crops) == MAX_REINSPECT_CROPS == 2
    assert [c.view_id for c in crops] == ["re0", "re1"]
    assert all(c.kind == "reinspect" for c in crops)
    for c in crops:
        assert 0 <= c.box["x0"] < c.box["x1"] <= 1000
        assert 0 <= c.box["y0"] < c.box["y1"] <= 1000
    # Degenerate boxes drop out; tiny ones grow to a context window.
    assert reinspect_views([{"x0": 5, "y0": 5, "x1": 5, "y1": 9}],
                           9999, 9999) == []
    tiny = reinspect_views([{"x0": 4000, "y0": 4000, "x1": 4010, "y1": 4010}],
                           9999, 9999)
    assert len(tiny) == 1
    assert tiny[0].box["x1"] - tiny[0].box["x0"] >= 256
    assert tiny[0].box["x0"] <= 4000 and tiny[0].box["x1"] >= 4010
    # Frames smaller than the minimum context yield no crops.
    assert reinspect_views([{"x0": 0, "y0": 0, "x1": 10, "y1": 10}],
                           100, 100) == []


def test_provenance_records():
    notes = view_notes("r0c1", "tile", {"x0": 1, "y0": 2, "x1": 3, "y1": 4},
                       "abc", "crack")
    assert notes["strategy"] == STRATEGY_MULTIVIEW
    assert notes["view"] == "r0c1" and notes["source_sha256"] == "abc"
    assert STRATEGY_SINGLE == "single"
    prov = finding_provenance("whole", "whole",
                              {"x0": 0, "y0": 0, "x1": 9, "y1": 9},
                              "abc", "crack", "run9")
    assert prov["run_id"] == "run9" and prov["tile_box"]["x1"] == 9
    manifest = merge_manifest("abc", "crack", [{"a": 1}], [{"b": 2}],
                              [{"view_id": "whole"}])
    assert manifest["artifact"] == "multiview-merge"
    assert manifest["counts"] == {"kept": 1, "suppressed": 1}


# --- Step 2: transforms + reconciliation --------------------------------


def test_box_translation_and_clipping():
    tile = {"x0": 1028, "y0": 0, "x1": 2400, "y1": 1024}
    out = reconcile.to_source_box({"x0": 10, "y0": 10, "x1": 50, "y1": 60},
                                  tile, 2400, 1792)
    assert out == {"x0": 1038, "y0": 10, "x1": 1078, "y1": 60}
    assert all(isinstance(v, int) for v in out.values())
    # Past the source edge: clipped, kept.
    clipped = reconcile.to_source_box({"x0": 1300, "y0": 900, "x1": 1400,
                                       "y1": 1100}, tile, 2400, 1792)
    assert clipped == {"x0": 2328, "y0": 900, "x1": 2400, "y1": 1100}
    overflow = reconcile.to_source_box({"x0": 1300, "y0": 1500, "x1": 1400,
                                        "y1": 1900}, tile, 2400, 1792)
    assert overflow == {"x0": 2328, "y0": 1500, "x1": 2400, "y1": 1792}
    # Fully outside the source frame: nothing remains.
    assert reconcile.to_source_box({"x0": 2000, "y0": 2000, "x1": 2100,
                                    "y1": 2200}, tile, 2400, 1792) is None


def test_invalid_frame_dimensions_rejected():
    assert reconcile.validate_frame(10, 10, 10, 10) == []
    assert reconcile.validate_frame(10, 9, 10, 10) != []
    assert reconcile.validate_frame(None, 10, 10, 10) != []
    assert reconcile.validate_frame("10", 10, 10, 10) != []
    kept, suppressed, rejected = reconcile.reconcile(
        [det("crack", {"x0": 1, "y0": 1, "x1": 5, "y1": 5}, "r0c0",
             {"x0": 0, "y0": 0, "x1": 10, "y1": 10})],
        20, 20)
    assert len(kept) == 1 and not rejected
    bad = det("crack", {"x0": 1, "y0": 1, "x1": 5, "y1": 5}, "r0c0",
              {"x0": 0, "y0": 0, "x1": 10, "y1": 10})
    bad["frame_width"] = 999  # drifted echo
    kept, _, rejected = reconcile.reconcile([bad], 20, 20)
    assert kept == [] and len(rejected) == 1
    assert "frame" in rejected[0]["reason"]


def test_mask_offsets_and_edge_clipping():
    # 2x2 checker raster placed through a tile offset into a 10x10 frame.
    out = reconcile.project_mask(bytes([1, 0, 0, 1]), 2, 2,
                                 {"x0": 0, "y0": 0, "x1": 2, "y1": 2},
                                 {"x0": 4, "y0": 4, "x1": 10, "y1": 10}, 10, 10)
    assert len(out) == 100
    assert out[4 * 10 + 4] == 1 and out[4 * 10 + 5] == 0
    assert out[5 * 10 + 4] == 0 and out[5 * 10 + 5] == 1
    assert sum(out) == 2  # nothing leaked elsewhere in the frame
    # Box hanging past the frame edge clips instead of failing.
    edge = reconcile.project_mask(bytes([1]), 1, 1,
                                  {"x0": 0, "y0": 0, "x1": 4, "y1": 4},
                                  {"x0": 8, "y0": 8, "x1": 12, "y1": 12}, 10, 10)
    assert sum(edge) == 4


def test_duplicate_suppression_and_deterministic_survivor():
    big_tile = {"x0": 0, "y0": 0, "x1": 100, "y1": 100}
    small_tile = {"x0": 10, "y0": 0, "x1": 110, "y1": 100}
    big = det("crack", {"x0": 10, "y0": 10, "x1": 60, "y1": 60}, "r0c0",
              big_tile)
    small = det("crack", {"x0": 10, "y0": 10, "x1": 50, "y1": 55}, "r0c1",
                small_tile)  # source 20,10-60,55: IoU 0.72 with big
    kept, suppressed, rejected = reconcile.reconcile([small, big], 200, 200)
    assert not rejected
    assert len(kept) == 1 and len(suppressed) == 1
    assert kept[0]["source_box"] == {"x0": 10, "y0": 10, "x1": 60, "y1": 60}
    assert suppressed[0]["reason"] == "duplicate-iou"
    assert suppressed[0]["survivor"]["box"] == kept[0]["source_box"]
    assert suppressed[0]["iou"] >= 0.5
    # Order-independent survivor.
    kept2, _, _ = reconcile.reconcile([big, small], 200, 200)
    assert kept2[0]["source_box"] == kept[0]["source_box"]
    # Exact duplicates collapse with their own reason.
    twin = det("crack", {"x0": 10, "y0": 10, "x1": 60, "y1": 60}, "whole",
               {"x0": 0, "y0": 0, "x1": 200, "y1": 200})
    kept3, supp3, _ = reconcile.reconcile([big, twin], 200, 200)
    assert len(kept3) == 1
    assert supp3[0]["reason"] == "exact-duplicate"
    assert supp3[0]["iou"] == 1.0


def test_nearby_distinct_defects_stay_separate():
    frame = {"x0": 0, "y0": 0, "x1": 200, "y1": 200}
    left = det("crack", {"x0": 10, "y0": 10, "x1": 20, "y1": 100}, "whole",
               frame, kind="whole")
    right = det("crack", {"x0": 30, "y0": 10, "x1": 40, "y1": 100}, "whole",
                frame, kind="whole")  # parallel trace, IoU 0
    kept, suppressed, _ = reconcile.reconcile([left, right], 200, 200)
    assert len(kept) == 2 and suppressed == []


def test_different_prompts_never_merge():
    frame = {"x0": 0, "y0": 0, "x1": 200, "y1": 200}
    same = {"x0": 10, "y0": 10, "x1": 60, "y1": 60}
    kept, suppressed, _ = reconcile.reconcile(
        [det("crack", same, "whole", frame, kind="whole"),
         det("crack in wall", same, "whole", frame, kind="whole")], 200, 200)
    assert len(kept) == 2 and suppressed == []


def test_suppression_provenance_complete():
    frame = {"x0": 0, "y0": 0, "x1": 200, "y1": 200}
    dets = [
        det("crack", {"x0": 10, "y0": 10, "x1": 60, "y1": 60}, "whole",
            frame, run_id="runW", kind="whole"),
        det("crack", {"x0": 12, "y0": 12, "x1": 58, "y1": 58}, "r0c0",
            frame, run_id="runT"),
        "not-a-detection",
        det("crack", {"x0": 5, "y0": 5, "x1": 5, "y1": 9}, "whole",
            frame, run_id="runW", kind="whole"),
    ]
    kept, suppressed, rejected = reconcile.reconcile(dets, 200, 200, "sha")
    assert len(kept) == 1 and len(suppressed) == 1 and len(rejected) == 2
    assert kept[0]["provenance"]["source_sha256"] == "sha"
    assert kept[0]["provenance"]["run_id"] == "runW"
    entry = suppressed[0]
    assert entry["detection"]["source_box"] == {"x0": 12, "y0": 12,
                                                "x1": 58, "y1": 58}
    assert entry["survivor"]["mv_id"] == kept[0]["mv_id"]
    assert entry["survivor"]["view"] == "whole"
    assert isinstance(entry["iou"], float)
    assert all("reason" in r for r in rejected)


# --- Step 3: runner strategy ----------------------------------------------


def test_single_strategy_omitted_keeps_exact_record_shape(tmp_path):
    """Mandatory 11: default output semantics frozen (no new keys)."""
    raw = make_png(600)
    runs = RunStore(tmp_path)
    whole_id = seed_cell(runs, raw, "image/png", "crack", "  ")
    inspections, iid = make_inspection(tmp_path, raw, 600)
    record = runner_mod.execute_run(
        inspections, runs, iid, make_config(tmp_path), phrases=["crack"],
        transport=ExplodingTransport())
    assert record["status"] == "complete"
    assert set(record["run"]) == {"created_utc", "confirm_live", "phrases",
                                 "cells", "cache_hits", "live_calls",
                                 "failures", "model"}
    assert "strategy" not in record["run"] and "merged" not in record["run"]
    assert record["run"]["cells"] == [{"phrase": "crack", "status": "cached",
                                       "run_id": whole_id}]
    explicit = runner_mod.execute_run(
        inspections, runs, iid, make_config(tmp_path), phrases=["crack"],
        transport=ExplodingTransport(), strategy="single")
    assert explicit["run"]["cells"] == record["run"]["cells"]
    assert set(explicit["run"]) == set(record["run"])
    try:
        runner_mod.execute_run(
            inspections, runs, iid, make_config(tmp_path),
            phrases=["crack"], strategy="bogus")
    except runner_mod.BadStrategy:
        pass
    else:
        raise AssertionError("unknown strategy must raise BadStrategy")


def test_multiview_cached_only_zero_transport_and_merge(tmp_path):
    """Mandatory 9: cache-first multiview, merge, zero transport calls."""
    raw = make_png(600)
    runs = RunStore(tmp_path)
    whole_id = seed_cell(runs, raw, "image/png", "crack", "  ")
    tiles = {t.view_id: t for t in tile_views(600, 600)}
    tbytes = raster.view_bytes_map(raw, list(tiles.values()))
    seed_cell(runs, tbytes["r0c0"], "image/png", "crack",
              box_line(260, 100, 340, 200, 343, 343))
    seed_cell(runs, tbytes["r0c1"], "image/png", "crack",
              box_line(13, 110, 78, 195, 343, 343))
    inspections, iid = make_inspection(tmp_path, raw, 600)
    transport = ExplodingTransport()
    keys_before = set(runs.read_cache())
    record = runner_mod.execute_run(
        inspections, runs, iid, make_config(tmp_path), phrases=["crack"],
        transport=transport, strategy="multiview")
    assert transport.calls == []
    assert set(runs.read_cache()) == keys_before  # cached-only adds nothing
    run = record["run"]
    assert run["strategy"] == "multiview"
    assert run["projected_cost_usd"] == 0.0
    assert run["view_stats"]["whole"] == {"planned": 1, "cached": 1,
                                          "live": 0, "uncached": 0,
                                          "failed": 0}
    assert run["view_stats"]["tile"]["planned"] == 4
    assert run["view_stats"]["tile"]["cached"] == 2
    assert run["view_stats"]["tile"]["uncached"] == 2
    assert run["view_stats"]["reinspect"]["planned"] == 0
    whole = next(c for c in run["cells"] if c["view_kind"] == "whole")
    assert whole["run_id"] == whole_id  # whole shares the single-view key
    assert record["status"] == "partial"
    merged = run["merged"]["crack"]
    assert merged["counts"]["kept"] == 1
    assert merged["counts"]["suppressed"] == 1
    assert merged["kept"][0]["source_box"] == {"x0": 260, "y0": 100,
                                               "x1": 340, "y1": 200}
    assert merged["kept"][0]["provenance"]["view"] == "r0c0"
    assert merged["suppressed"][0]["reason"] == "duplicate-iou"
    assert merged["suppressed"][0]["survivor"]["box"] == \
        merged["kept"][0]["source_box"]


def test_multiview_tiles_never_hit_whole_cache(tmp_path):
    raw = make_png(600)
    runs = RunStore(tmp_path)
    seed_cell(runs, raw, "image/png", "crack", "  ")
    inspections, iid = make_inspection(tmp_path, raw, 600)
    record = runner_mod.execute_run(
        inspections, runs, iid, make_config(tmp_path), phrases=["crack"],
        transport=ExplodingTransport(), strategy="multiview")
    kinds = {(c["view_kind"], c["status"]) for c in record["run"]["cells"]}
    assert ("whole", "cached") in kinds
    assert ("tile", "cached") not in kinds  # tile bytes differ: no hit
    assert record["run"]["merged"]["crack"]["counts"]["kept"] == 0


def test_multiview_mocked_live_provenance_and_cost(tmp_path):
    raw = make_png(600)
    runs = RunStore(tmp_path)
    inspections, iid = make_inspection(tmp_path, raw, 600)
    fake = FakeTransport([
        {"output_text": "  "},  # whole: empty -> recall rescue tiles
        {"output_text": box_line(260, 100, 340, 200, 343, 343)},
        {"output_text": box_line(13, 110, 78, 195, 343, 343)},
        {"output_text": "  "},
        {"output_text": "  "},
    ])
    keys_before = set(runs.read_cache())
    record = runner_mod.execute_run(
        inspections, runs, iid, make_config(tmp_path, api_key="fake-key"),
        phrases=["crack"], transport=fake, strategy="multiview",
        confirm_live=True)
    assert len(fake.calls) == 5  # whole + 4 tiles, one attempt each
    assert record["run"]["live_calls"] == 5
    assert record["run"]["projected_cost_usd"] == round(5 * 0.0025, 4)
    assert record["status"] == "complete"
    added = set(runs.read_cache()) - keys_before
    assert len(added) == 5  # one key per live view, none for the merge
    tile_cell = next(c for c in record["run"]["cells"]
                     if c["view"] == "r0c0")
    notes = runs.load(tile_cell["run_id"])["meta"]["notes"]
    assert notes["strategy"] == "multiview"
    assert notes["view"] == "r0c0" and notes["view_kind"] == "tile"
    assert notes["tile_box"] == {"x0": 0, "y0": 0, "x1": 343, "y1": 343}
    assert notes["source_sha256"] == sha256_bytes(raw)
    assert notes["phrase"] == "crack"
    merged = record["run"]["merged"]["crack"]
    assert merged["counts"] == {"kept": 1, "suppressed": 1, "rejected": 0}
    assert "fake-key" not in str(record["run"])


def test_multiview_reinspect_planned_capped_and_unexecuted(tmp_path):
    raw = make_png(600)
    runs = RunStore(tmp_path)
    seed_cell(runs, raw, "image/png", "crack", "  ")
    tiles = {t.view_id: t for t in tile_views(600, 600)}
    tbytes = raster.view_bytes_map(raw, list(tiles.values()))
    seed_cell(runs, tbytes["r0c0"], "image/png", "crack",
              box_line(10, 10, 20, 20, 343, 343))  # tiny: 100px on 360k
    inspections, iid = make_inspection(tmp_path, raw, 600)
    transport = ExplodingTransport()
    record = runner_mod.execute_run(
        inspections, runs, iid, make_config(tmp_path), phrases=["crack"],
        transport=transport, strategy="multiview")
    assert transport.calls == []
    reinspect = [c for c in record["run"]["cells"]
                 if c["view_kind"] == "reinspect"]
    assert len(reinspect) == 1  # one tiny kept box -> one crop (cap 2)
    assert reinspect[0]["status"] == "uncached"  # unconfirmed: unexecuted
    crop = reinspect[0]["tile_box"]
    assert crop["x1"] - crop["x0"] >= 256  # minimum context window


@needs_web
def test_multiview_endpoint_plumbing(tmp_path):
    raw = make_png(600)
    app = create_app(config=make_config(tmp_path),
                     transport=ExplodingTransport(),
                     runs_dir=str(tmp_path))
    client = TestClient(app)
    iid = client.post("/inspections", files={"file": ("big.png", raw)}
                      ).json()["inspection_id"]
    res = client.post(f"/inspections/{iid}/run",
                      json={"phrases": ["crack"], "strategy": "multiview"})
    assert res.status_code == 200
    assert res.json()["run"]["strategy"] == "multiview"
    assert "merged" in res.json()["run"]
    bad = client.post(f"/inspections/{iid}/run",
                      json={"phrases": ["crack"], "strategy": "bogus"})
    assert bad.status_code == 422
    default = client.post(f"/inspections/{iid}/run",
                          json={"phrases": ["window"]})
    assert "strategy" not in default.json()["run"]
    assert "merged" not in default.json()["run"]


# --- Step 4: offline evaluation adapter ----------------------------------


def test_override_cost_and_report_helpers(tmp_path):
    from neev.multiview import evaluate as mv_eval

    base = {("sha", "p"): {"boxes": []}}
    merged = {("sha", "p"): {"boxes": [{"x0": 1}]}, ("sha", "q"): {}}
    out = mv_eval.override_predictions(base, merged)
    assert out[("sha", "p")] == {"boxes": [{"x0": 1}]}
    assert base == {("sha", "p"): {"boxes": []}}  # never mutates
    assert mv_eval.estimate_cost(9) == 0.0225
    report = mv_eval.cost_report(2, 9)
    assert report["requests_if_confirmed"] == 9
    assert report["projected_usd"] == 0.0225
    target = tmp_path / "rep.json"
    mv_eval.write_report(target, {"a": 1})
    assert target.is_file()
    try:
        mv_eval.write_report(target, {"a": 2})
    except FileExistsError:
        pass
    else:
        raise AssertionError("write_report must refuse overwrite")


def test_merge_views_from_synthetic_cells():
    from neev.multiview import evaluate as mv_eval

    def cell(run_id, box, frame):
        return {"run_id": run_id,
                "detections": [{"box": dict(box), "frame_width": frame[0],
                                "frame_height": frame[1], "source": run_id,
                                "mask_payload": None, "mask_width": None,
                                "mask_height": None,
                                "mask_encoding": None}]}
    view_cells = [
        {"view_id": "r0c0", "view_kind": "tile",
         "tile_box": {"x0": 0, "y0": 0, "x1": 100, "y1": 100},
         "cell": cell("runA", {"x0": 10, "y0": 10, "x1": 60, "y1": 60},
                      (100, 100))},
        {"view_id": "r0c1", "view_kind": "tile",
         "tile_box": {"x0": 10, "y0": 0, "x1": 110, "y1": 100},
         "cell": cell("runB", {"x0": 10, "y0": 10, "x1": 50, "y1": 55},
                      (100, 100))},
    ]
    merged = mv_eval.merge_views("src", "crack", view_cells, 200, 200)
    assert merged["run_id"] is None  # derived artifact, not a response
    assert merged["boxes"] == [{"x0": 10, "y0": 10, "x1": 60, "y1": 60}]
    assert merged["merged_from"] == ["runA", "runB"]
    assert len(merged["suppressed"]) == 1 and merged["rejected"] == []
    assert [v["view_id"] for v in merged["views"]] == ["r0c0", "r0c1"]


def _repo_root():
    return Path(__file__).resolve().parent.parent


def test_pilot_tile_merge_matches_frozen_counts():
    """Adapter over real pilot cache reproduces frozen merge counts."""
    import json

    from neev.eval.compare import collect_predictions
    from neev.multiview import evaluate as mv_eval

    root = _repo_root()
    tiles_doc = json.loads((root / "runs/crack_tiling_tiles.json"
                            ).read_text(encoding="utf-8"))
    frozen = json.loads((root / "runs/crack_tiling_experiment.json"
                         ).read_text(encoding="utf-8"))
    predictions = collect_predictions(root / "runs")
    expected = {"dm1": (18, 14, 4), "k1": (6, 4, 2)}
    for image_id, (n_raw, n_kept, n_supp) in expected.items():
        boxes = mv_eval.tile_boxes_for_image(tiles_doc, image_id)
        dims = {"dm1": (2400, 1792), "k1": (2752, 1536)}[image_id]
        view_cells = []
        for tile_id, box in sorted(boxes.items()):
            sha = mv_eval.sha_of_file(
                root / f"runs/crack_tiles/{image_id}_{tile_id}.png")
            key = (sha, "crack")
            assert key in predictions, f"missing cached tile {image_id}_{tile_id}"
            view_cells.append({"view_id": tile_id, "view_kind": "tile",
                               "tile_box": box, "cell": predictions[key]})
        n_inputs = sum(len(v["cell"]["detections"]) for v in view_cells)
        assert n_inputs == n_raw
        merged = mv_eval.merge_views("src", "crack", view_cells, *dims)
        assert len(merged["boxes"]) == n_kept
        assert len(merged["suppressed"]) == n_supp
        assert merged["rejected"] == []
        assert len(merged["merged_from"]) == 4  # one run per tile
        assert frozen["images"][image_id]["kept"] == n_kept
        assert frozen["images"][image_id]["suppressed"] == n_supp


def test_frozen_baseline_reproduced_exactly():
    """Mandatory 12: frozen LS eval output reproduced exactly.

    Same export, same matcher and threshold as live/evaluate_ls.py,
    replaying exactly the runs that existed when the frozen file was
    written (run-meta timestamps <= frozen created_utc). Later cache
    additions (tiling + crack-variant sweeps) are excluded by the pin,
    so every evaluated subtree must match, not just the metrics.
    """
    import json

    from neev.eval.compare import collect_predictions, run_rotated_evaluation
    from neev.eval.labelstudio import load_export, parse_task

    root = _repo_root()
    frozen = json.loads((root / "runs/ls_eval_20261101_sweep.json"
                         ).read_text(encoding="utf-8"))
    cutoff = frozen["created_utc"]
    run_ids = []
    for child in sorted((root / "runs").iterdir()):
        meta_path = child / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("created_utc", "") <= cutoff:
            run_ids.append(child.name)
    assert len(run_ids) > 100  # pin must keep the frozen sweep runs
    tasks = load_export(root / "eval/annotations_label_studio.json")
    gt_images = []
    for task in tasks:
        image, errors = parse_task(task, str(root / "dataset"))
        assert not errors
        if image is not None:
            gt_images.append(image)
    assert len(gt_images) == 29
    predictions = collect_predictions(root / "runs", run_ids=run_ids)
    results = run_rotated_evaluation(gt_images, predictions,
                                     iou_threshold=0.5)
    for key in ("results", "negative_fp", "unevaluated_phrases", "warnings"):
        assert results[key] == frozen[key], key
    crack = results["results"]["crack"]["crack in wall"]
    assert (crack["tp"], crack["fp"], crack["fn"]) == (3, 15, 162)
    assert crack["f1"] == frozen["results"]["crack"]["crack in wall"]["f1"]


def test_multiview_report_data_wireup(tmp_path):
    """Test that build_report_data wires up multiview merged detections and physics."""
    from neev.app import reports as reports_mod
    from neev.app.store import InspectionStore
    from PIL import Image
    import io

    from neev.persistence.runs import sha256_bytes

    inspections = InspectionStore(tmp_path)
    img = Image.new("RGB", (800, 800), "white")
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    raw = bio.getvalue()
    sha = sha256_bytes(raw)

    rec = inspections.create(
        raw,
        extension=".png",
        image_name="facade.png",
        sha256=sha,
        mime="image/png",
        width=800,
        height=800,
        quality_warnings=[],
        quality_metrics={},
    )
    iid = rec["inspection_id"]

    # Seed record with multiview merged manifest
    rec["run"] = {
        "strategy": "multiview",
        "phrases": ["crack"],
        "cells": [{"phrase": "crack", "view": "r0c1", "status": "cached", "run_id": "r1"}],
        "merged": {
            "crack": {
                "kept": [
                    {
                        "phrase": "crack",
                        "view": "r0c1",
                        "view_kind": "tile",
                        "source_box": {"x0": 200, "y0": 200, "x1": 400, "y1": 400},
                        "run_id": "r1",
                        "provenance": {"view": "r0c1", "suppressed_count": 1},
                    }
                ],
                "suppressed": [],
            }
        },
    }
    inspections.update(iid, rec)

    data = reports_mod.build_report_data(inspections, iid)
    assert data["finding_count"] == 1
    f = data["findings"][0]
    assert f["box"] == {"x0": 200, "y0": 200, "x1": 400, "y1": 400}
    assert f["view"] == "r0c1"
    assert f["provenance"]["view"] == "r0c1"
    assert f["physics"] is not None
    assert f["physics"]["orientation_type"] == "diagonal_shear"
    assert f["physics"]["seismic_risk_factor"] == 1.5
    assert "Seismic In-Plane Shear" in f["physics"]["failure_mechanism"]

    # Verify that ensure_assets and ensure_report_html use the same prepared detections
    from urllib.parse import quote

    rendered = reports_mod.ensure_assets(inspections, iid)
    assert rendered["overlay"] == f"{iid}_overlay.png"
    assert (inspections.dir_for(iid) / "assets" / rendered["overlay"]).is_file()
    assert "crack#0" in rendered["crops"]
    crop_path = inspections.dir_for(iid) / "assets" / rendered["crops"]["crack#0"]
    assert crop_path.is_file()
    assert crop_path.stat().st_size > 0

    page = reports_mod.ensure_report_html(inspections, iid)
    assert "crack#0" in page
    assert "No findings" not in page
    assert quote(rendered["crops"]["crack#0"], safe="") in page


def test_multiview_report_pipeline_with_differing_tile_hashes(tmp_path):
    """Regression test: tile hashes differ from source image hash but reports stay consistent."""
    from urllib.parse import quote
    from neev.app import reports as reports_mod

    # Source image (600x600)
    raw_source = make_png(600)
    source_sha = sha256_bytes(raw_source)
    inspections, iid = make_inspection(tmp_path, raw_source, 600)

    # Tile image (300x300, distinct bytes and distinct hash)
    raw_tile = make_png(300, color=(100, 150, 200))
    tile_sha = sha256_bytes(raw_tile)
    assert tile_sha != source_sha

    runs = RunStore(tmp_path / "runs")
    tile_run_id = runs.save(
        {"model": "sam-3.1", "phrase": "crack"},
        {"output_text": "<0f>0<|box;x1=10;y1=20;x2=80;y2=90;w=300;h=300|>"},
        RunMeta(
            purpose="test-tile",
            models={"segment": "sam-3.1"},
            input_hashes={"image": tile_sha},
            notes={"phrase": "crack", "view": "r0c0"},
        ),
    )

    rec = inspections.get(iid)
    rec["run"] = {
        "strategy": "multiview",
        "phrases": ["crack"],
        "cells": [
            {
                "phrase": "crack",
                "view": "r0c0",
                "status": "cached",
                "run_id": tile_run_id,
            }
        ],
        "merged": {
            "crack": {
                "kept": [
                    {
                        "phrase": "crack",
                        "view": "r0c0",
                        "view_kind": "tile",
                        "source_box": {"x0": 50, "y0": 60, "x1": 150, "y1": 160},
                        "run_id": tile_run_id,
                        "provenance": {"view": "r0c0", "tile_box": {"x0": 0, "y0": 0, "x1": 300, "y1": 300}},
                    }
                ],
                "suppressed": [],
            }
        },
    }
    inspections.update(iid, rec)

    # 1. build_report_data
    data = reports_mod.build_report_data(inspections, iid)
    assert data["finding_count"] == 1
    f = data["findings"][0]
    assert f["finding_id"] == "crack#0"
    assert f["box"] == {"x0": 50, "y0": 60, "x1": 150, "y1": 160}
    assert f["run_id"] == tile_run_id

    # 2. ensure_assets
    assets = reports_mod.ensure_assets(inspections, iid)
    assert "crack#0" in assets["crops"]
    crop_path = inspections.dir_for(iid) / "assets" / assets["crops"]["crack#0"]
    assert crop_path.is_file()
    assert crop_path.stat().st_size > 0

    # 3. ensure_report_html
    html_page = reports_mod.ensure_report_html(inspections, iid)
    assert "crack#0" in html_page
    assert "No findings" not in html_page
    assert quote(assets["crops"]["crack#0"], safe="") in html_page


def test_multiview_reports_empty_and_pruned_runs(tmp_path):
    """Edge cases: empty multiview predictions and pruned run directories."""
    from neev.app import reports as reports_mod

    raw = make_png(600)
    inspections, iid = make_inspection(tmp_path, raw, 600)

    # Case A: Empty multiview kept detections
    rec = inspections.get(iid)
    rec["run"] = {
        "strategy": "multiview",
        "phrases": ["crack"],
        "cells": [
            {
                "phrase": "crack",
                "view": "whole",
                "status": "cached",
                "run_id": "nonexistent_run_id",
            }
        ],
        "merged": {
            "crack": {
                "kept": [],
                "suppressed": [],
            }
        },
    }
    inspections.update(iid, rec)

    # Degrades cleanly without crash
    data = reports_mod.build_report_data(inspections, iid)
    assert data["finding_count"] == 0

    assets = reports_mod.ensure_assets(inspections, iid)
    assert len(assets["crops"]) == 0
    assert (inspections.dir_for(iid) / "assets" / assets["overlay"]).is_file()

    html_page = reports_mod.ensure_report_html(inspections, iid)
    assert "No findings" in html_page


def test_adaptive_tiling_candidate_manifest_and_determinism(tmp_path):
    """Regression test: verify should_tile policy, determinism, and single-view stability."""
    # 1. should_tile logic on the 4 candidate image dimensions
    assert should_tile("crack in wall", 2400, 1792, whole_detections=0) is True  # dm1
    assert should_tile("crack", 2400, 1792, whole_detections=8) is True         # dm1 (area >= 1M)
    assert should_tile("crack in wall", 1448, 1086, whole_detections=0) is True  # h2
    assert should_tile("crack in wall", 1448, 1086, whole_detections=0) is True  # dv2
    assert should_tile("crack in wall", 1448, 1086, whole_detections=0) is True  # k2

    # Negative controls for should_tile
    assert should_tile("crack", 400, 400, whole_detections=0) is False          # side < 480
    assert should_tile("window", 2400, 1792, whole_detections=0) is False       # not in crack family
    assert should_tile("crack", 600, 600, whole_detections=3) is False          # whole found defects & area < 1M

    # 2. Reconcile determinism
    detections = [
        {
            "phrase": "crack",
            "box": {"x0": 100, "y0": 100, "x1": 200, "y1": 200},
            "view": "r0c0",
            "tile_box": {"x0": 0, "y0": 0, "x1": 500, "y1": 500},
            "frame_width": 500,
            "frame_height": 500,
            "run_id": "run_a",
        },
        {
            "phrase": "crack",
            "box": {"x0": 50, "y0": 100, "x1": 150, "y1": 200},
            "view": "r0c1",
            "tile_box": {"x0": 50, "y0": 0, "x1": 550, "y1": 500},
            "frame_width": 500,
            "frame_height": 500,
            "run_id": "run_b",
        },
    ]
    kept1, supp1, rej1 = reconcile.reconcile(detections, 1000, 1000, source_sha="dummy")
    kept2, supp2, rej2 = reconcile.reconcile(detections, 1000, 1000, source_sha="dummy")
    assert kept1 == kept2
    assert supp1 == supp2
    assert rej1 == rej2

    # 3. Verify single-view execution is unchanged
    raw = make_png(600)
    runs = RunStore(tmp_path / "runs")
    seed_cell(runs, raw, "image/png", "crack", "  ")
    inspections, iid = make_inspection(tmp_path, raw, 600)
    rec_single = runner_mod.execute_run(
        inspections, runs, iid, make_config(tmp_path), phrases=["crack"],
        transport=ExplodingTransport(), strategy="single",
    )
    assert rec_single["run"].get("strategy") is None  # Frozen single-view record shape preserved
    assert "merged" not in rec_single["run"]

