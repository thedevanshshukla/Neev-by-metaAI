"""Offline tests for the mask-pilot artifact validator (synthetic)."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "mask_pilot_validate", ROOT / "eval" / "mask_pilot" / "validate.py")
validate_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_mod)

validate_artifact = validate_mod.validate_artifact
ring_self_intersects = validate_mod.ring_self_intersects

IDS = ["cr4", "cr5", "dm1", "k1", "h2", "i1", "in1"]
TASKS = {"cr4": 7, "cr5": 8, "dm1": 2, "k1": 13, "h2": 10, "i1": 21, "in1": 17}
DIMS = {i: (100, 80) for i in IDS}
SQ = [[10, 10], [40, 10], [40, 40], [10, 40]]


def entry(iid, instances="auto"):
    if instances == "auto":
        instances = [] if iid == "in1" else [
            {"instance_id": f"{iid}-c1", "thin": False, "polygons": [SQ]}]
    return {"image_id": iid, "task_id": TASKS[iid], "width": 100, "height": 80,
            "negative": iid == "in1", "instances": instances}


def artifact(entries):
    return {"version": 1, "spec": "x", "images": entries}


def good():
    return artifact([entry(i) for i in IDS])


def test_valid_artifact():
    errors, warnings = validate_artifact(good(), DIMS)
    assert errors == [] and warnings == []


def test_missing_extra_and_dim_errors():
    miss = artifact([entry(i) for i in IDS if i != "k1"])
    errors, _ = validate_artifact(miss, DIMS)
    assert any("k1: missing" in e for e in errors)
    extra = artifact([entry(i) for i in IDS] + [entry("cr4")])
    errors, _ = validate_artifact(extra, DIMS)
    assert any("duplicate image" in e for e in errors)
    wrong = good()
    wrong["images"][0]["width"] = 638
    errors, _ = validate_artifact(wrong, DIMS)
    assert any("dims" in e for e in errors)
    wrong_task = good()
    wrong_task["images"][1]["task_id"] = 99
    errors, _ = validate_artifact(wrong_task, DIMS)
    assert any("task_id" in e for e in errors)


def test_instance_id_rules():
    dup = good()
    dup["images"][0]["instances"].append(
        {"instance_id": "cr4-c1", "thin": False, "polygons": [SQ]})
    errors, _ = validate_artifact(dup, DIMS)
    assert any("duplicate instance_id" in e for e in errors)
    bad = good()
    bad["images"][0]["instances"][0]["instance_id"] = ""
    errors, _ = validate_artifact(bad, DIMS)
    assert any("non-empty string" in e for e in errors)


def test_polygon_geometry_errors():
    oob = good()
    oob["images"][0]["instances"][0]["polygons"] = [[[0, 0], [200, 10], [10, 10]]]
    errors, _ = validate_artifact(oob, DIMS)
    assert any("outside" in e for e in errors)
    short = good()
    short["images"][0]["instances"][0]["polygons"] = [[[0, 0], [1, 1]]]
    errors, _ = validate_artifact(short, DIMS)
    assert any(">= 3 points" in e for e in errors)
    bowtie = good()
    bowtie["images"][0]["instances"][0]["polygons"] = [
        [[0, 0], [30, 30], [0, 30], [30, 0]]]
    errors, _ = validate_artifact(bowtie, DIMS)
    assert any("self-intersecting" in e for e in errors)
    assert ring_self_intersects([[0, 0], [30, 30], [0, 30], [30, 0]]) is True
    assert ring_self_intersects(SQ) is False


def test_thin_and_negative_rules():
    line = [[10, 10], [40, 10], [70, 10]]  # zero area
    thin_ok = good()
    thin_ok["images"][0]["instances"][0]["polygons"] = [line]
    thin_ok["images"][0]["instances"][0]["thin"] = True
    errors, warnings = validate_artifact(thin_ok, DIMS)
    assert errors == [] and len(warnings) == 1 and "zero pixels" in warnings[0]
    thin_bad = good()
    thin_bad["images"][0]["instances"][0]["polygons"] = [line]
    errors, _ = validate_artifact(thin_bad, DIMS)
    assert any("thin:true" in e for e in errors)
    neg = good()
    neg["images"][-1]["instances"] = [
        {"instance_id": "in1-c1", "thin": False, "polygons": [SQ]}]
    errors, _ = validate_artifact(neg, DIMS)
    assert any("zero instances" in e for e in errors)
    empty_pos = good()
    empty_pos["images"][0]["instances"] = []
    errors, warnings = validate_artifact(empty_pos, DIMS)
    assert errors == [] and any("zero instances" in w for w in warnings)
