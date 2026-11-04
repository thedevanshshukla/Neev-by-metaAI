"""Tests for Cluster-Split-v1 integrity and validation set audit (Experiment 2.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neev.eval.labelstudio import load_export, parse_task
from runs.experiments.experiment_2_3_validation.run_experiment_2_3 import (
    DEV_CLUSTERS,
    DEV_IMAGES,
    VAL_CLUSTERS,
    VAL_IMAGES,
    audit_dataset_split,
)


def test_cluster_split_completeness():
    """Verify that all 29 images in the dataset are assigned exactly once."""
    tasks = load_export("eval/annotations_label_studio.json")
    all_task_ids = sorted([parse_task(t, "dataset")[0].image_id for t in tasks])

    assert len(all_task_ids) == 29
    all_assigned = sorted(DEV_IMAGES + VAL_IMAGES)
    assert all_assigned == all_task_ids
    assert len(set(all_assigned)) == 29


def test_cluster_split_non_overlapping():
    """Verify that Dev and Val sets are strictly disjoint."""
    dev_set = set(DEV_IMAGES)
    val_set = set(VAL_IMAGES)
    overlap = dev_set & val_set
    assert len(overlap) == 0, f"Found overlapping images between dev and val: {overlap}"
    assert len(dev_set) == 10
    assert len(val_set) == 19


def test_scene_cluster_atomicity():
    """Verify that every multi-view capture cluster is kept atomically in either dev or val."""
    for c_name, c_imgs in DEV_CLUSTERS.items():
        for img in c_imgs:
            assert img in DEV_IMAGES, f"Dev cluster {c_name} image {img} not in DEV_IMAGES"
            assert img not in VAL_IMAGES, f"Dev cluster {c_name} image {img} leaked into VAL_IMAGES"

    for c_name, c_imgs in VAL_CLUSTERS.items():
        for img in c_imgs:
            assert img in VAL_IMAGES, f"Val cluster {c_name} image {img} not in VAL_IMAGES"
            assert img not in DEV_IMAGES, f"Val cluster {c_name} image {img} leaked into DEV_IMAGES"


def test_paired_overview_medium_shots_remain_together():
    """Explicitly verify paired same-building scene groups are never split across dev/val."""
    pairs = [
        ("h1", "h2"),    # Crack_Detection_Edge_Cases
        ("k1", "k2"),    # Difficult_Visibility
        ("dv1", "dv2"),  # Difficult_Viewpoints
        ("i1", "i2"),    # Multiple_Defects_in_One_Building
        ("dm1", "dm2"),  # confusing_category
        ("fa1", "fa2"),  # Real_Building_Facades
        ("j1", "j2"),    # Hard_Negative_Examples
        ("st1", "st2"),  # Structural_Components
    ]
    dev_set = set(DEV_IMAGES)
    val_set = set(VAL_IMAGES)

    for img_a, img_b in pairs:
        both_in_dev = (img_a in dev_set and img_b in dev_set)
        both_in_val = (img_a in val_set and img_b in val_set)
        assert both_in_dev or both_in_val, f"Paired scene {img_a}/{img_b} was split across partitions!"


def test_validation_experiment_artifacts_exist():
    """Verify that Experiment 2.3 produced all required persistent artifacts."""
    exp_dir = Path("runs/experiments/experiment_2_3_validation")
    assert (exp_dir / "validation_results.json").exists()
    assert (exp_dir / "per_image_metrics.csv").exists()
    assert (exp_dir / "provenance_manifest.json").exists()
    assert (exp_dir / "visualizations").is_dir()

    with open(exp_dir / "validation_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["metadata"]["matching_mode"] == "threshold_constrained_maximum_cardinality"
    assert data["split_audit"]["all_images_assigned_exactly_once"] is True
    assert data["cache_audit"]["available_count"] == 19
    assert data["cache_audit"]["missing_count"] == 0
    assert data["metadata"]["live_api_calls_made"] == 12
    assert data["metadata"]["actual_spend_usd"] == 0.03
