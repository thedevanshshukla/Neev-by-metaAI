"""Unit tests for SAM 3.1 visual box prompt construction and skeleton decomposition."""

import numpy as np
import pytest

from neev.adapters.visual_prompting import (
    build_box_prompt_request,
    build_sam_box_prompt,
    clamp_box,
    decompose_skeleton_adaptive,
    decompose_skeleton_n_segments,
    decompose_skeleton_unsplit,
)


def test_clamp_box():
    # Normal box within bounds
    b = clamp_box(10, 20, 100, 150, 640, 480, pad=5)
    assert b == {"x0": 5, "y0": 15, "x1": 105, "y1": 155}

    # Underflow clipping
    b_under = clamp_box(2, 3, 50, 60, 640, 480, pad=10)
    assert b_under["x0"] == 0
    assert b_under["y0"] == 0

    # Overflow clipping
    b_over = clamp_box(600, 450, 638, 478, 640, 480, pad=10)
    assert b_over["x1"] == 640
    assert b_over["y1"] == 480


def test_build_sam_box_prompt_official_grammar():
    boxes = [
        {"x0": 100, "y0": 150, "x1": 300, "y1": 400},
        {"x0": 50, "y0": 60, "x1": 200, "y1": 180},
    ]
    tokens = build_sam_box_prompt(boxes, width=1280, height=720)

    # In official token grammar, right and bottom are half-open, so x2 = right-1, y2 = bottom-1
    expected = "<0f>0<|box;x1=100;y1=150;x2=299;y2=399;w=1280;h=720|>,1<|box;x1=50;y1=60;x2=199;y2=179;w=1280;h=720|>"
    assert tokens == expected


def test_build_box_prompt_request():
    boxes = [{"x0": 10, "y0": 20, "x1": 50, "y1": 80}]
    req = build_box_prompt_request(boxes, "AAAA", width=640, height=480, mime="image/png")
    assert req["model"] == "sam-3.1"
    assert req["metadata"] == {"mask_encoding": "one_bit"}
    assert req["input"][0]["content"][0]["type"] == "input_text"
    assert req["input"][0]["content"][0]["text"].startswith("<0f>0<|box;x1=10;y1=20;x2=49;y2=79;w=640;h=480|>")
    assert req["input"][0]["content"][1]["type"] == "input_image"
    assert req["input"][0]["content"][1]["image_url"] == "data:image/png;base64,AAAA"


def test_skeleton_decomposition_area_reduction():
    # Construct an artificial diagonal crack line
    # Diagonal line from (10, 10) to (110, 110)
    comp_mask = np.zeros((120, 120), dtype=bool)
    for i in range(10, 110):
        comp_mask[i, i] = True
        comp_mask[i, i + 1] = True
    sl = (slice(10, 110), slice(10, 111))

    # 1. Unsplit
    unsplit = decompose_skeleton_unsplit(sl, width=200, height=200, pad_px=4)
    assert len(unsplit) == 1
    unsplit_area = (unsplit[0]["x1"] - unsplit[0]["x0"]) * (unsplit[0]["y1"] - unsplit[0]["y0"])

    # 2. 2-segment decomposition
    two_segs = decompose_skeleton_n_segments(comp_mask, sl, n_segments=2, width=200, height=200, pad_px=4)
    assert len(two_segs) == 2
    two_seg_area = sum((b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) for b in two_segs)

    # 3. 3-segment decomposition
    three_segs = decompose_skeleton_n_segments(comp_mask, sl, n_segments=3, width=200, height=200, pad_px=4)
    assert len(three_segs) == 3
    three_seg_area = sum((b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) for b in three_segs)

    # Area should decrease significantly (at least 1.5x)
    assert two_seg_area < unsplit_area * 0.70
    assert three_seg_area < two_seg_area

    # 4. Adaptive decomposition
    adaptive = decompose_skeleton_adaptive(comp_mask, sl, width=200, height=200, pad_px=4)
    assert len(adaptive) >= 2
    adapt_area = sum((b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) for b in adaptive)
    assert adapt_area < unsplit_area * 0.70
