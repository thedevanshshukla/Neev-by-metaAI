"""Visual prompt builder and skeleton decomposition for SAM 3.1.

Implements the official Meta Model API SAM 3.1 object prompt grammar via
`meta_sam_parser` and provides medial-skeleton box decomposition to reduce
the proposal-to-GT area ratio.
"""

from __future__ import annotations

import math
from typing import Any, Sequence
import numpy as np
from scipy import ndimage

from meta_sam_parser import (
    FramePrompt,
    PromptBox,
    PromptObject,
    build_object_prompt,
)
from neev.measurements.crack_dsp import extract_skeleton_topology


def clamp_box(
    x0: int, y0: int, x1: int, y1: int, width: int, height: int, pad: int = 0
) -> dict[str, int]:
    """Clamp and pad pixel coordinates to valid non-negative image bounds."""
    left = max(0, min(width - 1, int(x0) - pad))
    top = max(0, min(height - 1, int(y0) - pad))
    right = max(left + 1, min(width, int(x1) + pad))
    bottom = max(top + 1, min(height, int(y1) + pad))
    return {"x0": left, "y0": top, "x1": right, "y1": bottom}


def build_sam_box_prompt(
    boxes: Sequence[dict[str, int]],
    width: int,
    height: int,
) -> str:
    """Serialize a list of bounding boxes into the official Meta SAM 3.1 object prompt string.

    Format: <0f>0<|box;x1=...;y1=...;x2=...;y2=...;w=W;h=H|>,1<|box;...|>
    Coordinates in PromptBox are [left, right) and [top, bottom) half-open integers.
    """
    if not boxes:
        raise ValueError("At least one bounding box is required to build a box prompt.")

    objects: list[PromptObject] = []
    for idx, b in enumerate(boxes):
        c = clamp_box(b["x0"], b["y0"], b["x1"], b["y1"], width, height)
        # Ensure box is valid
        if c["x1"] <= c["x0"]:
            c["x1"] = min(width, c["x0"] + 1)
        if c["y1"] <= c["y0"]:
            c["y1"] = min(height, c["y0"] + 1)

        p_box = PromptBox(
            left=c["x0"],
            top=c["y0"],
            right=c["x1"],
            bottom=c["y1"],
        )
        objects.append(
            PromptObject(
                id=idx,
                prompts=[FramePrompt(frame=0, box=p_box)],
            )
        )

    return build_object_prompt(size=(width, height), objects=objects)


def build_box_prompt_request(
    boxes: Sequence[dict[str, int]],
    image_b64: str,
    width: int,
    height: int,
    mime: str = "image/jpeg",
    stream: bool = False,
) -> dict[str, Any]:
    """Construct an OpenAI-compatible POST /responses request payload with spatial box prompts."""
    prompt_tokens = build_sam_box_prompt(boxes, width, height)
    return {
        "model": "sam-3.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_tokens},
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,{image_b64}",
                    },
                ],
            }
        ],
        "stream": stream,
        "metadata": {"mask_encoding": "one_bit"},
    }


def extract_ordered_skeleton_points(comp_mask: np.ndarray, sl: tuple[slice, slice]) -> list[tuple[int, int]]:
    """Extract ordered 1-pixel skeleton points in absolute image coordinates."""
    dist = ndimage.distance_transform_edt(comp_mask)
    kernel_8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    skel = (dist > 0.5) & (dist >= ndimage.maximum_filter(dist, footprint=kernel_8) - 0.25)
    if skel.sum() == 0:
        skel = comp_mask > 0

    ys, xs = np.where(skel)
    if len(xs) == 0:
        return []

    # Shift to absolute image space
    abs_xs = xs + sl[1].start
    abs_ys = ys + sl[0].start

    # Principal Axis ordering: Project points onto first principal component
    coords = np.column_stack([abs_xs, abs_ys])
    if len(coords) > 2:
        mean = coords.mean(axis=0)
        cov = np.cov(coords, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_axis = eigenvectors[:, -1]
        projections = (coords - mean) @ principal_axis
        sort_indices = np.argsort(projections)
        ordered_coords = coords[sort_indices]
    else:
        ordered_coords = coords

    return [(int(pt[0]), int(pt[1])) for pt in ordered_coords]


def decompose_skeleton_unsplit(
    sl: tuple[slice, slice],
    width: int,
    height: int,
    pad_px: int = 8,
) -> list[dict[str, int]]:
    """Unsplit: Single enclosing bounding box with padding."""
    b = clamp_box(sl[1].start, sl[0].start, sl[1].stop, sl[0].stop, width, height, pad=pad_px)
    return [b]


def decompose_skeleton_n_segments(
    comp_mask: np.ndarray,
    sl: tuple[slice, slice],
    n_segments: int,
    width: int,
    height: int,
    pad_px: int = 8,
) -> list[dict[str, int]]:
    """Uniform N-segment decomposition along the medial skeleton."""
    pts = extract_ordered_skeleton_points(comp_mask, sl)
    if len(pts) < n_segments * 5:
        # Too few skeleton pixels to split meaningfully
        return decompose_skeleton_unsplit(sl, width, height, pad_px=pad_px)

    total_pts = len(pts)
    chunk = total_pts // n_segments
    boxes: list[dict[str, int]] = []

    for i in range(n_segments):
        start_idx = i * chunk
        end_idx = (i + 1) * chunk if i < n_segments - 1 else total_pts
        seg_pts = pts[start_idx:end_idx]
        if not seg_pts:
            continue
        xs = [p[0] for p in seg_pts]
        ys = [p[1] for p in seg_pts]
        boxes.append(clamp_box(min(xs), min(ys), max(xs), max(ys), width, height, pad=pad_px))

    return boxes if boxes else decompose_skeleton_unsplit(sl, width, height, pad_px=pad_px)


def decompose_skeleton_adaptive(
    comp_mask: np.ndarray,
    sl: tuple[slice, slice],
    width: int,
    height: int,
    max_segment_length_px: int = 90,
    pad_px: int = 8,
) -> list[dict[str, int]]:
    """Adaptive decomposition: Splits along length and branch junctions while preserving short cracks."""
    skel_metrics = extract_skeleton_topology(comp_mask)
    length = skel_metrics.length_px

    if length < 50.0:
        # Short fracture: keep unsplit
        return decompose_skeleton_unsplit(sl, width, height, pad_px=pad_px)

    pts = extract_ordered_skeleton_points(comp_mask, sl)
    if len(pts) < 20:
        return decompose_skeleton_unsplit(sl, width, height, pad_px=pad_px)

    # Calculate optimal number of segments based on skeleton length
    n_segments = max(2, min(5, int(math.ceil(length / max_segment_length_px))))
    return decompose_skeleton_n_segments(comp_mask, sl, n_segments, width, height, pad_px=pad_px)
