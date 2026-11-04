"""Slicing Aided Hyper Inference (SAHI) for High-Resolution Facade Inspection.

Addresses the resolution downsampling loss where 1-2px hairline micro-cracks
in 4K/12MP drone imagery vanish when resized to standard model input resolutions.

Slices high-resolution facades into overlapping native-scale windows, executes
inference at 1:1 sensor resolution, maps coordinates back to full image space,
and fuses cross-scale envelopes via Cross-Scale Containment Suppression (CCS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional
import numpy as np
from PIL import Image

from neev.eval.tiles import aabb_iou, merge_cross_tile


@dataclass
class SlicedWindow:
    """Represents one localized spatial tile slice."""
    slice_id: str
    x0: int
    y0: int
    x1: int
    y1: int
    image: Image.Image

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


class SAHISlicer:
    """Generates overlapping sliding window crops with configurable overlap."""

    def __init__(
        self,
        slice_size: int = 800,
        overlap_ratio: float = 0.20,
        min_image_dim_for_slicing: int = 1200,
    ):
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.min_image_dim_for_slicing = min_image_dim_for_slicing

    def should_slice(self, image_width: int, image_height: int) -> bool:
        """Determines whether slicing is warranted based on image resolution."""
        return max(image_width, image_height) >= self.min_image_dim_for_slicing

    def generate_slices(self, image: Image.Image | np.ndarray) -> list[SlicedWindow]:
        """Slices the image into overlapping regional windows."""
        if isinstance(image, np.ndarray):
            im_pil = Image.fromarray(image).convert("RGB")
        else:
            im_pil = image.convert("RGB")

        w, h = im_pil.size
        if not self.should_slice(w, h):
            return [SlicedWindow(slice_id="full", x0=0, y0=0, x1=w, y1=h, image=im_pil)]

        step = int(self.slice_size * (1.0 - self.overlap_ratio))
        step = max(50, step)

        slices = []
        y_starts = list(range(0, max(1, h - self.slice_size + 1), step))
        if y_starts[-1] + self.slice_size < h:
            y_starts.append(h - self.slice_size)

        x_starts = list(range(0, max(1, w - self.slice_size + 1), step))
        if x_starts[-1] + self.slice_size < w:
            x_starts.append(w - self.slice_size)

        for row_idx, y0 in enumerate(y_starts):
            y1 = min(h, y0 + self.slice_size)
            for col_idx, x0 in enumerate(x_starts):
                x1 = min(w, x0 + self.slice_size)
                crop = im_pil.crop((x0, y0, x1, y1))
                slices.append(
                    SlicedWindow(
                        slice_id=f"r{row_idx}c{col_idx}",
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        image=crop,
                    )
                )

        return slices

    @staticmethod
    def map_box_to_global(box: dict[str, int], slice_window: SlicedWindow, max_w: int, max_h: int) -> dict[str, int]:
        """Translates a bounding box from local slice coordinates to full-canvas coordinates."""
        return {
            "x0": max(0, min(max_w, box["x0"] + slice_window.x0)),
            "y0": max(0, min(max_h, box["y0"] + slice_window.y0)),
            "x1": max(0, min(max_w, box["x1"] + slice_window.x0)),
            "y1": max(0, min(max_h, box["y1"] + slice_window.y0)),
        }

    @staticmethod
    def box_containment(b1: dict[str, int], b2: dict[str, int]) -> float:
        """Area(Intersection) / min(Area(b1), Area(b2))."""
        ix0 = max(b1["x0"], b2["x0"])
        iy0 = max(b1["y0"], b2["y0"])
        ix1 = min(b1["x1"], b2["x1"])
        iy1 = min(b1["y1"], b2["y1"])
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        a1 = (b1["x1"] - b1["x0"]) * (b1["y1"] - b1["y0"])
        a2 = (b2["x1"] - b2["x0"]) * (b2["y1"] - b2["y0"])
        min_a = min(a1, a2)
        return inter / min_a if min_a > 0 else 0.0

    @classmethod
    def fuse_sahi_with_global(
        cls,
        global_candidates: list[dict[str, Any]],
        tile_candidates: list[dict[str, Any]],
        containment_threshold: float = 0.65,
        nms_iou_threshold: float = 0.40,
    ) -> list[dict[str, Any]]:
        """Fuses global and tile candidate envelopes using Cross-Scale Containment Suppression (CCS).

        Prevents small regional sub-boxes from becoming duplicate false positives
        when already covered by a global macro-envelope.
        """
        # Step 1: Merge redundant detections within tile space
        tile_boxes = [c if "x0" in c else c["box"] for c in tile_candidates]
        merged_tiles, _ = merge_cross_tile(tile_boxes, iou_threshold=nms_iou_threshold) if tile_boxes else ([], [])

        # Re-attach metadata if available
        kept_tile_cands = []
        for tb in merged_tiles:
            # Check containment against global candidates
            is_contained = False
            for gb in global_candidates:
                g_box = gb if "x0" in gb else gb.get("box", gb)
                if cls.box_containment(g_box, tb) >= containment_threshold or cls.box_containment(tb, g_box) >= containment_threshold:
                    is_contained = True
                    break
            if not is_contained:
                kept_tile_cands.append({
                    "x0": tb["x0"], "y0": tb["y0"], "x1": tb["x1"], "y1": tb["y1"],
                    "source": "sahi_tile_slice",
                    "model_confidence": 0.82,
                    "defect_type": "crack",
                })

        # Step 2: Combine with global candidates and perform final NMS
        combined = list(global_candidates) + kept_tile_cands
        combined_boxes = [c if "x0" in c else c["box"] for c in combined]
        final_boxes, _ = merge_cross_tile(combined_boxes, iou_threshold=nms_iou_threshold)

        # Build clean candidate list
        clean_candidates = []
        for fb in final_boxes:
            if (fb["x1"] - fb["x0"]) >= 5 and (fb["y1"] - fb["y0"]) >= 5:
                # Find matching original candidate for metadata provenance
                matched_source = "detector"
                matched_conf = 0.85
                matched_type = "crack"
                for orig in combined:
                    ob = orig if "x0" in orig else orig.get("box", orig)
                    if aabb_iou(fb, ob) >= 0.50:
                        matched_source = orig.get("source", matched_source)
                        matched_conf = orig.get("model_confidence", matched_conf)
                        matched_type = orig.get("defect_type", matched_type)
                        break

                clean_candidates.append({
                    "x0": fb["x0"], "y0": fb["y0"], "x1": fb["x1"], "y1": fb["y1"],
                    "source": matched_source,
                    "model_confidence": matched_conf,
                    "defect_type": matched_type,
                })

        return clean_candidates
