"""Facade Architectural Context Adapter using Pretrained SegFormer.

Model: Marco333/segformer-b0-facade-cmp (Hugging Face)
Backbone: MiT-B0 (~3.7M parameters, ~14 MB)
Trained on: CMP Facade benchmark
Classes:
  0: background
  1: facade_wall
  2: window
  3: door
  4: balcony
  5: vegetation_occluder

Provides Level-1 Macro Facade Context to evaluate candidate defects against
architectural boundaries (windows, balconies, ledges, doors).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


@dataclass(frozen=True)
class FacadeContextResult:
    """Facade semantic segmentation maps and architectural boundary masks."""
    image_width: int
    image_height: int
    wall_mask: np.ndarray      # 2D boolean mask of facade wall
    window_mask: np.ndarray    # 2D boolean mask of windows
    balcony_mask: np.ndarray   # 2D boolean mask of balconies
    door_mask: np.ndarray      # 2D boolean mask of doors
    class_map: np.ndarray      # 2D uint8 class index map


class FacadeContextAdapter:
    """Inference adapter for pretrained SegFormer facade semantic segmentation."""

    MODEL_NAME = "Marco333/segformer-b0-facade-cmp"

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._processor = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._processor = AutoImageProcessor.from_pretrained(self.MODEL_NAME)
            self._model = SegformerForSemanticSegmentation.from_pretrained(self.MODEL_NAME)
            self._model.to(self.device)
            self._model.eval()

    def segment_facade(self, image: Image.Image | np.ndarray) -> FacadeContextResult:
        """Runs facade parsing and outputs full-resolution architectural masks."""
        self._ensure_loaded()

        if isinstance(image, np.ndarray):
            im_pil = Image.fromarray(image).convert("RGB")
        else:
            im_pil = image.convert("RGB")

        w, h = im_pil.size
        inputs = self._processor(images=im_pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits  # shape [1, 6, H_feat, W_feat]

            # Upsample logits to original image resolution
            upsampled = torch.nn.functional.interpolate(
                logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            pred_classes = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        # Classes: 1=facade_wall, 2=window, 3=door, 4=balcony, 5=vegetation
        wall_mask = (pred_classes == 1)
        window_mask = (pred_classes == 2)
        door_mask = (pred_classes == 3)
        balcony_mask = (pred_classes == 4)

        return FacadeContextResult(
            image_width=w,
            image_height=h,
            wall_mask=wall_mask,
            window_mask=window_mask,
            balcony_mask=balcony_mask,
            door_mask=door_mask,
            class_map=pred_classes,
        )

    def evaluate_box_context(
        self,
        box: dict[str, int],
        facade_res: FacadeContextResult,
    ) -> dict[str, float]:
        """Calculates architectural spatial overlap for a candidate bounding box.

        Returns:
        - wall_ratio: fraction of box area located on genuine wall surface.
        - window_ratio: fraction of box area located on window opening/frame.
        - balcony_ratio: fraction of box area located on balcony element.
        - is_architectural_boundary: whether box aligns with window/balcony perimeter.
        """
        x0 = max(0, min(facade_res.image_width, box["x0"]))
        y0 = max(0, min(facade_res.image_height, box["y0"]))
        x1 = max(0, min(facade_res.image_width, box["x1"]))
        y1 = max(0, min(facade_res.image_height, box["y1"]))

        bw = x1 - x0
        bh = y1 - y0
        area = bw * bh
        if area <= 0:
            return {"wall_ratio": 1.0, "window_ratio": 0.0, "balcony_ratio": 0.0, "is_boundary": 0.0}

        sub_wall = facade_res.wall_mask[y0:y1, x0:x1]
        sub_win = facade_res.window_mask[y0:y1, x0:x1]
        sub_bal = facade_res.balcony_mask[y0:y1, x0:x1]

        wall_ratio = float(np.sum(sub_wall)) / area
        win_ratio = float(np.sum(sub_win)) / area
        bal_ratio = float(np.sum(sub_bal)) / area

        # Check edge perimeter transition
        is_boundary = 1.0 if (win_ratio > 0.15 and wall_ratio > 0.15) or (bal_ratio > 0.15 and wall_ratio > 0.15) else 0.0

        return {
            "wall_ratio": round(wall_ratio, 3),
            "window_ratio": round(win_ratio, 3),
            "balcony_ratio": round(bal_ratio, 3),
            "is_boundary": is_boundary,
        }
