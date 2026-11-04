"""Specialist Transformer Crack & Spalling Segmenter Adapter.

Model: eoruadl/segformer-b0-finetuned-segments-crop_crack-2 (Hugging Face)
Backbone: MiT-B0 (~3.7M parameters, ~14 MB)
Classes:
  0: unlabeled (background)
  1: crack
  2: potholes (spalling, material loss, concrete cavities)

Provides high-resolution local patch defect segmentation for hairline cracks
and localized concrete surface cavities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


@dataclass(frozen=True)
class DefectInstance:
    box: dict[str, int]
    defect_type: str
    confidence: float
    area_px: int
    mask: np.ndarray


@dataclass(frozen=True)
class SegformerDefectResult:
    image_width: int
    image_height: int
    crack_prob: np.ndarray      # 2D float32 probability map for crack
    spall_prob: np.ndarray      # 2D float32 probability map for spalling/pothole
    instances: list[DefectInstance]


class SegformerCrackAdapter:
    """Inference adapter for SegFormer-B0 crack and spalling segmentation."""

    MODEL_NAME = "eoruadl/segformer-b0-finetuned-segments-crop_crack-2"

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

    def segment(
        self,
        image: Image.Image | np.ndarray,
        threshold: float = 0.35,
        min_area_px: int = 25,
    ) -> SegformerDefectResult:
        """Runs transformer segmentation across the image and extracts defect instances."""
        self._ensure_loaded()

        if isinstance(image, np.ndarray):
            im_pil = Image.fromarray(image).convert("RGB")
        else:
            im_pil = image.convert("RGB")

        w, h = im_pil.size
        inputs = self._processor(images=im_pil, return_tensors="pt").to(self.device)

        # Determine internal processing dimensions to prevent OpenCV buffer exhaustion on 12MP images
        max_dim = max(w, h)
        if max_dim > 1536:
            scale = 1536.0 / max_dim
            proc_w = int(round(w * scale))
            proc_h = int(round(h * scale))
        else:
            scale = 1.0
            proc_w, proc_h = w, h

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits  # shape [1, 3, H_feat, W_feat]

            # Upsample logits to internal processing resolution
            upsampled = torch.nn.functional.interpolate(
                logits,
                size=(proc_h, proc_w),
                mode="bilinear",
                align_corners=False,
            )
            probs = torch.softmax(upsampled, dim=1)[0].cpu().numpy()
            del inputs, outputs, logits, upsampled

        crack_prob = probs[1]  # class 1: crack
        spall_prob = probs[2]  # class 2: potholes/spalling

        instances: list[DefectInstance] = []
        inv_s = 1.0 / scale if scale != 1.0 else 1.0

        # Extract crack instances
        crack_bin = (crack_prob >= threshold).astype(np.uint8)
        num_c, labels_c, stats_c, _ = cv2.connectedComponentsWithStats(crack_bin, connectivity=8)
        for i in range(1, num_c):
            area = int(stats_c[i, cv2.CC_STAT_AREA])
            if area >= min_area_px:
                bx = int(stats_c[i, cv2.CC_STAT_LEFT])
                by = int(stats_c[i, cv2.CC_STAT_TOP])
                bw = int(stats_c[i, cv2.CC_STAT_WIDTH])
                bh = int(stats_c[i, cv2.CC_STAT_HEIGHT])
                comp_m = (labels_c[by:by + bh, bx:bx + bw] == i)
                mean_p = float(np.mean(crack_prob[by:by + bh, bx:bx + bw][comp_m]))

                gx0 = max(0, min(w, int(round(bx * inv_s))))
                gy0 = max(0, min(h, int(round(by * inv_s))))
                gx1 = max(0, min(w, int(round((bx + bw) * inv_s))))
                gy1 = max(0, min(h, int(round((by + bh) * inv_s))))

                instances.append(DefectInstance(
                    box={"x0": gx0, "y0": gy0, "x1": gx1, "y1": gy1},
                    defect_type="crack",
                    confidence=round(mean_p, 3),
                    area_px=int(round(area * inv_s * inv_s)),
                    mask=comp_m,
                ))

        # Extract spalling instances
        spall_bin = (spall_prob >= threshold).astype(np.uint8)
        num_s, labels_s, stats_s, _ = cv2.connectedComponentsWithStats(spall_bin, connectivity=8)
        for j in range(1, num_s):
            area = int(stats_s[j, cv2.CC_STAT_AREA])
            if area >= min_area_px:
                bx = int(stats_s[j, cv2.CC_STAT_LEFT])
                by = int(stats_s[j, cv2.CC_STAT_TOP])
                bw = int(stats_s[j, cv2.CC_STAT_WIDTH])
                bh = int(stats_s[j, cv2.CC_STAT_HEIGHT])
                comp_m = (labels_s[by:by + bh, bx:bx + bw] == j)
                mean_p = float(np.mean(spall_prob[by:by + bh, bx:bx + bw][comp_m]))

                gx0 = max(0, min(w, int(round(bx * inv_s))))
                gy0 = max(0, min(h, int(round(by * inv_s))))
                gx1 = max(0, min(w, int(round((bx + bw) * inv_s))))
                gy1 = max(0, min(h, int(round((by + bh) * inv_s))))

                instances.append(DefectInstance(
                    box={"x0": gx0, "y0": gy0, "x1": gx1, "y1": gy1},
                    defect_type="spalling",
                    confidence=round(mean_p, 3),
                    area_px=int(round(area * inv_s * inv_s)),
                    mask=comp_m,
                ))

        return SegformerDefectResult(
            image_width=w,
            image_height=h,
            crack_prob=crack_prob,
            spall_prob=spall_prob,
            instances=instances,
        )

    def evaluate_crop_probabilities(self, crop: Image.Image | np.ndarray) -> dict[str, float]:
        """Evaluates crack and spalling probability for a localized crop."""
        self._ensure_loaded()
        if isinstance(crop, np.ndarray):
            im_pil = Image.fromarray(crop).convert("RGB")
        else:
            im_pil = crop.convert("RGB")

        inputs = self._processor(images=im_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()

        return {
            "mean_crack_prob": round(float(np.mean(probs[1])), 3),
            "max_crack_prob": round(float(np.max(probs[1])), 3),
            "mean_spall_prob": round(float(np.mean(probs[2])), 3),
            "max_spall_prob": round(float(np.max(probs[2])), 3),
        }
