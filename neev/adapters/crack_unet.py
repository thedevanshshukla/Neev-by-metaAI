"""Crack U-Net adapter for local, zero-cost semantic crack segmentation.

Wraps the pretrained U-Net from third_party/crack-seg (MIT licence,
ishaan1402/crack-seg), using sliding-window inference with Gaussian
blending to produce pixel-level probability maps and segmented instances.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import scipy.ndimage as ndi
import torch
import cv2

# Ensure third_party/crack-seg is on sys.path
CRACK_SEG_ROOT = Path(__file__).resolve().parent.parent.parent / "third_party" / "crack-seg"
if str(CRACK_SEG_ROOT) not in sys.path:
    sys.path.insert(0, str(CRACK_SEG_ROOT))

from src.config.schema import SystemSettings
from src.inference.sliding_window import SlidingWindowPredictor
from src.models.checkpoint import load_unet_checkpoint
from neev.eval.geometry_eval import minimum_bounding_rectangle


DEFAULT_CHECKPOINT = CRACK_SEG_ROOT / "checkpoints" / "unet_v3.pth"
DEFAULT_CONFIG = CRACK_SEG_ROOT / "config" / "config.yaml"


@dataclass
class UNetCrackInstance:
    """One connected crack instance derived from the segmentation mask."""
    instance_id: int
    box: dict[str, int]               # {"x0": ..., "y0": ..., "x1": ..., "y1": ...}
    polygon: list[tuple[float, float]] # 4-corner oriented rectangle or AABB
    area_px: int
    mean_prob: float
    mask: np.ndarray                  # Local binary mask cropped to box


@dataclass
class UNetSegmentationResult:
    """Full-frame segmentation and derived defect candidates."""
    probs: np.ndarray                 # (H, W) float32 probabilities in [0, 1]
    binary: np.ndarray                # (H, W) uint8 binary mask in {0, 1}
    crack_area_ratio: float           # cracked_px / total_px
    latency_s: float                  # Wall-clock inference time in seconds
    tiles: int                        # Number of patches evaluated
    device: str                       # Device used ("cpu" or "cuda")
    instances: list[UNetCrackInstance] = field(default_factory=list)


class CrackUNetAdapter:
    """Adapter for offline local U-Net crack segmentation."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
        device: str | torch.device | None = None,
        min_component_area_px: int = 30,
    ):
        self.ckpt_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        self.cfg_path = Path(config_path or DEFAULT_CONFIG)
        self.min_area = min_component_area_px

        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {self.ckpt_path}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.settings = SystemSettings.load_from_yaml(str(self.cfg_path))
        self.model, self.features = load_unet_checkpoint(str(self.ckpt_path), self.device)
        self.predictor = SlidingWindowPredictor(self.model, self.settings, self.device)

    def segment(
        self,
        image_rgb: np.ndarray,
        threshold: float = 0.5,
        overlap: float = 0.5,
        extract_instances: bool = True,
    ) -> UNetSegmentationResult:
        """Run sliding-window segmentation on an HxWx3 RGB uint8 image."""
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got shape {image_rgb.shape}")

        h_img, w_img, _ = image_rgb.shape
        t0 = time.perf_counter()

        max_dim = max(h_img, w_img)
        if max_dim > 1536:
            scale = 1536.0 / max_dim
            proc_w = int(round(w_img * scale))
            proc_h = int(round(h_img * scale))
            proc_rgb = cv2.resize(image_rgb, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            proc_w, proc_h = w_img, h_img
            proc_rgb = image_rgb

        # Compute number of tiles planned
        stride = int(self.settings.inference.patch_size * (1.0 - overlap))
        if max(proc_h, proc_w) <= self.settings.inference.patch_size:
            n_tiles = 1
        else:
            pad_h = max(0, self.settings.inference.patch_size - proc_h)
            pad_w = max(0, self.settings.inference.patch_size - proc_w)
            eff_h = proc_h + pad_h
            eff_w = proc_w + pad_w
            y_starts = list(range(0, eff_h - self.settings.inference.patch_size + 1, stride)) or [0]
            if y_starts[-1] + self.settings.inference.patch_size < eff_h:
                y_starts.append(eff_h - self.settings.inference.patch_size)
            x_starts = list(range(0, eff_w - self.settings.inference.patch_size + 1, stride)) or [0]
            if x_starts[-1] + self.settings.inference.patch_size < eff_w:
                x_starts.append(eff_w - self.settings.inference.patch_size)
            n_tiles = len(y_starts) * len(x_starts)

        probs, binary, crack_ratio = self.predictor.predict_large_image(
            proc_rgb, threshold=threshold, overlap=overlap
        )
        latency = time.perf_counter() - t0

        instances = []
        if extract_instances and np.any(binary):
            raw_instances = self._extract_instances(probs, binary, proc_h, proc_w)
            if scale != 1.0:
                inv_s = 1.0 / scale
                for inst in raw_instances:
                    bx0 = max(0, min(w_img, int(round(inst.box["x0"] * inv_s))))
                    by0 = max(0, min(h_img, int(round(inst.box["y0"] * inv_s))))
                    bx1 = max(0, min(w_img, int(round(inst.box["x1"] * inv_s))))
                    by1 = max(0, min(h_img, int(round(inst.box["y1"] * inv_s))))
                    scaled_poly = [(float(pt[0] * inv_s), float(pt[1] * inv_s)) for pt in inst.polygon]
                    instances.append(UNetCrackInstance(
                        instance_id=inst.instance_id,
                        box={"x0": bx0, "y0": by0, "x1": bx1, "y1": by1},
                        polygon=scaled_poly,
                        area_px=int(round(inst.area_px * inv_s * inv_s)),
                        mean_prob=inst.mean_prob,
                        mask=inst.mask,
                    ))
            else:
                instances = raw_instances

        return UNetSegmentationResult(
            probs=probs,
            binary=binary,
            crack_area_ratio=crack_ratio,
            latency_s=latency,
            tiles=n_tiles,
            device=self.device.type,
            instances=instances,
        )

    def _extract_instances(
        self,
        probs: np.ndarray,
        binary: np.ndarray,
        height: int,
        width: int,
    ) -> list[UNetCrackInstance]:
        """Convert binary mask into connected crack bounding instances."""
        structure = np.ones((3, 3), dtype=int)
        labels, num_features = ndi.label(binary, structure=structure)
        slices = ndi.find_objects(labels)
        instances = []

        for label in range(1, num_features + 1):
            sl = slices[label - 1]
            if sl is None:
                continue
            y_slice, x_slice = sl
            y0, y1 = y_slice.start, y_slice.stop
            x0, x1 = x_slice.start, x_slice.stop

            comp_mask = (labels[y0:y1, x0:x1] == label).astype(np.uint8)
            area = int(comp_mask.sum())
            if area < self.min_area:
                continue

            mean_prob = float(probs[y0:y1, x0:x1][comp_mask > 0].mean())

            # Active pixels in source frame
            ys, xs = np.where(comp_mask > 0)
            pts = np.column_stack([xs + x0, ys + y0])
            poly = None
            if len(pts) >= 3:
                try:
                    poly = minimum_bounding_rectangle(pts)
                except Exception:
                    poly = None

            if not poly or len(poly) != 4:
                poly = [(float(x0), float(y0)), (float(x1), float(y0)),
                        (float(x1), float(y1)), (float(x0), float(y1))]

            instances.append(
                UNetCrackInstance(
                    instance_id=label,
                    box={"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)},
                    polygon=poly,
                    area_px=area,
                    mean_prob=mean_prob,
                    mask=comp_mask,
                )
            )

        return instances
