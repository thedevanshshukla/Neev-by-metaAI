"""Inference and Performance Optimization Infrastructure for Neev.

Implements system-level engineering enhancements:
1. GPU memory management and FP16 half-precision execution context.
2. Stage-by-stage latency and throughput profiler.
3. Batch tile tensor preparation for high-throughput GPU inference.
4. Model singleton caching and lazy warm-up.
5. In-memory cache for enhanced multi-view tensors.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Optional
import numpy as np
import torch


class DeviceManager:
    """Manages PyTorch device placement, FP16 execution, and VRAM monitoring."""

    def __init__(self, force_cpu: bool = False):
        if not force_cpu and torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
            torch.backends.cudnn.benchmark = True
        else:
            self.device = "cpu"
            self.dtype = torch.float32

    @contextlib.contextmanager
    def autocast_context(self) -> Generator[None, None, None]:
        """Provides torch.amp.autocast context for FP16 inference on CUDA."""
        if self.device == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                yield
        else:
            yield

    def get_vram_usage_mb(self) -> float:
        """Returns current allocated VRAM in megabytes."""
        if self.device == "cuda":
            return round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        return 0.0

    def get_max_vram_mb(self) -> float:
        """Returns peak allocated VRAM in megabytes."""
        if self.device == "cuda":
            return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        return 0.0

    def reset_peak_memory(self) -> None:
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()


class LatencyProfiler:
    """Stage-by-stage performance and execution latency tracker."""

    def __init__(self):
        self.stage_timings: dict[str, float] = {}
        self._start_times: dict[str, float] = {}

    @contextlib.contextmanager
    def track(self, stage_name: str) -> Generator[None, None, None]:
        """Measures execution duration of a processing block."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self.stage_timings[stage_name] = round(elapsed * 1000.0, 2)  # store in ms

    def to_dict(self) -> dict[str, Any]:
        total_ms = sum(self.stage_timings.values())
        return {
            "stages_ms": self.stage_timings,
            "total_latency_ms": round(total_ms, 2),
            "total_latency_s": round(total_ms / 1000.0, 4),
        }


class TileBatcher:
    """Prepares and batches multiple regional tiles into single forward-pass tensors."""

    @staticmethod
    def prepare_tile_batch(
        tiles: list[np.ndarray],
        target_size: tuple[int, int] = (512, 512),
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Converts a list of HWC uint8 images into a BCHW normalized batch tensor."""
        import cv2
        batch_tensors = []
        for t in tiles:
            resized = cv2.resize(t, target_size, interpolation=cv2.INTER_LINEAR)
            # Normalize to [0, 1]
            tensor = torch.from_numpy(resized).permute(2, 0, 1).to(dtype=dtype) / 255.0
            batch_tensors.append(tensor)

        if not batch_tensors:
            return torch.empty((0, 3, target_size[1], target_size[0]), device=device, dtype=dtype)

        batch = torch.stack(batch_tensors).to(device=device)
        return batch
