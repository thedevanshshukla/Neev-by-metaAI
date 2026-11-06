"""Neev Memory Guard & 4 GB Peak RAM Protection Module.

Protects the server from Out-Of-Memory (OOM) crashes by:
1. Enforcing an upper safety ceiling of 4 GB (4096 MB) process RAM.
2. Monitoring available system RAM before and during heavy inference.
3. Forcing aggressive garbage collection and buffer cleanup.
4. Catching memory spikes safely and returning structured fallback responses
   so the server never crashes or goes offline.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("neev.memory_guard")

# Configurable peak RAM ceiling (defaults to 4 GB / 4096 MB)
DEFAULT_MAX_PEAK_RAM_MB = float(os.environ.get("NEEV_MAX_RAM_MB", "4096.0"))
MIN_FREE_SYSTEM_RAM_MB = float(os.environ.get("NEEV_MIN_FREE_RAM_MB", "350.0"))


class InsufficientRAMError(RuntimeError):
    """Raised when available RAM is insufficient to complete inference safely within the 4 GB ceiling."""
    def __init__(self, message: str, current_rss_mb: float = 0.0, limit_mb: float = DEFAULT_MAX_PEAK_RAM_MB):
        super().__init__(message)
        self.message = message
        self.current_rss_mb = current_rss_mb
        self.limit_mb = limit_mb


def get_memory_stats() -> dict[str, float]:
    """Returns current process RSS and system memory metrics in Megabytes."""
    rss_mb = 0.0
    avail_mb = 0.0
    total_mb = 0.0
    if psutil is not None:
        try:
            proc = psutil.Process()
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            vmem = psutil.virtual_memory()
            avail_mb = vmem.available / (1024 * 1024)
            total_mb = vmem.total / (1024 * 1024)
        except Exception as exc:
            logger.debug("Failed to query psutil memory metrics: %s", exc)
    return {
        "process_rss_mb": round(rss_mb, 1),
        "system_avail_mb": round(avail_mb, 1),
        "system_total_mb": round(total_mb, 1),
    }


def force_cleanup() -> None:
    """Aggressively reclaims unreferenced Python, PyTorch, and OS memory buffers."""
    gc.collect()
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def check_ram_headroom(
    estimated_need_mb: float = 250.0,
    peak_limit_mb: float = DEFAULT_MAX_PEAK_RAM_MB,
    min_system_avail_mb: float = MIN_FREE_SYSTEM_RAM_MB,
) -> None:
    """Verifies that current memory usage plus estimated need remains under the 4 GB peak ceiling.
    
    Raises:
        InsufficientRAMError: if executing would violate the 4GB ceiling or starve the OS.
    """
    stats = get_memory_stats()
    rss = stats["process_rss_mb"]
    avail = stats["system_avail_mb"]

    # Check 1: Process RSS plus estimated need against 4 GB peak ceiling
    if rss > 0 and (rss + estimated_need_mb > peak_limit_mb):
        force_cleanup()
        stats_after = get_memory_stats()
        rss_after = stats_after["process_rss_mb"]
        if rss_after + estimated_need_mb > peak_limit_mb:
            raise InsufficientRAMError(
                f"Peak RAM limit reached: Server process uses {rss_after:.1f} MB (safety ceiling: {peak_limit_mb:.0f} MB). "
                f"Inference requires ~{estimated_need_mb:.0f} MB additional buffer. "
                f"Execution safely paused to keep server online.",
                current_rss_mb=rss_after,
                limit_mb=peak_limit_mb,
            )

    # Check 2: Available system memory against critical threshold
    if avail > 0 and avail < min_system_avail_mb:
        force_cleanup()
        stats_after = get_memory_stats()
        avail_after = stats_after["system_avail_mb"]
        if avail_after < min_system_avail_mb:
            raise InsufficientRAMError(
                f"System available RAM critically low: only {avail_after:.1f} MB free "
                f"(safety threshold: {min_system_avail_mb:.0f} MB). "
                f"Execution safely paused to keep server online.",
                current_rss_mb=rss,
                limit_mb=peak_limit_mb,
            )
