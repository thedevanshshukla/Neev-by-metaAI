"""Frame sampling plans and sharpness selection (pure functions).

Blur/sharpness scores come from a decoder (ffmpeg/OpenCV) that is not part
of this harness yet. These functions decide *which* frames to keep once
each candidate has a score, so the policy is unit-testable without video.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredFrame:
    frame_id: str
    timestamp_s: float
    sharpness: float  # higher is sharper; e.g. Laplacian variance


def frame_name(index: int) -> str:
    """Docs-style frame id: f_0132.jpg (grows past 4 digits naturally)."""
    return f"f_{index:04d}.jpg"


def sample_plan(
    duration_s: float, fps: float, max_frames: int | None = None
) -> list[float]:
    """Evenly spaced timestamps starting at 0. Empty for nonsense inputs."""
    if duration_s <= 0 or fps <= 0:
        return []
    count = int(duration_s * fps)
    stamps = [i / fps for i in range(count)]
    if max_frames is not None and len(stamps) > max_frames > 0:
        stride = len(stamps) / max_frames
        stamps = [stamps[int(i * stride)] for i in range(max_frames)]
    return stamps


def select_frames(
    candidates: list[ScoredFrame], min_sharpness: float, max_frames: int
) -> list[ScoredFrame]:
    """Keep sharp frames, oldest first, capped with an even stride.

    A lane where every frame is blurry yields an empty list: the caller
    must treat that as a re-film signal, not as a safe street.
    """
    sharp = [c for c in sorted(candidates, key=lambda c: c.timestamp_s)
             if c.sharpness >= min_sharpness]
    if max_frames <= 0 or len(sharp) <= max_frames:
        return sharp if max_frames > 0 else []
    stride = len(sharp) / max_frames
    return [sharp[int(i * stride)] for i in range(max_frames)]
