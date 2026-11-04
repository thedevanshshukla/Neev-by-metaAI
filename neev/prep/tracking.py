"""Greedy IoU tracking fallback with stable building IDs (pure).

Used when SAM 3.1 video tracking is unavailable or hits its limits
(docs.md risk table). When SAM returns track_ids, prefer those and use
this only to stitch per-frame calls together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from neev.adapters.base import BoundingBox


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection over union in [0, 1]. Degenerate boxes score 0."""
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = a.area + b.area - inter
    return inter / union if union else 0.0


@dataclass
class Track:
    track_id: str
    box: BoundingBox
    misses: int = 0
    hits: int = 1


@dataclass
class Tracker:
    """Greedy best-first matcher. IDs look like B1, B2, ..."""

    iou_threshold: float = 0.3
    max_missing: int = 2
    id_prefix: str = "B"
    _tracks: dict[str, Track] = field(default_factory=dict, repr=False)
    _next: int = field(default=1, repr=False)

    @property
    def tracks(self) -> dict[str, Track]:
        return dict(self._tracks)

    def update(self, boxes: list[BoundingBox]) -> dict[int, str]:
        """Match boxes to tracks; returns {box_index: track_id}."""
        assignment: dict[int, str] = {}
        if not boxes:
            self._age(set())
            return assignment
        scored: list[tuple[float, str, int]] = []
        for index, box in enumerate(boxes):
            for tid, track in self._tracks.items():
                scored.append((iou(track.box, box), tid, index))
        scored.sort(reverse=True)
        used_tracks, used_boxes = set(), set()
        for score, tid, index in scored:
            if score < self.iou_threshold or tid in used_tracks or index in used_boxes:
                continue
            used_tracks.add(tid)
            used_boxes.add(index)
            assignment[index] = tid
            track = self._tracks[tid]
            track.box = boxes[index]
            track.misses = 0
            track.hits += 1
        for index, box in enumerate(boxes):
            if index in used_boxes:
                continue
            tid = f"{self.id_prefix}{self._next}"
            self._next += 1
            self._tracks[tid] = Track(track_id=tid, box=box)
            assignment[index] = tid
        self._age(set(assignment.values()))
        return assignment

    def _age(self, seen: set[str]) -> None:
        for tid in list(self._tracks):
            if tid not in seen:
                track = self._tracks[tid]
                track.misses += 1
                if track.misses > self.max_missing:
                    del self._tracks[tid]
