"""Optional multi-view detection strategy (Phase 8, steps 1-4).

Pure view planning, coordinate reconciliation, and provenance behind
an explicit runner strategy flag. Single-view remains the default;
nothing here executes inference on import.
"""

from neev.multiview import evaluate, provenance, raster, reconcile, views

__all__ = ["evaluate", "provenance", "raster", "reconcile", "views"]
