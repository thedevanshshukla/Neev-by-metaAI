"""Neev inspection web application (Phase 7).

Thin FastAPI layer over the existing CLI pipeline: uploads land in
per-inspection dirs, runs reuse SAM31Adapter + RunStore + the
sanitized-trace cache, and evidence reuses the Phase 6 report modules.
``create_app`` needs FastAPI and is imported lazily so the pure
helpers (validation, store, runner, reports) stay importable without
the web extra.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from neev.app.main import create_app as _create_app

        return _create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["create_app"]
