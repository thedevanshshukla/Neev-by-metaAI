"""File-based run store: caching and traceability (JSON, no database).

Layout under runs_dir:
  <run_id>/request.json    canonical request (prompts, params, input hashes)
  <run_id>/response.json   raw model response (parseable bytes, verbatim)
  <run_id>/meta.json       model ids, prompt text hashes, timestamps, notes
  cache.json               request-hash -> run_id for exact-repeat calls

Image/video bytes are hashed, not embedded: large binaries would bloat
every trace. Keep the source files next to the run or under versioned
sample data and record their paths in meta.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neev.config import redact


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_json(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class RunMeta:
    purpose: str
    models: dict[str, str]  # e.g. {"segment": "sam-3.1"}
    input_hashes: dict[str, str] = field(default_factory=dict)
    prompt_hashes: dict[str, str] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
    created_utc: str = ""

    def finalize(self) -> "RunMeta":
        if not self.created_utc:
            self.created_utc = datetime.now(timezone.utc).isoformat()
        return self


class RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _cache_path(self) -> Path:
        return self.root / "cache.json"

    def read_cache(self) -> dict[str, str]:
        path = self._cache_path()
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def find_cached(self, request: dict) -> str | None:
        """Return the run_id of an identical earlier request, if any."""
        return self.read_cache().get(sha256_json(request))

    def save(
        self, request: dict, response: dict, meta: RunMeta
    ) -> str:
        """Persist a run; secrets in meta are scrubbed before writing."""
        meta.finalize()
        run_id = uuid.uuid4().hex[:12]
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "request.json").write_text(
            json.dumps(request, indent=2, sort_keys=True), encoding="utf-8"
        )
        (run_dir / "response.json").write_text(
            json.dumps(response, indent=2, sort_keys=True), encoding="utf-8"
        )
        (run_dir / "meta.json").write_text(
            json.dumps(redact(asdict(meta)), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        cache = self.read_cache()
        cache.setdefault(sha256_json(request), run_id)
        self._cache_path().write_text(json.dumps(cache, indent=2), encoding="utf-8")
        return run_id

    def load(self, run_id: str) -> dict[str, Any]:
        """Load request/response/meta for a run_id. KeyError if missing."""
        run_dir = self.root / run_id
        if not run_dir.is_dir():
            raise KeyError(f"unknown run_id: {run_id}")
        return {
            name: json.loads((run_dir / f"{name}.json").read_text(encoding="utf-8"))
            for name in ("request", "response", "meta")
        }
