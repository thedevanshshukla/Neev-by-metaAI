"""File-based inspection persistence for the Neev app (JSON, no database).

Layout under <runs_dir>/inspections/:
  <inspection_id>/inspection.json   metadata, quality, run record
  <inspection_id>/original.<ext>    uploaded bytes (server-chosen extension)
  <inspection_id>/assets/           overlay + crops rendered on demand
  <inspection_id>/report.html       static report rendered on demand

Inspection ids are server-generated (uuid hex, like RunStore run ids),
so concurrent uploads never collide and client filenames never reach
the filesystem. The runs/ cache and CLI run dirs are untouched: this
store only adds the inspections/ subtree.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

INSPECTIONS_SUBDIR = "inspections"
ORIGINAL_STEM = "original"

# Serializes every inspection-dir mutation and read (run invalidation,
# asset/report rendering, record loads/stores). A browser fires these
# concurrently: without it, POST /run can unlink an overlay while GET
# /assets is writing it (a 500 on Windows) and readers can catch
# partial writes. Re-entrant: update() locks and is called holding it.
io_lock = threading.RLock()

# Server-generated ids only; this also blocks path traversal in URLs.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_inspection_id(inspection_id: str) -> str:
    """Return the id, or raise KeyError (unknown/malformed)."""
    if not isinstance(inspection_id, str) or not _ID_RE.match(inspection_id):
        raise KeyError(f"unknown inspection_id: {inspection_id!r}")
    return inspection_id


class InspectionStore:
    def __init__(self, runs_dir: str | Path):
        self.runs_dir = Path(runs_dir)
        self.root = self.runs_dir / INSPECTIONS_SUBDIR
        self.root.mkdir(parents=True, exist_ok=True)

    def new_id(self) -> str:
        """A fresh unique inspection id (retries on the unlikely collision)."""
        for _ in range(10):
            candidate = uuid.uuid4().hex[:12]
            if not (self.root / candidate).exists():
                return candidate
        raise RuntimeError("could not mint a unique inspection id")

    def dir_for(self, inspection_id: str) -> Path:
        """Directory for an id. Raises KeyError if malformed or missing."""
        check_inspection_id(inspection_id)
        path = self.root / inspection_id
        # Defense in depth: the regex already forbids separators, but
        # never trust a path without confirming containment.
        if path.resolve() != (self.root / inspection_id).resolve():
            raise KeyError(f"unknown inspection_id: {inspection_id!r}")
        if not path.is_dir():
            raise KeyError(f"unknown inspection_id: {inspection_id!r}")
        return path

    def create(
        self,
        raw: bytes,
        *,
        extension: str,
        image_name: str,
        sha256: str,
        mime: str,
        width: int,
        height: int,
        quality_warnings: list[dict[str, str]],
        quality_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist a new uploaded inspection. Returns its record."""
        inspection_id = self.new_id()
        idir = self.root / inspection_id
        idir.mkdir(parents=False, exist_ok=False)
        (idir / f"{ORIGINAL_STEM}.{extension}").write_bytes(raw)
        record = {
            "inspection_id": inspection_id,
            # Display only: the client filename is never a filesystem path.
            "image_name": image_name[:200] if image_name else "upload",
            "stored_image": f"{ORIGINAL_STEM}.{extension}",
            "sha256": sha256,
            "mime": mime,
            "width": width,
            "height": height,
            "quality_warnings": quality_warnings,
            "quality_metrics": quality_metrics,
            "status": "uploaded",
            "created_utc": utcnow(),
            "run": None,
        }
        self._write(idir, record)
        return record

    def get(self, inspection_id: str) -> dict[str, Any]:
        with io_lock:
            idir = self.dir_for(inspection_id)
            return json.loads(
                (idir / "inspection.json").read_text(encoding="utf-8"))

    def update(self, inspection_id: str, record: dict[str, Any]) -> dict[str, Any]:
        with io_lock:
            idir = self.dir_for(inspection_id)
            record = dict(record)
            record["inspection_id"] = inspection_id
            self._write(idir, record)
            return record

    def image_path(self, inspection_id: str) -> tuple[Path, str]:
        """(path, mime) of the stored original. KeyError if missing."""
        record = self.get(inspection_id)
        path = self.dir_for(inspection_id) / record["stored_image"]
        if not path.is_file():
            raise KeyError(f"unknown inspection_id: {inspection_id!r}")
        return path, record["mime"]

    def list_all(self) -> list[dict[str, Any]]:
        """All inspection records, newest first. Skips partial dirs."""
        with io_lock:
            records = []
            for child in self.root.iterdir():
                if not child.is_dir():
                    continue
                meta = child / "inspection.json"
                if not meta.is_file():
                    continue
                try:
                    records.append(json.loads(meta.read_text(encoding="utf-8")))
                except (ValueError, OSError):
                    continue
            records.sort(key=lambda r: r.get("created_utc", ""), reverse=True)
            return records

    def reset(self, keep_ids: set[str] | None = None) -> int:
        """Remove inspection directories. If keep_ids is provided, skip those ids.
        Returns the number of deleted inspections.
        """
        import os
        import stat
        import shutil

        def _remove_readonly(func, path, excinfo):
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                pass

        deleted = 0
        keep = set(keep_ids) if keep_ids else set()
        with io_lock:
            if not self.root.exists():
                return 0
            for child in list(self.root.iterdir()):
                if not child.is_dir():
                    continue
                if child.name in keep:
                    continue
                try:
                    shutil.rmtree(child, onerror=_remove_readonly)
                    deleted += 1
                except Exception:
                    try:
                        shutil.rmtree(child, ignore_errors=True)
                        deleted += 1
                    except Exception:
                        pass
            return deleted

    @staticmethod
    def _write(idir: Path, record: dict[str, Any]) -> None:
        (idir / "inspection.json").write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
