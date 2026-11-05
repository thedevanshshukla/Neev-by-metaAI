"""Neev inspection web application (FastAPI, file-based, no database).

Endpoints:
  GET  /health                        liveness + capability flags
  POST /inspections                    upload an image (no inference, ever)
  GET  /inspections/{id}               inspection metadata + status
  POST /inspections/{id}/run           cached-first execution (live needs
                                       explicit confirm_live + API key)
  GET  /inspections/{id}/report-data   findings, quality, provenance (offline)
  GET  /inspections/{id}/image         stored original bytes
  GET  /inspections/{id}/assets/{name} overlay/crop PNGs (offline render)
  GET  /inspections/{id}/report        static HTML report (offline render)
  GET  /reports                        inspection index with report flags
  GET  /                               interactive evidence viewer

Launch:
  python -m uvicorn neev.app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from neev.app import runner as runner_mod
from neev.app import reports as reports_mod
from neev.app.store import InspectionStore, io_lock
from neev.app.validation import UploadRejected, validate_upload
from neev.config import Config, load_config
from neev.evidence.quality import assess_image
from neev.persistence.runs import RunStore

STATIC_DIR = Path(__file__).resolve().parent / "static"


class RunRequest(BaseModel):
    confirm_live: bool = False
    phrases: list[str] | None = None
    strategy: str | None = None  # None/"single" (default) or "multiview"
    architecture: str | None = None  # "baseline", "hybrid", "unet_standalone"
    allow_offline_fallback: bool = False
    wonder_mode: str | None = None  # "disabled", "shadow", "advisory"


class ResetHistoryRequest(BaseModel):
    keep_showcase: bool = True


def _masks_available() -> bool:
    try:
        import meta_sam_parser  # noqa: F401
        return True
    except ImportError:
        return False


def _links(inspection_id: str) -> dict[str, str]:
    base = f"/inspections/{inspection_id}"
    return {
        "self": base,
        "run": f"{base}/run",
        "report_data": f"{base}/report-data",
        "image": f"{base}/image",
        "report": f"{base}/report",
    }


def _record_payload(store: InspectionStore, record: dict[str, Any]) -> dict:
    inspection_id = record["inspection_id"]
    idir = store.dir_for(inspection_id)
    return {
        **record,
        "report_exists": (idir / "report.html").is_file(),
        "links": _links(inspection_id),
    }


def create_app(
    config: Config | None = None,
    transport=None,
    runs_dir: str | Path | None = None,
) -> FastAPI:
    """Build the app. Tests inject an isolated config + fake transport."""
    cfg = config or load_config()
    root = Path(runs_dir) if runs_dir is not None else Path(cfg.runs_dir)
    inspections = InspectionStore(root)
    runs = RunStore(root)
    app = FastAPI(title="Neev inspection app")

    @app.api_route("/health", methods=["GET", "HEAD"])
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "app": "neev-inspection",
            "masks_available": _masks_available(),
            "live_configured": bool(cfg.model_api_key),
        }

    @app.post("/inspections", status_code=201)
    async def create_inspection(file: UploadFile = File(...)) -> JSONResponse:
        try:
            raw = await file.read()
            checked = validate_upload(raw)
        except UploadRejected as exc:
            return JSONResponse(
                status_code=exc.status, content={"detail": exc.detail}
            )
        from PIL import Image

        quality = assess_image(Image.open(io.BytesIO(raw)).convert("RGB"))
        # Display-only filename: basename, length-capped, never a path.
        image_name = Path(file.filename or "upload").name[:200] or "upload"
        record = inspections.create(
            raw,
            extension=checked.extension,
            image_name=image_name,
            sha256=checked.sha256,
            mime=checked.mime,
            width=checked.width,
            height=checked.height,
            quality_warnings=[
                {"check": w["check"], "detail": w["detail"]}
                for w in quality.warnings
            ],
            quality_metrics=dict(quality.metrics),
        )
        return JSONResponse(status_code=201, content=_record_payload(
            inspections, record))

    @app.get("/inspections/{inspection_id}")
    def get_inspection(inspection_id: str) -> JSONResponse:
        try:
            record = inspections.get(inspection_id)
        except KeyError:
            return JSONResponse(
                status_code=404, content={"detail": "unknown inspection_id"})
        return JSONResponse(content=_record_payload(inspections, record))

    @app.post("/inspections/{inspection_id}/run")
    def run_inspection(inspection_id: str, body: RunRequest) -> JSONResponse:
        try:
            inspections.get(inspection_id)
        except KeyError:
            return JSONResponse(
                status_code=404, content={"detail": "unknown inspection_id"})
        try:
            record = runner_mod.execute_run(
                inspections, runs, inspection_id, cfg,
                confirm_live=body.confirm_live,
                phrases=body.phrases,
                transport=transport,
                strategy=body.strategy,
                architecture=body.architecture or "baseline",
                allow_offline_fallback=body.allow_offline_fallback,
                wonder_mode=body.wonder_mode,
            )
        except runner_mod.BadPhrases as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})
        except runner_mod.BadStrategy as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})
        except runner_mod.LiveNotConfigured as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})
        return JSONResponse(content=_record_payload(inspections, record))

    @app.get("/inspections/{inspection_id}/report-data")
    def report_data(inspection_id: str) -> JSONResponse:
        try:
            payload = reports_mod.build_report_data(inspections, inspection_id)
        except KeyError:
            return JSONResponse(
                status_code=404, content={"detail": "unknown inspection_id"})
        return JSONResponse(content=payload)

    @app.get("/inspections/{inspection_id}/image")
    def inspection_image(inspection_id: str):
        try:
            path, mime = inspections.image_path(inspection_id)
        except KeyError:
            return JSONResponse(
                status_code=404, content={"detail": "unknown inspection_id"})
        # Bytes under the lock: FileResponse would stream after return,
        # racing a concurrent run's invalidation of the same file.
        with io_lock:
            return Response(content=path.read_bytes(), media_type=mime)

    @app.get("/inspections/{inspection_id}/assets/{name}")
    def inspection_asset(inspection_id: str, name: str):
        try:
            idir = inspections.dir_for(inspection_id)
        except KeyError:
            return JSONResponse(
                status_code=404, content={"detail": "unknown inspection_id"})
        if ("/" in name or "\\" in name or not name.endswith(".png")
                or len(name) > 260):
            return JSONResponse(
                status_code=404, content={"detail": "unknown asset"})
        with io_lock:
            candidate = (idir / "assets" / name).resolve()
            if (candidate.parent != (idir / "assets").resolve()
                    or not candidate.is_file()):
                # Render on demand once: a first viewer hit materializes assets.
                try:
                    rendered = reports_mod.ensure_assets(
                        inspections, inspection_id)
                except Exception:
                    return JSONResponse(
                        status_code=404, content={"detail": "unknown asset"})
                if name not in ([rendered["overlay"]]
                                + list(rendered["crops"].values())):
                    return JSONResponse(
                        status_code=404, content={"detail": "unknown asset"})
                candidate = (idir / "assets" / name).resolve()
                if not candidate.is_file():
                    return JSONResponse(
                        status_code=404, content={"detail": "unknown asset"})
            # Bytes under the lock (see /image above).
            return Response(content=candidate.read_bytes(),
                            media_type="image/png")

    @app.get("/inspections/{inspection_id}/report", response_class=HTMLResponse)
    def inspection_report(inspection_id: str):
        try:
            inspections.get(inspection_id)
        except KeyError:
            return JSONResponse(
                status_code=404, content={"detail": "unknown inspection_id"})
        page = reports_mod.ensure_report_html(inspections, inspection_id)
        return HTMLResponse(content=page)

    @app.get("/reports")
    def report_index() -> dict[str, Any]:
        items = []
        for record in inspections.list_all():
            idir = inspections.root / record["inspection_id"]
            run = record.get("run") or {}
            items.append({
                "inspection_id": record["inspection_id"],
                "image_name": record.get("image_name"),
                "status": record.get("status"),
                "created_utc": record.get("created_utc"),
                "width": record.get("width"),
                "height": record.get("height"),
                "finding_count": None,  # filled by report-data, not the index
                "prompts_screened": None,
                "cache_hits": run.get("cache_hits"),
                "live_calls": run.get("live_calls"),
                "report_exists": (idir / "report.html").is_file(),
                "report_url": f"/inspections/{record['inspection_id']}/report",
                "links": _links(record["inspection_id"]),
            })
        return {"reports": items, "count": len(items)}

    @app.post("/history/reset")
    @app.post("/inspections/reset")
    def reset_history(payload: ResetHistoryRequest | None = None) -> dict[str, Any]:
        """Clean up inspection history, optionally preserving benchmark showcase runs."""
        keep_showcase = True if payload is None else payload.keep_showcase
        keep_ids = {
            "174d4b204d85",
            "e254a194d2b7",
            "28c9ad312990",
            "f408f415dd77",
        } if keep_showcase else set()

        deleted = inspections.reset(keep_ids=keep_ids)
        remaining = inspections.list_all()
        return {
            "status": "ok",
            "deleted_count": deleted,
            "remaining_count": len(remaining),
            "message": f"Successfully cleaned up {deleted} inspection record(s)."
        }

    @app.delete("/history")
    def delete_history() -> dict[str, Any]:
        return reset_history(None)

    async def _static(name: str, media_type: str):
        path = STATIC_DIR / name
        if not path.is_file():
            return JSONResponse(
                status_code=404, content={"detail": "frontend not installed"})
        return FileResponse(path, media_type=media_type)

    @app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
    async def index():
        return await _static("index.html", "text/html")

    @app.api_route("/app.js", methods=["GET", "HEAD"])
    async def app_js():
        return await _static("app.js", "application/javascript")

    @app.api_route("/styles.css", methods=["GET", "HEAD"])
    async def styles_css():
        return await _static("styles.css", "text/css")

    return app


app = create_app()


if __name__ == "__main__":  # python -m neev.app.main
    import os
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
