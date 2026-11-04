"""Configuration from environment variables and an optional .env file.

Secrets are loaded for use, never printed. Use credentials_status() for
safe presence checks and redact() before writing anything to traces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_SAM_MODEL = "sam-3.1"
DEFAULT_WONDER_MODEL = "muse-wonder-1.0"

MODEL_API_KEY_NAME = "MODEL_API_KEY"
MAPILLARY_TOKEN_NAME = "MAPILLARY_TOKEN"

_SECRET_MARKERS = ("key", "token", "secret", "auth")


@dataclass(frozen=True)
class Config:
    """Resolved runtime configuration. model_api_key may be None (offline)."""

    model_api_key: str | None
    mapillary_token: str | None
    base_url: str = DEFAULT_BASE_URL
    sam_model: str = DEFAULT_SAM_MODEL
    wonder_model: str = DEFAULT_WONDER_MODEL
    runs_dir: str = "runs"


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines. Skips blanks and # comments, strips quotes."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, raw = line.partition("=")
        name, raw = name.strip(), raw.strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
            raw = raw[1:-1]
        if name:
            values[name] = raw
    return values


def load_config(
    env: dict[str, str] | None = None,
    dotenv_path: str | Path | None = None,
) -> Config:
    """Environment wins; .env file fills gaps. Missing secrets stay None."""
    source = os.environ if env is None else env
    path = Path(dotenv_path) if dotenv_path else Path(".env")
    file_values = parse_dotenv(path.read_text(encoding="utf-8")) if path.exists() else {}

    def pick(name: str) -> str | None:
        return source.get(name) or file_values.get(name) or None

    return Config(
        model_api_key=pick(MODEL_API_KEY_NAME),
        mapillary_token=pick(MAPILLARY_TOKEN_NAME),
        base_url=source.get("MODEL_API_BASE_URL")
        or file_values.get("MODEL_API_BASE_URL")
        or DEFAULT_BASE_URL,
        sam_model=source.get("SAM_MODEL")
        or file_values.get("SAM_MODEL")
        or DEFAULT_SAM_MODEL,
        wonder_model=source.get("WONDER_MODEL")
        or file_values.get("WONDER_MODEL")
        or DEFAULT_WONDER_MODEL,
        runs_dir=source.get("NEEV_RUNS_DIR")
        or file_values.get("NEEV_RUNS_DIR")
        or "runs",
    )


def credentials_status(config: Config) -> dict[str, bool]:
    """Presence-only report. Safe to print: never contains secret values."""
    return {
        MODEL_API_KEY_NAME: bool(config.model_api_key),
        MAPILLARY_TOKEN_NAME: bool(config.mapillary_token),
    }


def redact(obj):
    """Deep-copy obj with secret-looking values replaced. Safe to print."""

    def scrub(key: str, value):
        if isinstance(key, str) and any(m in key.lower() for m in _SECRET_MARKERS):
            return "[REDACTED]"
        return redact(value)

    if isinstance(obj, dict):
        return {k: scrub(k, v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


def auth_headers(config: Config) -> dict[str, str]:
    """Live Authorization header. NEVER print or store this mapping raw."""
    if not config.model_api_key:
        raise ValueError("MODEL_API_KEY is not configured")
    return {"Authorization": f"Bearer {config.model_api_key}"}
