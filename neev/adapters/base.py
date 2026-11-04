"""Shared adapter types and HTTP transport (standard library only).

Adapters convert between Neev dataclasses and the Meta Model API Responses
endpoint (POST {base_url}/responses). Network I/O goes through a Transport
so unit tests can substitute a fake and never touch the network.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class AdapterError(Exception):
    """Raised for transport failures and unparseable model responses."""


@dataclass(frozen=True)
class BoundingBox:
    """Pixel box, origin at top-left. x1/y1 are exclusive."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict[str, int]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass(frozen=True)
class Detection:
    """One segmented object for one text prompt.

    track_id is the wire object id (stable across frames, not dense).
    frame_width/height echo the source size from the box token for
    coordinate validation. mask_encoding is "one_bit"/"lossless" as read
    from the payload marker, never assumed from the request. source names
    the origin (a run id) once the detection is traced; None until then,
    so unattributed detections are visible, not silently sourceless.
    """

    phrase: str
    box: BoundingBox
    track_id: str = "0"
    frame_index: int = 0
    mask_payload: str | None = None
    mask_width: int | None = None
    mask_height: int | None = None
    mask_encoding: str | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    confidence: float | None = None
    source: str | None = None


class Transport(Protocol):
    """POST a JSON payload, return the decoded JSON response."""

    def post_json(
        self, url: str, headers: dict[str, str], payload: dict, timeout: float
    ) -> dict:
        ...


_HEADER_SECRET_MARKERS = ("authoriz", "api-key", "apikey", "token", "secret")


def _scrub_header_values(text: str, headers: dict[str, str]) -> str:
    """Remove secret header values from server-controlled error text.

    A hostile or buggy server echo must not turn an error message into a
    credential leak. Only values of secret-looking headers are scrubbed,
    so ordinary messages pass through untouched.
    """
    for name, value in headers.items():
        if not value or len(value) < 8:
            continue
        if any(m in name.lower() for m in _HEADER_SECRET_MARKERS):
            text = text.replace(value, "[REDACTED]")
    return text


class UrllibTransport:
    """Real transport over urllib. No third-party dependency."""

    def post_json(
        self, url: str, headers: dict[str, str], payload: dict, timeout: float = 120.0
    ) -> dict:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={**headers, "Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            detail = _scrub_header_values(detail, headers)
            raise AdapterError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise AdapterError(f"Request to {url} failed: {exc.reason}") from exc


def extract_output_text(response: dict) -> str:
    """Pull output_text out of a Responses API payload.

    Accepts the top-level "output_text" convenience field or the full
    output[{type: message, content: [...]}] shape. Raises AdapterError
    (listing available keys) when neither is present, so schema drift
    fails loudly instead of silently yielding nothing.
    """
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts: list[str] = []
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text = part.get("text", "")
                    if isinstance(text, str):
                        parts.append(text)
    if parts:
        return "".join(parts)
    raise AdapterError(
        "No output_text in response; top-level keys: "
        + ", ".join(sorted(str(k) for k in response))
    )
