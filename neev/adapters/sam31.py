"""SAM 3.1 adapter: building and facade-part segmentation.

Wire contract from the official Meta docs (fetched 2026-10-31):
ai.developer.meta.com/docs/sam/overview.md and
.../sam/reading-segmentation.md: POST {base_url}/responses, model
"sam-3.1", one text phrase plus one image/video, optional
metadata {"mask_encoding": "one_bit"}. The response output_text holds
one line per frame: <Nf> then comma-separated records of
OBJECT_ID + box + mask. Zero matches (empty output_text) is valid.

Neev's own inference outputs are NOT yet validated: the first live
response must be eyeballed before trusting this parser end to end.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from pathlib import Path

from neev.adapters.base import (
    AdapterError,
    BoundingBox,
    Detection,
    Transport,
    UrllibTransport,
    extract_output_text,
)
from neev.config import Config, auth_headers

MODEL = "sam-3.1"
ENDPOINT_PATH = "/responses"
MASK_ENCODING = "one_bit"

# Starting prompt guesses from docs.md section 10. Validate on real
# street photos (day-one experiment) and prune what SAM handles badly.
# Starting prompt guesses from docs.md section 10. Validate on real
# street photos (day-one experiment) and prune what SAM handles badly.
SAM31_PROMPTS: dict[str, list[str]] = {
    "buildings": ["building"],
    "ground_floor": [
        "car parked under building",
        "open ground floor",
        "shop shutter",
        "column",
    ],
    "storeys": ["window", "floor slab", "balcony"],
    "hazards": ["water tank", "hoarding", "parapet", "air conditioner unit"],
    "condition": ["crack in wall", "exposed steel bar"],
    "privacy": ["face", "license plate"],
}

# Supported facade defect targets
DEFECT_PROMPTS = ["crack", "crack in wall", "damaged wall", "exposed steel bar"]

# Official line grammar: line ::= FRAME record ("," record)*,
# record ::= OBJECT_ID box mask. Object ids are stable across frames but
# NOT dense: a line may carry a sparse set like 0, 2. Confidence (;c=)
# appears only with include_confidence; tolerated but never required.
# Example: <0f>0<|box;x1=353;y1=382;x2=453;y2=427;w=537;h=561|>
#          <|mask;x=0;y=0;data=46,101,~...|>,1<|box;...|>...
_TOKEN_RE = re.compile(
    r"(?:<(\d+)f>(\d+)|,(\d+))"
    r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)"
    r"(?:;c=([0-9]+(?:\.[0-9]+)?))?\|>"
    r"(?:<\|mask;x=0;y=0(?:;c=[0-9]+(?:\.[0-9]+)?)?;data=(\d+),(\d+),([^|]+)\|>)?"
)

# Marker -> encoding name, read per mask as the docs require. Never assume
# the requested encoding: box/point prompts always return one_bit.
_MARKER_TO_ENCODING = {"!": "one_bit", "~": "lossless"}


def build_image_request(
    phrase: str,
    image_b64: str,
    mime: str = "image/jpeg",
    stream: bool = False,
) -> dict:
    """One phrase + one image. stream=False keeps the smoke test to one JSON reply."""
    return {
        "model": MODEL,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": phrase},
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,{image_b64}",
                    },
                ],
            }
        ],
        "stream": stream,
        "metadata": {"mask_encoding": MASK_ENCODING},
    }


def build_video_request(
    phrase: str, video_b64: str, stream: bool = True
) -> dict:
    """One phrase + one video. Streaming is recommended for video."""
    return {
        "model": MODEL,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": phrase},
                    {
                        "type": "input_video",
                        "video_url": f"data:video/mp4;base64,{video_b64}",
                    },
                ],
            }
        ],
        "stream": stream,
        "metadata": {"mask_encoding": MASK_ENCODING},
    }


def parse_output_text(output_text: str, phrase: str) -> list[Detection]:
    """Parse SAM token lines into detections.

    Blank text is the documented empty result (zero matches) -> [].
    Non-blank text with no parseable record is a FAILURE -> AdapterError,
    so API drift can never masquerade as "nothing found".
    """
    detections: list[Detection] = []
    if not output_text or not output_text.strip():
        return detections
    frame = 0
    for match in _TOKEN_RE.finditer(output_text):
        if match.group(1) is not None:
            frame = int(match.group(1))
            track_id = match.group(2)
        else:
            track_id = match.group(3)
        box = BoundingBox(
            x0=int(match.group(4)),
            y0=int(match.group(5)),
            x1=int(match.group(6)),
            y1=int(match.group(7)),
        )
        payload = match.group(13)
        marker = payload[0] if payload else None
        confidence = match.group(10)
        detections.append(
            Detection(
                phrase=phrase,
                box=box,
                track_id=track_id,
                frame_index=frame,
                mask_payload=payload,
                mask_width=int(match.group(12)) if payload else None,
                mask_height=int(match.group(11)) if payload else None,
                mask_encoding=_MARKER_TO_ENCODING.get(marker) if marker else None,
                frame_width=int(match.group(8)),
                frame_height=int(match.group(9)),
                confidence=float(confidence) if confidence is not None else None,
            )
        )
    if not detections:
        raise AdapterError(
            "Unparseable SAM output: "
            f"{len(output_text)} non-blank chars hold no box record; "
            f"first 120 chars: {output_text[:120]!r}"
        )
    return detections


# Official pricing (dev.meta.ai/docs/pricing-rate-limits.md, mirrored in
# live/sweep_sam31.py): SAM 3.1 image segmentation at $2.50/1,000 images.
# Used for projected-cost accounting only; never triggers billing.
PRICE_PER_IMAGE_USD = 2.50 / 1000

_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def sniff_image_mime(raw: bytes) -> str | None:
    """MIME type from magic bytes, or None when the file is not an image.

    WebP rides on RIFF (bytes 0:4 == RIFF, 8:12 == WEBP). Anything else is
    rejected before it can burn a billed request on a doomed call.
    """
    for magic, mime in _IMAGE_MAGIC:
        if raw[: len(magic)] == magic:
            return mime
    if len(raw) >= 12 and raw[0:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def sanitized_request(request: dict, image_sha256: str) -> dict:
    """Trace-safe copy of a request: media bytes become a hash reference.

    The full base64 image must never bloat (or leak from) run traces.
    Cache identity also uses this form: same model/phrase/image-sha is an
    exact repeat, so callers must look up and save the SAME form.
    """
    import copy

    shown = copy.deepcopy(request)
    for item in shown.get("input", []):
        for part in item.get("content", []):
            for key in ("image_url", "video_url"):
                if key in part:
                    value = part[key]
                    head = value.partition(";base64,")[0]
                    part[key] = {
                        "media": head,
                        "sha256": image_sha256,
                        "chars": len(value),
                    }
    return shown


@dataclass
class SAM31Result:
    phrase: str
    model: str
    detections: list[Detection] = field(default_factory=list)
    raw_output_text: str = ""
    raw_response: dict = field(default_factory=dict)


class SAM31Adapter:
    """Thin wrapper: build request, POST via transport, parse detections."""

    def __init__(self, config: Config, transport: Transport | None = None):
        self.config = config
        self.transport: Transport = transport or UrllibTransport()
        self.url = config.base_url.rstrip("/") + ENDPOINT_PATH

    def segment_image(
        self,
        image: bytes | Path,
        phrase: str,
        mime: str = "image/jpeg",
        timeout: float = 180.0,
    ) -> SAM31Result:
        raw = Path(image).read_bytes() if isinstance(image, Path) else image
        payload = build_image_request(phrase, base64.b64encode(raw).decode("ascii"), mime)
        try:
            response = self.transport.post_json(
                self.url, auth_headers(self.config), payload, timeout
            )
        except ValueError as exc:  # missing API key
            raise AdapterError(str(exc)) from exc
        text = extract_output_text(response)
        return SAM31Result(
            phrase=phrase, model=MODEL, detections=parse_output_text(text, phrase),
            raw_output_text=text, raw_response=response,
        )
