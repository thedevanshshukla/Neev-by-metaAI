"""Upload validation for the Neev inspection app.

Trusts bytes, not filenames: the MIME type comes from magic bytes via
``sniff_image_mime``, and readability/dimensions come from actually
decoding the image with Pillow. Client-provided extensions and
Content-Type headers are ignored for every security decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from neev.adapters.sam31 import sniff_image_mime
from neev.persistence.runs import sha256_bytes

# Rejects anything bigger before decoding. 15 MiB comfortably covers
# phone photos while bounding memory and disk per upload.
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
# Pillow can report absurd dimensions for corrupt headers; real facade
# photos are far smaller than this guard.
MAX_SIDE_PX = 12_000

_MIME_TO_EXT = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "image/webp": "webp",
}


@dataclass(frozen=True)
class ValidatedUpload:
    mime: str
    extension: str
    width: int
    height: int
    sha256: str


class UploadRejected(Exception):
    """An upload failed validation. Carries the HTTP status to return."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def validate_upload(raw: bytes) -> ValidatedUpload:
    """Validate raw upload bytes. Raises UploadRejected on any problem."""
    if not raw:
        raise UploadRejected(400, "empty file: no image bytes received")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise UploadRejected(
            413,
            f"file is {len(raw)} bytes (limit {MAX_UPLOAD_BYTES}); "
            "upload a smaller image",
        )
    mime = sniff_image_mime(raw)
    if mime is None:
        raise UploadRejected(
            415,
            "unsupported file: magic bytes do not match "
            "JPEG, PNG, GIF, BMP, or WebP",
        )
    try:
        from PIL import Image
    except ImportError as exc:
        raise UploadRejected(
            500, "image decoding is unavailable on this server"
        ) from exc
    import io

    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.load()
            width, height = probe.size
    except Exception as exc:
        raise UploadRejected(
            400, f"unreadable image: file header claims {mime} "
            f"but pixels do not decode ({type(exc).__name__})"
        ) from exc
    if width <= 0 or height <= 0:
        raise UploadRejected(400, "unreadable image: zero dimension")
    if max(width, height) > MAX_SIDE_PX:
        raise UploadRejected(
            400,
            f"image is {width}x{height}px (limit {MAX_SIDE_PX}px per side)",
        )
    return ValidatedUpload(
        mime=mime,
        extension=_MIME_TO_EXT[mime],
        width=width,
        height=height,
        sha256=sha256_bytes(raw),
    )
