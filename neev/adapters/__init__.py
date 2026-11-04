"""Model adapters: request builders, response parsers, transports."""

from neev.adapters.base import (
    AdapterError,
    BoundingBox,
    Detection,
    UrllibTransport,
    extract_output_text,
)

__all__ = [
    "AdapterError",
    "BoundingBox",
    "Detection",
    "UrllibTransport",
    "extract_output_text",
]
