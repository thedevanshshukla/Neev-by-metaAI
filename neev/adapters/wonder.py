"""Muse Wonder adapter: checklist judging with evidence references.

Request shape follows the draft in docs.md section 10 (OpenAI Responses
style: instructions + one user message holding feature text and images).
The image field names are UNVERIFIED against a live call: confirm on the
first Wonder experiment and update build_judge_request if the API differs.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from neev.adapters.base import (
    AdapterError,
    Transport,
    UrllibTransport,
    extract_output_text,
)
from neev.config import Config, auth_headers

MODEL = "muse-wonder-1.0"
ENDPOINT_PATH = "/responses"

WONDER_INPUT_PRICE_PER_M_USD = 4.00
WONDER_OUTPUT_PRICE_PER_M_USD = 20.00


def compute_wonder_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Compute exact cost based on Meta Model API token pricing for Muse Wonder."""
    in_cost = (input_tokens * WONDER_INPUT_PRICE_PER_M_USD) / 1_000_000.0
    out_cost = (output_tokens * WONDER_OUTPUT_PRICE_PER_M_USD) / 1_000_000.0
    return round(in_cost + out_cost, 6)


def build_judge_request(
    features: dict,
    images_b64: list[str],
    instructions: str,
    mime: str = "image/jpeg",
) -> dict:
    """Features as JSON text first, then one input_image part per photo."""
    content: list[dict] = [
        {"type": "input_text", "text": json.dumps(features, sort_keys=True)}
    ]
    for image_b64 in images_b64:
        content.append(
            {"type": "input_image", "image_url": f"data:{mime};base64,{image_b64}"}
        )
    return {
        "model": MODEL,
        "instructions": instructions,
        "input": [{"role": "user", "content": content}],
    }


def extract_json(text: str) -> dict:
    """Parse model JSON, tolerating wrapper text outside the braces."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise AdapterError(
                "Judge reply is not JSON; first 200 chars: " + text[:200]
            )
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise AdapterError(
                "Judge reply is not valid JSON: " + str(exc)
            ) from exc
    if not isinstance(parsed, dict):
        raise AdapterError("Judge reply JSON is not an object")
    return parsed


def parse_judge_response(response: dict) -> dict:
    """output_text -> dict. Schema checks live in neev.validation."""
    return extract_json(extract_output_text(response))


class WonderAdapter:
    """Thin wrapper: build judge request, POST via transport, parse JSON."""

    def __init__(self, config: Config, transport: Transport | None = None):
        self.config = config
        self.transport: Transport = transport or UrllibTransport()
        self.url = config.base_url.rstrip("/") + ENDPOINT_PATH

    def judge(
        self,
        features: dict,
        images: list[bytes | Path],
        instructions: str,
        timeout: float = 180.0,
    ) -> dict:
        detailed = self.judge_detailed(features, images, instructions, timeout=timeout)
        return detailed["parsed"]

    def judge_detailed(
        self,
        features: dict,
        images: list[bytes | Path],
        instructions: str,
        timeout: float = 180.0,
    ) -> dict:
        encoded = []
        for image in images:
            raw = Path(image).read_bytes() if isinstance(image, Path) else image
            encoded.append(base64.b64encode(raw).decode("ascii"))
        payload = build_judge_request(features, encoded, instructions)
        try:
            response = self.transport.post_json(
                self.url, auth_headers(self.config), payload, timeout
            )
        except ValueError as exc:  # missing API key
            raise AdapterError(str(exc)) from exc
        parsed = parse_judge_response(response)
        req_id = response.get("id")
        usage = response.get("usage", {})
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        cost = compute_wonder_cost_usd(in_tok, out_tok)
        return {
            "parsed": parsed,
            "raw_response": response,
            "request_id": req_id,
            "usage": usage,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": cost,
        }
