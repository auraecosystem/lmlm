"""Validation and serialization helpers for the LMLM Processor v1 contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1
SUPPORTED_MODALITIES = {"text", "image", "audio", "video", "multimodal"}
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024


class ProtocolError(ValueError):
    """A request does not satisfy the processor protocol."""


@dataclass(frozen=True)
class ProcessorRequest:
    request_id: str
    modality: str
    content_type: str
    payload: bytes
    prompt: str | None = None
    parameters: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProcessorRequest":
        if value.get("protocol") != "lmlm.processor.v1" or value.get("protocol_version") != PROTOCOL_VERSION:
            raise ProtocolError("unsupported processor protocol version")
        request_id = value.get("request_id")
        modality = value.get("modality")
        content_type = value.get("content_type")
        payload_hex = value.get("payload_hex")
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ProtocolError("request_id must be a non-empty string of at most 128 characters")
        if modality not in SUPPORTED_MODALITIES:
            raise ProtocolError("unsupported modality")
        if not isinstance(content_type, str) or not content_type or len(content_type) > 255:
            raise ProtocolError("content_type must be a non-empty string of at most 255 characters")
        if not isinstance(payload_hex, str) or len(payload_hex) > MAX_PAYLOAD_BYTES * 2:
            raise ProtocolError("payload_hex is missing or exceeds the payload limit")
        try:
            payload = bytes.fromhex(payload_hex)
        except ValueError as exc:
            raise ProtocolError("payload_hex must contain valid hexadecimal data") from exc
        if len(payload) > MAX_PAYLOAD_BYTES:
            raise ProtocolError("payload exceeds the payload limit")
        prompt = value.get("prompt")
        if prompt is not None and (not isinstance(prompt, str) or len(prompt) > 32_000):
            raise ProtocolError("prompt must be a string of at most 32000 characters")
        parameters = value.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ProtocolError("parameters must be an object")
        return cls(request_id, modality, content_type, payload, prompt, parameters)


@dataclass(frozen=True)
class ProcessorResponse:
    request_id: str
    content_type: str
    payload: bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": "lmlm.processor.v1",
            "protocol_version": PROTOCOL_VERSION,
            "request_id": self.request_id,
            "content_type": self.content_type,
            "payload_hex": self.payload.hex(),
        }


def error_response(request_id: str | None, code: str, message: str) -> dict[str, Any]:
    return {
        "protocol": "lmlm.processor.v1",
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "error": {"code": code, "message": message[:1000]},
    }
