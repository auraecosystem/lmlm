"""Local-only inference adapter for an OpenAI-compatible llama.cpp server."""

from __future__ import annotations

import http.client
import json
from urllib.parse import urlsplit

from .protocol import ProcessorRequest, ProcessorResponse


class ProcessorBackendError(RuntimeError):
    """The configured local inference backend could not process a request."""


class LocalOpenAIBackend:
    """Use a local inference server; remote hosts are deliberately rejected.

    The model itself remains a local LMLM deployment artifact. This adapter does
    not download weights, call cloud APIs, or silently fall back to a remote model.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1", timeout: float = 120.0):
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("LMLM inference URL must use plain HTTP on a loopback address")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("credentials, query strings, and fragments are not allowed in inference URL")
        if timeout <= 0 or timeout > 600:
            raise ValueError("timeout must be between 0 and 600 seconds")
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.base_path = parsed.path.rstrip("/")
        self.timeout = timeout

    def _connection(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def health_check(self) -> None:
        connection = self._connection()
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            response.read(4096)
            if response.status not in (200, 204):
                raise ProcessorBackendError(f"local inference health check returned HTTP {response.status}")
        except OSError as exc:
            raise ProcessorBackendError(f"local inference server unavailable: {exc}") from exc
        finally:
            connection.close()

    def process(self, request: ProcessorRequest) -> ProcessorResponse:
        if request.modality != "text":
            raise ProcessorBackendError(f"local text backend does not support modality: {request.modality}")
        try:
            text = request.payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProcessorBackendError("text payload must be UTF-8") from exc
        params = request.parameters or {}
        max_tokens = params.get("max_tokens", 512)
        temperature = params.get("temperature", 0.2)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 4096:
            raise ProcessorBackendError("max_tokens must be an integer between 1 and 4096")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            raise ProcessorBackendError("temperature must be between 0 and 2")
        messages = []
        if request.prompt:
            messages.append({"role": "system", "content": request.prompt})
        messages.append({"role": "user", "content": text})
        body = json.dumps({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": float(temperature),
            "stream": False,
        }).encode("utf-8")
        connection = self._connection()
        try:
            connection.request(
                "POST",
                f"{self.base_path}/chat/completions",
                body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            response = connection.getresponse()
            raw = response.read(8 * 1024 * 1024)
            if response.status != 200:
                raise ProcessorBackendError(f"local inference server returned HTTP {response.status}")
            result = json.loads(raw.decode("utf-8"))
            output = result["choices"][0]["message"]["content"]
            if not isinstance(output, str):
                raise ProcessorBackendError("local inference server returned a non-text completion")
            return ProcessorResponse(request.request_id, "text/plain; charset=utf-8", output.encode("utf-8"))
        except (OSError, UnicodeDecodeError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            if isinstance(exc, ProcessorBackendError):
                raise
            raise ProcessorBackendError(f"local inference request failed: {exc}") from exc
        finally:
            connection.close()
