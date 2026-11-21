"""Loopback HTTP service implementing the LMLM Processor v1 contract."""

from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .backend import LocalOpenAIBackend, ProcessorBackendError
from .protocol import ProcessorRequest, error_response

LOG = logging.getLogger("lmlm_processor")
MAX_BODY_BYTES = 16 * 1024 * 1024


class ProcessorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, backend):
        self.backend = backend
        super().__init__(address, ProcessorHandler)


class ProcessorHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: ProcessorServer

    def log_message(self, fmt, *args):
        LOG.info("%s - %s", self.client_address[0], fmt % args)

    def _send(self, status: int, value: dict):
        payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def do_GET(self):
        if self.path != "/v1/health":
            self._send(404, {"error": {"code": "not_found", "message": "unknown endpoint"}})
            return
        try:
            self.server.backend.health_check()
            self._send(200, {"protocol": "lmlm.processor.v1", "protocol_version": 1, "status": "ready"})
        except ProcessorBackendError as exc:
            self._send(503, {"error": {"code": "backend_unavailable", "message": str(exc)}})

    def do_POST(self):
        if self.path != "/v1/process":
            self._send(404, {"error": {"code": "not_found", "message": "unknown endpoint"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, error_response(None, "invalid_content_length", "Content-Length must be an integer"))
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send(413, error_response(None, "request_too_large", "request body is empty or exceeds the limit"))
            return
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            request = ProcessorRequest.from_dict(value)
            response = self.server.backend.process(request)
            self._send(200, response.to_dict())
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            self._send(400, error_response(None, "invalid_request", str(exc)))
        except ProcessorBackendError as exc:
            self._send(502, error_response(value.get("request_id") if isinstance(value, dict) else None, "backend_error", str(exc)))


def main():
    logging.basicConfig(level=os.getenv("LMLM_LOG_LEVEL", "INFO"))
    host = os.getenv("LMLM_PROCESSOR_HOST", "127.0.0.1")
    if host not in {"127.0.0.1"}:
        raise SystemExit("LMLM processor service may only bind to loopback")
    port = int(os.getenv("LMLM_PROCESSOR_PORT", "8765"))
    inference_url = os.getenv("LMLM_INFERENCE_URL", "http://127.0.0.1:8080/v1")
    backend = LocalOpenAIBackend(inference_url)
    server = ProcessorServer((host, port), backend)
    LOG.info("LMLM Processor v1 listening on %s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("Shutting down LMLM Processor v1")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
