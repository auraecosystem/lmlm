import unittest

from lmlm_processor.backend import LocalOpenAIBackend
from lmlm_processor.protocol import ProcessorRequest, ProcessorResponse, ProtocolError


class ProtocolTests(unittest.TestCase):
    def test_request_round_trip_payload(self):
        value = {
            "protocol": "lmlm.processor.v1",
            "protocol_version": 1,
            "request_id": "test-001",
            "modality": "text",
            "content_type": "text/plain; charset=utf-8",
            "payload_hex": "68656c6c6f",
            "prompt": "Be concise",
            "parameters": {"max_tokens": 12},
        }
        request = ProcessorRequest.from_dict(value)
        self.assertEqual(request.payload, b"hello")
        self.assertEqual(request.prompt, "Be concise")

    def test_response_uses_versioned_envelope(self):
        value = ProcessorResponse("test-001", "text/plain", b"ok").to_dict()
        self.assertEqual(value["protocol"], "lmlm.processor.v1")
        self.assertEqual(value["protocol_version"], 1)
        self.assertEqual(value["payload_hex"], "6f6b")

    def test_rejects_unknown_protocol_version(self):
        with self.assertRaises(ProtocolError):
            ProcessorRequest.from_dict({"protocol": "lmlm.processor.v2", "protocol_version": 2})

    def test_rejects_invalid_hex_payload(self):
        value = {"protocol": "lmlm.processor.v1", "protocol_version": 1, "request_id": "x",
                 "modality": "text", "content_type": "text/plain", "payload_hex": "not-hex"}
        with self.assertRaises(ProtocolError):
            ProcessorRequest.from_dict(value)


class LocalBackendSafetyTests(unittest.TestCase):
    def test_rejects_non_loopback_inference_url(self):
        with self.assertRaises(ValueError):
            LocalOpenAIBackend("https://api.example.com/v1")

    def test_accepts_loopback_inference_url(self):
        backend = LocalOpenAIBackend("http://127.0.0.1:8080/v1")
        self.assertEqual(backend.host, "127.0.0.1")
        self.assertEqual(backend.port, 8080)


if __name__ == "__main__":
    unittest.main()
