import json
import threading
import unittest
from urllib.request import Request, urlopen

from lmlm_processor.protocol import ProcessorResponse
from lmlm_processor.server import ProcessorServer


class EchoBackend:
    def health_check(self):
        return None

    def process(self, request):
        return ProcessorResponse(request.request_id, request.content_type, request.payload)


class ProcessorServerTests(unittest.TestCase):
    def setUp(self):
        self.server = ProcessorServer(("127.0.0.1", 0), EchoBackend())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_health_endpoint(self):
        with urlopen(self.base_url + "/v1/health", timeout=2) as response:
            value = json.loads(response.read())
        self.assertEqual(value["protocol"], "lmlm.processor.v1")
        self.assertEqual(value["status"], "ready")

    def test_process_endpoint_preserves_payload_and_request_id(self):
        value = {
            "protocol": "lmlm.processor.v1",
            "protocol_version": 1,
            "request_id": "server-test",
            "modality": "text",
            "content_type": "text/plain",
            "payload_hex": "68656c6c6f",
        }
        request = Request(
            self.base_url + "/v1/process",
            data=json.dumps(value).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            result = json.loads(response.read())
        self.assertEqual(result["request_id"], "server-test")
        self.assertEqual(result["payload_hex"], "68656c6c6f")


if __name__ == "__main__":
    unittest.main()
