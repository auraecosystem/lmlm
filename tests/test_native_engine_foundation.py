import tempfile
import unittest
from pathlib import Path

from lmlm_core import (
    EngineState,
    GenerationRequest,
    GenerationResult,
    LMLMEngine,
    ManifestError,
    ModelManifest,
)


def manifest():
    return ModelManifest.from_dict({
        "format_version": 1,
        "model_id": "lmlm-test",
        "architecture": "lmlm-transformer-v1",
        "tokenizer": "lmlm-bpe-v1",
        "weights": "weights/model.bin",
        "tensor_dtype": "f32",
        "context_length": 2048,
        "vocabulary_size": 32000,
    })


class FakeBackend:
    def __init__(self):
        self.loaded = False
        self.closed = False

    def load(self, model_manifest, model_dir):
        self.loaded = True

    def generate(self, request):
        return GenerationResult("local test output", 3, 4, "stop")

    def close(self):
        self.closed = True


class ManifestTests(unittest.TestCase):
    def test_rejects_parent_traversal(self):
        value = {
            "format_version": 1, "model_id": "test", "architecture": "test",
            "tokenizer": "test", "weights": "../outside.bin", "tensor_dtype": "f32",
            "context_length": 128, "vocabulary_size": 100,
        }
        with self.assertRaises(ManifestError):
            ModelManifest.from_dict(value)

    def test_resolves_weights_inside_model_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "weights" / "model.bin"
            target.parent.mkdir()
            target.write_bytes(b"test")
            self.assertEqual(manifest().resolve_weights(directory), target.resolve())


class EngineTests(unittest.TestCase):
    def test_lifecycle_and_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "weights" / "model.bin"
            weights.parent.mkdir()
            weights.write_bytes(b"fixture")
            backend = FakeBackend()
            engine = LMLMEngine(backend)
            self.assertEqual(engine.state, EngineState.UNLOADED)
            engine.load(manifest(), directory)
            self.assertEqual(engine.state, EngineState.READY)
            result = engine.generate(GenerationRequest("hello"))
            self.assertEqual(result.text, "local test output")
            engine.close()
            self.assertTrue(backend.closed)
            self.assertEqual(engine.state, EngineState.CLOSED)
            with self.assertRaises(RuntimeError):
                engine.generate(GenerationRequest("hello"))

    def test_load_failure_is_terminal(self):
        backend = FakeBackend()
        engine = LMLMEngine(backend)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ManifestError):
                engine.load(manifest(), directory)
        self.assertEqual(engine.state, EngineState.FAILED)
        self.assertIsNotNone(engine.failure)


if __name__ == "__main__":
    unittest.main()
