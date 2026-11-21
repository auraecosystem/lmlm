import json
import tempfile
import unittest
from pathlib import Path

from lmlm_core import (
    ByteLevelBPETokenizer,
    GenerationRequest,
    LMLMEngine,
    ModelFormatError,
    ModelManifest,
    NativeCPUBackend,
    TokenizerError,
)
from lmlm_core.tokenizer import _BYTE_ENCODER
from lmlm_processor.native_backend import NativeCPUProcessorBackend
from lmlm_processor.protocol import ProcessorRequest


def make_tiny_model(directory: Path) -> ModelManifest:
    vocab = {character: byte for byte, character in _BYTE_ENCODER.items()}
    vocab["<|bos|>"] = 256
    tokenizer = {
        "format_version": 1,
        "vocab": vocab,
        "merges": [],
        "bos_token": "<|bos|>",
    }
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")

    hidden = 2
    embedding = [[0.0, 0.0] for _ in range(257)]
    embedding[ord("!")] = [1.0, 0.0]
    layer = {
        "ln1_scale": [1.0, 1.0],
        "ln1_bias": [0.0, 0.0],
        "ln2_scale": [1.0, 1.0],
        "ln2_bias": [0.0, 0.0],
        "q_weight": [[0.0] * hidden for _ in range(hidden)],
        "q_bias": [0.0] * hidden,
        "k_weight": [[0.0] * hidden for _ in range(hidden)],
        "k_bias": [0.0] * hidden,
        "v_weight": [[0.0] * hidden for _ in range(hidden)],
        "v_bias": [0.0] * hidden,
        "o_weight": [[0.0] * hidden for _ in range(hidden)],
        "o_bias": [0.0] * hidden,
        "up_weight": [[0.0] * hidden for _ in range(hidden)],
        "up_bias": [0.0] * hidden,
        "down_weight": [[0.0] * hidden for _ in range(hidden)],
        "down_bias": [0.0] * hidden,
    }
    model = {
        "format": "lmlm.cpu.transformer.v1",
        "config": {
            "vocab_size": 257,
            "hidden_size": 2,
            "num_heads": 1,
            "num_layers": 1,
            "intermediate_size": 2,
            "max_seq_len": 16,
        },
        "weights": {
            "token_embedding": embedding,
            "position_embedding": [[0.0, 0.0] for _ in range(16)],
            "final_norm_scale": [1.0, 1.0],
            "final_norm_bias": [1.0, 0.0],
            "layers": [layer],
        },
    }
    (directory / "weights.json").write_text(json.dumps(model), encoding="utf-8")
    return ModelManifest.from_dict({
        "format_version": 1,
        "model_id": "lmlm-cpu-test",
        "architecture": "lmlm-cpu-transformer-v1",
        "tokenizer": "tokenizer.json",
        "weights": "weights.json",
        "tensor_dtype": "f32",
        "context_length": 16,
        "vocabulary_size": 257,
    })


class NativeTokenizerTests(unittest.TestCase):
    def test_byte_level_round_trip(self):
        vocab = {character: byte for byte, character in _BYTE_ENCODER.items()}
        tokenizer = ByteLevelBPETokenizer(vocab, [])
        text = "LMLM — local 🧠"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_missing_token_without_unknown_token_fails(self):
        tokenizer = ByteLevelBPETokenizer({"x": 0}, [])
        with self.assertRaises(TokenizerError):
            tokenizer.encode("y")


class NativeCPUBackendTests(unittest.TestCase):
    def test_loads_local_assets_and_generates_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = make_tiny_model(directory)
            engine = LMLMEngine(NativeCPUBackend(seed=7))
            engine.load(manifest, str(directory))
            result = engine.generate(GenerationRequest("x", max_new_tokens=2, temperature=0))
            self.assertEqual(result.text, "!!")
            self.assertEqual(result.generated_tokens, 2)
            self.assertEqual(result.finish_reason, "length")
            engine.close()

    def test_processor_v1_runs_on_native_cpu_engine(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = make_tiny_model(directory)
            (directory / "lmlm-model.json").write_text(json.dumps({
                "format_version": 1,
                "model_id": manifest.model_id,
                "architecture": manifest.architecture,
                "tokenizer": manifest.tokenizer,
                "weights": manifest.weights,
                "tensor_dtype": manifest.tensor_dtype,
                "context_length": manifest.context_length,
                "vocabulary_size": manifest.vocabulary_size,
            }), encoding="utf-8")
            backend = NativeCPUProcessorBackend(directory, seed=11)
            backend.health_check()
            response = backend.process(ProcessorRequest(
                request_id="native-processor-test",
                modality="text",
                content_type="text/plain",
                payload=b"x",
                parameters={"max_tokens": 2, "temperature": 0},
            ))
            self.assertEqual(response.request_id, "native-processor-test")
            self.assertEqual(response.payload.decode("utf-8"), "!!")
            backend.close()

    def test_rejects_weights_outside_float32_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = make_tiny_model(directory)
            weights_path = directory / "weights.json"
            weights = json.loads(weights_path.read_text(encoding="utf-8"))
            weights["weights"]["token_embedding"][0][0] = 1e100
            weights_path.write_text(json.dumps(weights), encoding="utf-8")
            with self.assertRaises(ModelFormatError):
                NativeCPUBackend().load(manifest, str(directory))

    def test_rejects_tokenizer_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = make_tiny_model(directory)
            unsafe = ModelManifest.from_dict({
                "format_version": 1,
                "model_id": "unsafe",
                "architecture": "lmlm-cpu-transformer-v1",
                "tokenizer": "../tokenizer.json",
                "weights": "weights.json",
                "tensor_dtype": "f32",
                "context_length": 16,
                "vocabulary_size": 257,
            })
            backend = NativeCPUBackend()
            with self.assertRaises(ValueError):
                backend.load(unsafe, str(directory))


if __name__ == "__main__":
    unittest.main()
