import tempfile
import unittest
from pathlib import Path

try:
    import numpy  # noqa: F401
except ImportError:
    numpy = None

from lmlm_core import GenerationRequest, NativeCPUBackend, ModelManifest
from lmlm_training.train import train


@unittest.skipIf(numpy is None, "NumPy is not installed")
class TinyTrainingTests(unittest.TestCase):
    def test_training_emits_loadable_model_and_resumable_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.txt"
            corpus.write_text("lmlm local model training sample " * 8, encoding="utf-8")
            model_dir = root / "model"
            metadata = train(corpus, model_dir, steps=4, batch_size=8, hidden_size=8, seed=19)
            self.assertEqual(metadata["total_steps"], 4)
            manifest = ModelManifest.from_dict(
                __import__("json").loads((model_dir / "lmlm-model.json").read_text())
            )
            backend = NativeCPUBackend(seed=1)
            backend.load(manifest, str(model_dir))
            result = backend.generate(GenerationRequest("lmlm", max_new_tokens=3, temperature=0))
            self.assertEqual(result.generated_tokens, 3)
            backend.close()

            resumed = train(
                corpus, model_dir, steps=2, batch_size=8, hidden_size=8, seed=19, resume=True
            )
            self.assertEqual(resumed["total_steps"], 6)


if __name__ == "__main__":
    unittest.main()
