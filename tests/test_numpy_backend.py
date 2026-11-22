import tempfile
import unittest
from pathlib import Path

try:
    import numpy  # noqa: F401
except ImportError:
    numpy = None

from lmlm_core import NativeCPUBackend
from lmlm_core.numpy_backend import NumpyCPUBackend
from tests.test_native_cpu_backend import make_tiny_model


@unittest.skipIf(numpy is None, "NumPy is not installed")
class NumpyBackendTests(unittest.TestCase):
    def test_vectorized_logits_match_reference_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = make_tiny_model(directory)
            reference = NativeCPUBackend(seed=1)
            optimized = NumpyCPUBackend(seed=1)
            reference.load(manifest, str(directory))
            optimized.load(manifest, str(directory))
            tokens = reference._tokenizer.encode("tiny test", add_bos=True)
            expected = reference._model.logits(tokens)
            actual = optimized._model.logits(tokens)
            for left, right in zip(expected, actual):
                self.assertAlmostEqual(left, right, places=8)
            reference.close()
            optimized.close()


if __name__ == "__main__":
    unittest.main()
