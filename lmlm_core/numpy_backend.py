"""Optional NumPy-accelerated backend for local LMLM inference."""

from __future__ import annotations

from pathlib import Path

from .cpu_backend import NativeCPUBackend
from .manifest import ModelManifest
from .numpy_model import NumpyCPUTransformer
from .tokenizer import ByteLevelBPETokenizer


class NumpyCPUBackend(NativeCPUBackend):
    """Drop-in backend using vectorized NumPy tensor operations."""

    def load(self, manifest: ModelManifest, model_dir: str) -> None:
        if manifest.architecture != "lmlm-cpu-transformer-v1":
            raise ValueError("manifest architecture must be lmlm-cpu-transformer-v1")
        if manifest.tensor_dtype != "f32":
            raise ValueError("NumPy CPU backend currently supports f32 model artifacts")
        weights_path = manifest.resolve_weights(model_dir, require_exists=True)
        tokenizer_path = self._resolve_local(model_dir, manifest.tokenizer)
        tokenizer = ByteLevelBPETokenizer.from_file(tokenizer_path)
        model = NumpyCPUTransformer(weights_path)
        if model.vocab_size != len(tokenizer.vocab) or model.vocab_size != manifest.vocabulary_size:
            raise ValueError("model and tokenizer vocabulary sizes do not match the manifest")
        eos_id = tokenizer.vocab.get(tokenizer.eos_token) if tokenizer.eos_token else None
        if model.eos_token_id != eos_id:
            raise ValueError("model and tokenizer EOS token IDs do not match")
        if model.max_seq_len != manifest.context_length:
            raise ValueError("model max_seq_len does not match manifest context_length")
        self._tokenizer = tokenizer
        self._model = model
