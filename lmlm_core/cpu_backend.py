"""Native CPU implementation of the LMLM EngineBackend contract."""

from __future__ import annotations

import random
from pathlib import Path

from .cpu_model import CPUTransformer
from .engine import GenerationRequest, GenerationResult
from .manifest import ModelManifest
from .tokenizer import ByteLevelBPETokenizer


class NativeCPUBackend:
    """Load a local BPE tokenizer and reference CPU transformer.

    All assets are resolved under the model directory. No network, subprocess,
    dynamic module import, or remote model fallback is used.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        self._seed = seed
        self._model: CPUTransformer | None = None
        self._tokenizer: ByteLevelBPETokenizer | None = None
        self._rng = random.Random(seed)

    @staticmethod
    def _resolve_local(model_dir: str | Path, relative_path: str) -> Path:
        root = Path(model_dir).resolve()
        candidate = (root / relative_path).resolve()
        if candidate == root or root not in candidate.parents:
            raise ValueError("model asset path escapes the model directory")
        if not candidate.is_file():
            raise FileNotFoundError(f"local model asset not found: {candidate}")
        return candidate

    def load(self, manifest: ModelManifest, model_dir: str) -> None:
        weights_path = manifest.resolve_weights(model_dir, require_exists=True)
        tokenizer_path = self._resolve_local(model_dir, manifest.tokenizer)
        tokenizer = ByteLevelBPETokenizer.from_file(tokenizer_path)
        model = CPUTransformer(weights_path)
        if model.vocab_size != len(tokenizer.vocab):
            raise ValueError(
                f"model vocabulary size ({model.vocab_size}) does not match "
                f"tokenizer vocabulary size ({len(tokenizer.vocab)})"
            )
        if tokenizer.eos_token is not None:
            eos_id = tokenizer.vocab[tokenizer.eos_token]
            if model.eos_token_id is not None and model.eos_token_id != eos_id:
                raise ValueError("model and tokenizer EOS token IDs do not match")
        self._tokenizer = tokenizer
        self._model = model

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("native CPU backend is not loaded")
        prompt_ids = self._tokenizer.encode(request.prompt, add_bos=True)
        if not prompt_ids:
            raise ValueError("tokenizer produced an empty prompt")
        output_ids = self._model.generate(
            prompt_ids,
            request.max_new_tokens,
            temperature=float(request.temperature),
            top_p=float(request.top_p),
            rng=self._rng,
        )
        generated_ids = output_ids[len(prompt_ids):]
        stop_id = self._tokenizer.vocab.get(self._tokenizer.eos_token) if self._tokenizer.eos_token else None
        if stop_id is not None and generated_ids and generated_ids[-1] == stop_id:
            generated_ids = generated_ids[:-1]
            finish_reason = "stop"
        else:
            finish_reason = "length"
        text = self._tokenizer.decode(generated_ids)
        return GenerationResult(
            text=text,
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(generated_ids),
            finish_reason=finish_reason,
        )

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
