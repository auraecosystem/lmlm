"""Optional NumPy-accelerated implementation of the LMLM reference transformer."""

from __future__ import annotations

import math
from pathlib import Path

from .cpu_model import CPUTransformer


class NumpyCPUTransformer(CPUTransformer):
    """Vectorized CPU transformer; NumPy is imported only when this class is used."""

    def __init__(self, weights_path: str | Path):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "NumPy backend requires NumPy; install it with: pip install 'numpy>=1.26'"
            ) from exc
        self._np = np
        super().__init__(weights_path)
        self.token_embedding = np.asarray(self.token_embedding, dtype=np.float64)
        self.position_embedding = np.asarray(self.position_embedding, dtype=np.float64)
        self.final_norm_scale = np.asarray(self.final_norm_scale, dtype=np.float64)
        self.final_norm_bias = np.asarray(self.final_norm_bias, dtype=np.float64)
        converted = []
        for layer in self.layers:
            converted.append({name: np.asarray(value, dtype=np.float64) for name, value in layer.items()})
        self.layers = converted

    def _norm(self, values, scale, bias):
        np = self._np
        mean = values.mean(axis=-1, keepdims=True)
        variance = ((values - mean) ** 2).mean(axis=-1, keepdims=True)
        return (values - mean) / np.sqrt(variance + 1e-5) * scale + bias

    def logits(self, token_ids: list[int]) -> list[float]:
        np = self._np
        if not token_ids:
            raise ValueError("at least one input token is required")
        if len(token_ids) > self.max_seq_len:
            raise ValueError("input exceeds model max_seq_len")
        if any(isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < self.vocab_size for token in token_ids):
            raise ValueError("input token ID outside model vocabulary")

        positions = np.arange(len(token_ids))
        hidden = self.token_embedding[np.asarray(token_ids)] + self.position_embedding[positions]
        head_size = self.head_size
        for layer in self.layers:
            normalized = self._norm(hidden, layer["ln1_scale"], layer["ln1_bias"])
            q = normalized @ layer["q_weight"].T + layer["q_bias"]
            k = normalized @ layer["k_weight"].T + layer["k_bias"]
            v = normalized @ layer["v_weight"].T + layer["v_bias"]
            length = len(token_ids)
            q = q.reshape(length, self.num_heads, head_size).transpose(1, 0, 2)
            k = k.reshape(length, self.num_heads, head_size).transpose(1, 0, 2)
            v = v.reshape(length, self.num_heads, head_size).transpose(1, 0, 2)
            scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(head_size)
            causal_mask = np.triu(np.ones((length, length), dtype=bool), k=1)
            scores = np.where(causal_mask[None, :, :], -1e30, scores)
            scores = scores - scores.max(axis=-1, keepdims=True)
            probabilities = np.exp(scores)
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            attended = (probabilities @ v).transpose(1, 0, 2).reshape(length, self.hidden_size)
            projected = attended @ layer["o_weight"].T + layer["o_bias"]
            hidden = hidden + projected
            normalized = self._norm(hidden, layer["ln2_scale"], layer["ln2_bias"])
            expanded = normalized @ layer["up_weight"].T + layer["up_bias"]
            activated = 0.5 * expanded * (
                1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (expanded + 0.044715 * expanded ** 3))
            )
            contracted = activated @ layer["down_weight"].T + layer["down_bias"]
            hidden = hidden + contracted

        final = self._norm(hidden[-1], self.final_norm_scale, self.final_norm_bias)
        return (final @ self.token_embedding.T).tolist()
