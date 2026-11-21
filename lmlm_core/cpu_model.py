"""Small decoder-only transformer executed with pure-Python CPU operations."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any


class ModelFormatError(ValueError):
    """Invalid LMLM CPU model weights or architecture."""


def _float32(value: int | float, name: str) -> float:
    try:
        converted = struct.unpack("!f", struct.pack("!f", float(value)))[0]
    except (OverflowError, struct.error) as exc:
        raise ModelFormatError(f"{name} value is outside the float32 range") from exc
    if not math.isfinite(converted):
        raise ModelFormatError(f"{name} contains a non-finite float32 value")
    return converted


def _matrix(value: Any, rows: int, columns: int, name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise ModelFormatError(f"{name} must have {rows} rows")
    result: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            raise ModelFormatError(f"{name} rows must have {columns} columns")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in row):
            raise ModelFormatError(f"{name} contains a non-numeric value")
        result.append([_float32(item, name) for item in row])
    return result


def _vector(value: Any, size: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ModelFormatError(f"{name} must contain {size} values")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ModelFormatError(f"{name} contains a non-numeric value")
    return [_float32(item, name) for item in value]


def _linear(vector: list[float], weight: list[list[float]], bias: list[float]) -> list[float]:
    return [
        sum(value * weight[row][column] for column, value in enumerate(vector)) + bias[row]
        for row in range(len(weight))
    ]


def _layer_norm(values: list[float], scale: list[float], bias: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    inverse = 1.0 / math.sqrt(variance + 1e-5)
    return [(value - mean) * inverse * scale[index] + bias[index] for index, value in enumerate(values)]


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(max(-80.0, value - maximum)) for value in values]
    total = sum(exponents)
    return [value / total for value in exponents]


class CPUTransformer:
    """Reference GPT-like decoder with learned positions and tied embeddings.

    This intentionally favors portability and inspectability over speed. It is
    suitable for tiny fixtures and correctness tests, not large production models.
    """

    def __init__(self, weights_path: str | Path):
        weights_file = Path(weights_path)
        if weights_file.stat().st_size > 64 * 1024 * 1024:
            raise ModelFormatError("reference CPU weights file exceeds the 64 MiB safety limit")
        try:
            value = json.loads(weights_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelFormatError(f"cannot read model weights: {exc}") from exc
        if not isinstance(value, dict) or value.get("format") != "lmlm.cpu.transformer.v1":
            raise ModelFormatError("unsupported CPU transformer weights format")
        config = value.get("config")
        raw = value.get("weights")
        if not isinstance(config, dict) or not isinstance(raw, dict):
            raise ModelFormatError("model config and weights must be JSON objects")
        self.vocab_size = self._integer(config, "vocab_size", 1, 1000000)
        self.hidden_size = self._integer(config, "hidden_size", 1, 4096)
        self.num_heads = self._integer(config, "num_heads", 1, 256)
        self.num_layers = self._integer(config, "num_layers", 1, 64)
        self.intermediate_size = self._integer(config, "intermediate_size", 1, 16384)
        self.max_seq_len = self._integer(config, "max_seq_len", 1, 131072)
        if self.hidden_size % self.num_heads:
            raise ModelFormatError("hidden_size must be divisible by num_heads")
        if self.hidden_size > 512 or self.num_layers > 8 or self.intermediate_size > 2048:
            raise ModelFormatError("reference CPU backend dimension limits exceeded")
        parameter_budget = (
            self.vocab_size * self.hidden_size
            + self.max_seq_len * self.hidden_size
            + self.num_layers * (
                4 * self.hidden_size * self.hidden_size
                + 2 * self.intermediate_size * self.hidden_size
                + 8 * self.hidden_size
                + 2 * self.intermediate_size
            )
            + 2 * self.hidden_size
        )
        if parameter_budget > 5_000_000:
            raise ModelFormatError("model exceeds the 5-million-parameter reference CPU budget")
        attention_work = self.num_layers * self.max_seq_len * self.max_seq_len * self.hidden_size
        feed_forward_work = 2 * self.num_layers * self.max_seq_len * self.hidden_size * self.intermediate_size
        if attention_work + feed_forward_work > 250_000_000:
            raise ModelFormatError("model exceeds the reference CPU compute budget")
        self.eos_token_id = config.get("eos_token_id")
        if self.eos_token_id is not None and (
            isinstance(self.eos_token_id, bool)
            or not isinstance(self.eos_token_id, int)
            or not 0 <= self.eos_token_id < self.vocab_size
        ):
            raise ModelFormatError("eos_token_id is outside the vocabulary")
        self.head_size = self.hidden_size // self.num_heads
        self.token_embedding = _matrix(raw.get("token_embedding"), self.vocab_size, self.hidden_size, "token_embedding")
        self.position_embedding = _matrix(raw.get("position_embedding"), self.max_seq_len, self.hidden_size, "position_embedding")
        self.final_norm_scale = _vector(raw.get("final_norm_scale"), self.hidden_size, "final_norm_scale")
        self.final_norm_bias = _vector(raw.get("final_norm_bias"), self.hidden_size, "final_norm_bias")
        layers = raw.get("layers")
        if not isinstance(layers, list) or len(layers) != self.num_layers:
            raise ModelFormatError(f"layers must contain {self.num_layers} entries")
        self.layers = [self._load_layer(layer, index) for index, layer in enumerate(layers)]

    @staticmethod
    def _integer(config: dict, name: str, minimum: int, maximum: int) -> int:
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ModelFormatError(f"{name} must be an integer between {minimum} and {maximum}")
        return value

    def _load_layer(self, layer: Any, index: int) -> dict[str, Any]:
        if not isinstance(layer, dict):
            raise ModelFormatError(f"layer {index} must be an object")
        h, inner = self.hidden_size, self.intermediate_size
        result: dict[str, Any] = {}
        for name in ("ln1_scale", "ln1_bias", "ln2_scale", "ln2_bias"):
            result[name] = _vector(layer.get(name), h, f"layers[{index}].{name}")
        for name in ("q", "k", "v", "o"):
            result[name + "_weight"] = _matrix(layer.get(name + "_weight"), h, h, f"layers[{index}].{name}_weight")
            result[name + "_bias"] = _vector(layer.get(name + "_bias"), h, f"layers[{index}].{name}_bias")
        result["up_weight"] = _matrix(layer.get("up_weight"), inner, h, f"layers[{index}].up_weight")
        result["up_bias"] = _vector(layer.get("up_bias"), inner, f"layers[{index}].up_bias")
        result["down_weight"] = _matrix(layer.get("down_weight"), h, inner, f"layers[{index}].down_weight")
        result["down_bias"] = _vector(layer.get("down_bias"), h, f"layers[{index}].down_bias")
        return result

    def logits(self, token_ids: list[int]) -> list[float]:
        if not token_ids:
            raise ValueError("at least one input token is required")
        if len(token_ids) > self.max_seq_len:
            raise ValueError("input exceeds model max_seq_len")
        if any(isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < self.vocab_size for token in token_ids):
            raise ValueError("input token ID outside model vocabulary")
        hidden = [
            [self.token_embedding[token][column] + self.position_embedding[position][column]
             for column in range(self.hidden_size)]
            for position, token in enumerate(token_ids)
        ]
        for layer in self.layers:
            normalized = [_layer_norm(row, layer["ln1_scale"], layer["ln1_bias"]) for row in hidden]
            q = [_linear(row, layer["q_weight"], layer["q_bias"]) for row in normalized]
            k = [_linear(row, layer["k_weight"], layer["k_bias"]) for row in normalized]
            v = [_linear(row, layer["v_weight"], layer["v_bias"]) for row in normalized]
            attended = [[0.0] * self.hidden_size for _ in hidden]
            for position in range(len(hidden)):
                for head in range(self.num_heads):
                    start, end = head * self.head_size, (head + 1) * self.head_size
                    scores = [
                        sum(q[position][d] * k[past][d] for d in range(start, end)) / math.sqrt(self.head_size)
                        for past in range(position + 1)
                    ]
                    probabilities = _softmax(scores)
                    for offset in range(self.head_size):
                        attended[position][start + offset] = sum(
                            probabilities[past] * v[past][start + offset]
                            for past in range(position + 1)
                        )
            for position, row in enumerate(hidden):
                projected = _linear(attended[position], layer["o_weight"], layer["o_bias"])
                hidden[position] = [row[i] + projected[i] for i in range(self.hidden_size)]
                normalized_row = _layer_norm(hidden[position], layer["ln2_scale"], layer["ln2_bias"])
                expanded = _linear(normalized_row, layer["up_weight"], layer["up_bias"])
                activated = [0.5 * value * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (value + 0.044715 * value ** 3))) for value in expanded]
                contracted = _linear(activated, layer["down_weight"], layer["down_bias"])
                hidden[position] = [hidden[position][i] + contracted[i] for i in range(self.hidden_size)]
        final = _layer_norm(hidden[-1], self.final_norm_scale, self.final_norm_bias)
        return [sum(final[column] * self.token_embedding[token][column] for column in range(self.hidden_size)) for token in range(self.vocab_size)]
