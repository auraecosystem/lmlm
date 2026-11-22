"""Train a reproducible tiny byte-level next-token model for LMLM.

This trainer intentionally creates a small educational model, not a
production-scale language model. It trains tied token embeddings with a
softmax next-byte objective and emits the existing LMLM CPU model format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    import numpy as np
except ImportError:  # Keep the core package importable without the optional trainer dependency.
    np = None

from lmlm_core.tokenizer import _BYTE_ENCODER


def _layer_norm(values):
    mean = values.mean(axis=1, keepdims=True)
    centered = values - mean
    inv_std = 1.0 / np.sqrt((centered * centered).mean(axis=1, keepdims=True) + 1e-5)
    return centered * inv_std, inv_std


def train(
    input_path: Path,
    output_dir: Path,
    *,
    steps: int = 500,
    batch_size: int = 64,
    hidden_size: int = 32,
    learning_rate: float = 0.03,
    seed: int = 7,
    resume: bool = False,
) -> dict:
    if np is None:
        raise RuntimeError("tiny-model training requires NumPy; install it with: pip install numpy")
    if not input_path.is_file():
        raise FileNotFoundError(f"training text file not found: {input_path}")
    if input_path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("tiny trainer input is limited to 8 MiB")
    if steps < 1 or batch_size < 1 or not 2 <= hidden_size <= 256:
        raise ValueError("steps/batch_size must be positive and hidden_size must be 2..256")
    if not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must be greater than 0 and at most 1")
    corpus = input_path.read_bytes()
    if len(corpus) < 2:
        raise ValueError("training corpus must contain at least two bytes")
    tokens = np.frombuffer(corpus, dtype=np.uint8).astype(np.int64)
    # BOS is a dedicated token; every byte retains its stable byte ID.
    inputs = np.concatenate((np.asarray([256], dtype=np.int64), tokens[:-1]))
    targets = tokens
    vocab_size = 257
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.npz"
    start_step = 0
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError("--resume requested but checkpoint.npz is missing")
        checkpoint = np.load(checkpoint_path)
        embedding = checkpoint["embedding"].astype(np.float64)
        first_moment = checkpoint["first_moment"].astype(np.float64)
        second_moment = checkpoint["second_moment"].astype(np.float64)
        start_step = int(checkpoint["step"])
        if embedding.shape != (vocab_size, hidden_size):
            raise ValueError("checkpoint shape does not match requested hidden_size")
    else:
        embedding = rng.normal(0.0, 0.02, size=(vocab_size, hidden_size))
        first_moment = np.zeros_like(embedding)
        second_moment = np.zeros_like(embedding)

    history = []
    for step in range(start_step + 1, start_step + steps + 1):
        indices = rng.integers(0, len(inputs), size=batch_size)
        x = inputs[indices]
        y = targets[indices]
        selected = embedding[x]
        normalized, inv_std = _layer_norm(selected)
        logits = normalized @ embedding.T
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        loss = -np.log(np.maximum(probabilities[np.arange(batch_size), y], 1e-12)).mean()

        grad_logits = probabilities
        grad_logits[np.arange(batch_size), y] -= 1.0
        grad_logits /= batch_size
        grad_embedding = grad_logits.T @ normalized
        grad_normalized = grad_logits @ embedding
        mean_grad = grad_normalized.mean(axis=1, keepdims=True)
        mean_projected = (grad_normalized * normalized).mean(axis=1, keepdims=True)
        grad_input = inv_std * (
            grad_normalized - mean_grad - normalized * mean_projected
        )
        np.add.at(grad_embedding, x, grad_input)

        first_moment = 0.9 * first_moment + 0.1 * grad_embedding
        second_moment = 0.999 * second_moment + 0.001 * (grad_embedding ** 2)
        corrected_first = first_moment / (1.0 - 0.9 ** step)
        corrected_second = second_moment / (1.0 - 0.999 ** step)
        embedding -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        if step == start_step + 1 or step % max(1, steps // 10) == 0 or step == start_step + steps:
            history.append({"step": step, "loss": float(loss)})

    vocab = {character: byte for byte, character in _BYTE_ENCODER.items()}
    vocab["<|bos|>"] = 256
    tokenizer = {
        "format_version": 1,
        "vocab": vocab,
        "merges": [],
        "bos_token": "<|bos|>",
    }
    (output_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    zero = lambda rows, columns: np.zeros((rows, columns), dtype=np.float64).tolist()
    layer = {
        "ln1_scale": [1.0] * hidden_size,
        "ln1_bias": [0.0] * hidden_size,
        "ln2_scale": [1.0] * hidden_size,
        "ln2_bias": [0.0] * hidden_size,
        "q_weight": zero(hidden_size, hidden_size),
        "q_bias": [0.0] * hidden_size,
        "k_weight": zero(hidden_size, hidden_size),
        "k_bias": [0.0] * hidden_size,
        "v_weight": zero(hidden_size, hidden_size),
        "v_bias": [0.0] * hidden_size,
        "o_weight": zero(hidden_size, hidden_size),
        "o_bias": [0.0] * hidden_size,
        "up_weight": zero(hidden_size, hidden_size),
        "up_bias": [0.0] * hidden_size,
        "down_weight": zero(hidden_size, hidden_size),
        "down_bias": [0.0] * hidden_size,
    }
    weights = {
        "format": "lmlm.cpu.transformer.v1",
        "config": {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_heads": 1,
            "num_layers": 1,
            "intermediate_size": hidden_size,
            "max_seq_len": 256,
            "eos_token_id": None,
        },
        "weights": {
            "token_embedding": embedding.tolist(),
            "position_embedding": zero(256, hidden_size),
            "final_norm_scale": [1.0] * hidden_size,
            "final_norm_bias": [0.0] * hidden_size,
            "layers": [layer],
        },
    }
    weights_path = output_dir / "weights.json"
    weights_path.write_text(json.dumps(weights, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest = {
        "format_version": 1,
        "model_id": "lmlm-tiny-byte-v1",
        "architecture": "lmlm-cpu-transformer-v1",
        "tokenizer": "tokenizer.json",
        "weights": "weights.json",
        "tensor_dtype": "f32",
        "context_length": 256,
        "vocabulary_size": vocab_size,
    }
    (output_dir / "lmlm-model.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    np.savez(
        checkpoint_path,
        embedding=embedding,
        first_moment=first_moment,
        second_moment=second_moment,
        step=np.asarray(start_step + steps),
    )
    metadata = {
        "trainer": "lmlm-tiny-byte-trainer-v1",
        "seed": seed,
        "steps_this_run": steps,
        "total_steps": start_step + steps,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "corpus_bytes": len(corpus),
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "loss_samples": history,
        "checkpoint": "checkpoint.npz",
        "model_format": "lmlm.cpu.transformer.v1",
        "note": "Educational tiny byte-level next-token model; not a production pretrained model.",
    }
    (output_dir / "training.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny local LMLM byte model")
    parser.add_argument("--input", required=True, type=Path, help="UTF-8 text corpus")
    parser.add_argument("--output", required=True, type=Path, help="local model output directory")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true", help="continue from output/checkpoint.npz")
    args = parser.parse_args()
    result = train(
        args.input,
        args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
