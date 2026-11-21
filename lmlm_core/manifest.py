"""Strict manifest parsing for local LMLM model artifacts.

This module validates metadata and local artifact paths. It does not load
weights, execute tensors, or fetch model files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ManifestError(ValueError):
    """A model manifest is missing required or safe metadata."""


@dataclass(frozen=True)
class ModelManifest:
    format_version: int
    model_id: str
    architecture: str
    tokenizer: str
    weights: str
    tensor_dtype: str
    context_length: int
    vocabulary_size: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelManifest":
        if not isinstance(value, Mapping):
            raise ManifestError("manifest must be a JSON object")
        if value.get("format_version") != 1:
            raise ManifestError("unsupported manifest format_version")
        fields = {
            "model_id": (str, 128),
            "architecture": (str, 128),
            "tokenizer": (str, 256),
            "weights": (str, 512),
            "tensor_dtype": (str, 32),
        }
        normalized: dict[str, str] = {}
        for name, (kind, limit) in fields.items():
            item = value.get(name)
            if not isinstance(item, kind) or not item.strip() or len(item) > limit:
                raise ManifestError(f"{name} must be a non-empty string of at most {limit} characters")
            normalized[name] = item.strip()

        for name in ("model_id", "architecture", "tokenizer", "tensor_dtype"):
            if any(ord(char) < 32 for char in normalized[name]):
                raise ManifestError(f"{name} contains control characters")

        weights = Path(normalized["weights"])
        if weights.is_absolute() or ".." in weights.parts:
            raise ManifestError("weights must be a relative path inside the model directory")
        if not weights.parts or normalized["weights"].strip() in {".", ""}:
            raise ManifestError("weights must identify a local artifact")

        context_length = value.get("context_length")
        vocabulary_size = value.get("vocabulary_size")
        for name, number, minimum, maximum in (
            ("context_length", context_length, 1, 2_000_000),
            ("vocabulary_size", vocabulary_size, 1, 10_000_000),
        ):
            if isinstance(number, bool) or not isinstance(number, int) or not minimum <= number <= maximum:
                raise ManifestError(f"{name} must be an integer between {minimum} and {maximum}")

        return cls(
            format_version=1,
            model_id=normalized["model_id"],
            architecture=normalized["architecture"],
            tokenizer=normalized["tokenizer"],
            weights=normalized["weights"],
            tensor_dtype=normalized["tensor_dtype"],
            context_length=context_length,
            vocabulary_size=vocabulary_size,
        )

    def resolve_weights(self, model_dir: str | Path, *, require_exists: bool = True) -> Path:
        """Resolve weights without allowing the path to escape model_dir."""
        root = Path(model_dir).resolve()
        candidate = (root / self.weights).resolve()
        if candidate != root and root not in candidate.parents:
            raise ManifestError("weights path escapes the model directory")
        if require_exists and not candidate.is_file():
            raise ManifestError(f"model weights not found: {candidate}")
        return candidate
