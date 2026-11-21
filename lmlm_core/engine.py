"""Backend-neutral LMLM engine lifecycle.

NativeCPUBackend is the first bundled implementation of EngineBackend.
Optimized CPU/GPU and multimodal backends can implement the same contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Protocol

from .manifest import ModelManifest


class EngineState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int):
            raise ValueError("max_new_tokens must be an integer")
        if not 1 <= self.max_new_tokens <= 32768:
            raise ValueError("max_new_tokens must be between 1 and 32768")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise ValueError("temperature must be numeric")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float)):
            raise ValueError("top_p must be numeric")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1")


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("generated text must be a string")
        if self.prompt_tokens < 0 or self.generated_tokens < 0:
            raise ValueError("token counts cannot be negative")
        if self.finish_reason not in {"stop", "length", "cancelled"}:
            raise ValueError("unsupported finish_reason")


class EngineBackend(Protocol):
    """Concrete backend boundary for tokenizer + tensor/model execution."""

    def load(self, manifest: ModelManifest, model_dir: str) -> None: ...

    def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def close(self) -> None: ...


class LMLMEngine:
    """Owns one backend instance and enforces a safe lifecycle."""

    def __init__(self, backend: EngineBackend):
        self._backend = backend
        self._state = EngineState.UNLOADED
        self._lock = RLock()
        self._failure: str | None = None

    @property
    def state(self) -> EngineState:
        with self._lock:
            return self._state

    @property
    def failure(self) -> str | None:
        with self._lock:
            return self._failure

    def load(self, manifest: ModelManifest, model_dir: str) -> None:
        with self._lock:
            if self._state is not EngineState.UNLOADED:
                raise RuntimeError(f"cannot load engine from state {self._state.value}")
            self._state = EngineState.LOADING
            try:
                manifest.resolve_weights(model_dir, require_exists=True)
                self._backend.load(manifest, model_dir)
            except Exception as exc:
                self._failure = f"{type(exc).__name__}: {exc}"
                self._state = EngineState.FAILED
                raise
            self._state = EngineState.READY

    def generate(self, request: GenerationRequest) -> GenerationResult:
        with self._lock:
            if self._state is not EngineState.READY:
                raise RuntimeError(f"engine is not ready (state: {self._state.value})")
            result = self._backend.generate(request)
            if not isinstance(result, GenerationResult):
                raise TypeError("backend.generate must return GenerationResult")
            return result

    def close(self) -> None:
        with self._lock:
            if self._state is EngineState.CLOSED:
                return
            try:
                self._backend.close()
            finally:
                self._state = EngineState.CLOSED
