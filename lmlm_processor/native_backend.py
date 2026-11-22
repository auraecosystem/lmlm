"""Processor v1 adapter backed directly by the native LMLM CPU engine."""

from __future__ import annotations

import json
from pathlib import Path

from lmlm_core import GenerationRequest, LMLMEngine, ModelManifest, NativeCPUBackend

from .backend import ProcessorBackendError
from .protocol import ProcessorRequest, ProcessorResponse


class NativeCPUProcessorBackend:
    """Expose an already-installed native LMLM model through Processor v1."""

    def __init__(self, model_dir: str | Path, *, seed: int | None = None, backend=None) -> None:
        self.model_dir = Path(model_dir).resolve()
        manifest_path = self.model_dir / "lmlm-model.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"LMLM model manifest not found: {manifest_path}")
        if manifest_path.stat().st_size > 1024 * 1024:
            raise ValueError("LMLM model manifest exceeds the 1 MiB safety limit")
        try:
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read LMLM model manifest: {exc}") from exc
        self.manifest = ModelManifest.from_dict(raw_manifest)
        self.engine = LMLMEngine(backend if backend is not None else NativeCPUBackend(seed=seed))
        self.engine.load(self.manifest, str(self.model_dir))

    def health_check(self) -> None:
        from lmlm_core import EngineState

        if self.engine.state is not EngineState.READY:
            raise ProcessorBackendError("native LMLM CPU engine is not ready")

    def process(self, request: ProcessorRequest) -> ProcessorResponse:
        if request.modality != "text":
            raise ProcessorBackendError(
                f"native CPU backend does not support modality: {request.modality}"
            )
        try:
            text = request.payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProcessorBackendError("text payload must be UTF-8") from exc
        parameters = request.parameters or {}
        unsupported = set(parameters) - {"max_tokens", "temperature", "top_p"}
        if unsupported:
            raise ProcessorBackendError(
                f"unsupported generation parameters: {', '.join(sorted(unsupported))}"
            )
        prompt = request.prompt
        if prompt:
            text = f"{prompt}\n\n{text}"
        try:
            result = self.engine.generate(
                GenerationRequest(
                    prompt=text,
                    max_new_tokens=parameters.get("max_tokens", 128),
                    temperature=parameters.get("temperature", 0.7),
                    top_p=parameters.get("top_p", 0.9),
                )
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ProcessorBackendError(f"native LMLM generation failed: {exc}") from exc
        return ProcessorResponse(
            request_id=request.request_id,
            content_type="text/plain; charset=utf-8",
            payload=result.text.encode("utf-8"),
        )

    def close(self) -> None:
        self.engine.close()
