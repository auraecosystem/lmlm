# LMLM Native Engine Foundation

This change establishes the first backend-neutral lifecycle contract for the
native LMLM model runtime. It is deliberately not presented as a working neural
network implementation.

## Components

- `ModelManifest`: validates model identity, architecture, tokenizer,
  tensor dtype, context length, vocabulary size, and a relative local weights path.
- `EngineBackend`: boundary that a real tokenizer + tensor/model backend must
  implement (`load`, `generate`, `close`).
- `LMLMEngine`: owns one backend and enforces the lifecycle
  `unloaded -> loading -> ready`, with terminal `failed` and `closed` states.
- `GenerationRequest` / `GenerationResult`: validated inference inputs and
  typed outputs.

## Model manifest v1

A model directory may contain `lmlm-model.json`:

```json
{
  "format_version": 1,
  "model_id": "lmlm-small-v1",
  "architecture": "lmlm-transformer-v1",
  "tokenizer": "lmlm-bpe-v1",
  "weights": "weights/model.bin",
  "tensor_dtype": "f32",
  "context_length": 2048,
  "vocabulary_size": 32000
}
```

The weights path must be relative and cannot escape the model directory.
The manifest loader does not download files or execute model code.

## Current implementation boundary

This foundation provides lifecycle and validation only. It does not yet include:

1. LMLM tokenizer implementation or vocabulary assets.
2. Tensor operators, attention kernels, or a CPU/GPU execution backend.
3. Weight deserialization, quantization, or checkpoint conversion.
4. Native text generation or multimodal encoders/decoders.

The existing Processor v1 service remains an adapter to a separately running
loopback inference process. It must not be confused with the native backend.

## Tests

Run all repository tests with:

```bash
python -m unittest discover -s tests -v
```
