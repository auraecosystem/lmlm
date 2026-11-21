# Native CPU Inference v1

LMLM now includes a dependency-free reference CPU path that executes a small
decoder-only transformer directly inside the LMLM runtime.

This is a real local forward-pass and autoregressive generation implementation.
It is intentionally a correctness/reference backend, not a high-performance
runtime for large models.

## Runtime components

- `ByteLevelBPETokenizer`: byte-level BPE tokenizer loaded from local JSON.
- `CPUTransformer`: learned token/position embeddings, pre-normalized
  multi-head causal self-attention, GELU feed-forward layers, residual paths,
  final layer normalization, and tied token-embedding output projection.
- `NativeCPUBackend`: implements the LMLM `EngineBackend` contract and
  performs greedy or temperature/top-p sampling.
- `LMLMEngine`: owns the backend lifecycle and rejects generation before load.

All model assets are local files. The implementation performs no network
requests, subprocess execution, dynamic imports, or cloud fallback.

## Model directory

A model directory contains:

```text
model/
├── lmlm-model.json
├── tokenizer.json
└── weights.json
```

Example `lmlm-model.json`:

```json
{
  "format_version": 1,
  "model_id": "lmlm-tiny-v1",
  "architecture": "lmlm-cpu-transformer-v1",
  "tokenizer": "tokenizer.json",
  "weights": "weights.json",
  "tensor_dtype": "f32",
  "context_length": 128,
  "vocabulary_size": 260
}
```

The manifest's tokenizer and weights paths must remain inside the model
directory. No automatic downloads are performed.

### Tokenizer JSON

```json
{
  "format_version": 1,
  "vocab": {
    "byte-token-0": 0,
    "byte-token-1": 1,
    "...": 259
  },
  "merges": [["left-piece", "right-piece"]],
  "bos_token": "<|bos|>",
  "eos_token": null,
  "unk_token": null
}
```

The JSON above is an abbreviated illustration, not a complete loadable
vocabulary. A real vocabulary must contain contiguous integer IDs starting at
zero. The byte-level alphabet must be represented by the vocabulary for
lossless encoding without an unknown token. Merge entries are ordered by
priority.

### Weights JSON

The weights file uses format `lmlm.cpu.transformer.v1`. It contains a
`config` object with `vocab_size`, `hidden_size`, `num_heads`,
`num_layers`, `intermediate_size`, `max_seq_len`, and optional
`eos_token_id`.

The `weights` object contains:

- `token_embedding[vocab_size][hidden_size]`
- `position_embedding[max_seq_len][hidden_size]`
- `final_norm_scale[hidden_size]` and `final_norm_bias[hidden_size]`
- `layers[num_layers]`, each containing layer norm vectors, Q/K/V/O
  projection matrices and biases, and up/down feed-forward matrices and biases.

Matrices are row-major JSON arrays of finite float values. Output projection
weights are tied to token embeddings. See the test fixture in
`tests/test_native_cpu_backend.py` for a complete minimal valid model.

## Run inference

```python
from lmlm_core import (
    GenerationRequest,
    LMLMEngine,
    ModelManifest,
    NativeCPUBackend,
)

manifest = ModelManifest.from_dict({
    "format_version": 1,
    "model_id": "lmlm-tiny-v1",
    "architecture": "lmlm-cpu-transformer-v1",
    "tokenizer": "tokenizer.json",
    "weights": "weights.json",
    "tensor_dtype": "f32",
    "context_length": 128,
    "vocabulary_size": 260,
})

engine = LMLMEngine(NativeCPUBackend(seed=7))
engine.load(manifest, "./model")
result = engine.generate(
    GenerationRequest("Hello from LMLM", max_new_tokens=32, temperature=0.7, top_p=0.9)
)
print(result.text)
engine.close()
```

## Current constraints

- Pure Python CPU math; no NumPy, BLAS, GPU, or accelerator dependency.
- Reference backend caps model dimensions, total parameters (5 million),
  estimated inference work, prompt input (at most 4096 UTF-8 bytes and the
  model context limit), and weights files (64 MiB).
- JSON decimal weights are range-checked and rounded to IEEE-754 float32 on
  load; arithmetic then uses Python floats. This is a portable interchange
  representation, not a compact or fast production checkpoint format.
- No KV cache: each token recomputes the active context.
- No quantization, optimized kernels, distributed execution, or streaming.
- The architecture is a small GPT-like decoder with learned position
  embeddings; it is not yet the complete multimodal LMLM architecture.
- Image, audio, video, and mixed-modality inference are not implemented here.

## Tests

```bash
python -m unittest discover -s tests -v
```
