# LMLM Runtime v2: NumPy CPU and Tiny Training

Runtime v2 adds an optional vectorized CPU backend and a reproducible local
training utility. The dependency-free `NativeCPUBackend` remains the reference
implementation and fallback.

## Install the optional tensor dependency

```bash
python -m pip install "numpy>=1.26"
```

## Select the optimized backend

The backend can be used directly through the engine contract:

```python
from lmlm_core import LMLMEngine, ModelManifest, GenerationRequest
from lmlm_core.numpy_backend import NumpyCPUBackend

manifest = ModelManifest.from_dict(...)
engine = LMLMEngine(NumpyCPUBackend())
engine.load(manifest, "./model")
result = engine.generate(GenerationRequest("Hello", max_new_tokens=32))
print(result.text)
engine.close()
```

The optimized backend uses the same local manifest, tokenizer, weights,
generation contract, and validation rules as the reference backend. It does
not download assets or call remote services. NumPy is imported lazily.

## Train a tiny local model

Prepare a UTF-8 text corpus and run:

```bash
python -m pip install "numpy>=1.26"
python -m lmlm_training.train \
  --input ./corpus.txt \
  --output ./model \
  --steps 500 \
  --batch-size 64 \
  --hidden-size 32 \
  --seed 7
```

The output directory contains:

- `lmlm-model.json`: manifest consumed by the native Processor
- `tokenizer.json`: deterministic byte-level vocabulary
- `weights.json`: weights in `lmlm.cpu.transformer.v1` format
- `checkpoint.npz`: embeddings, Adam optimizer state, and step counter
- `training.json`: corpus hash, seed, configuration, and sampled losses

Resume training with `--resume`; it loads the checkpoint in the output
directory and continues the optimizer state.

## Scope and limitations

This first trainer is intentionally a small byte-level next-token trainer. It
learns tied token embeddings and emits a structurally compatible one-layer
reference model. It is useful for testing the end-to-end local training,
checkpoint, tokenizer, loader, and inference pipeline. It is **not** a full
transformer pretraining implementation, instruction-tuned model, multimodal
model, or production-quality language model.

The NumPy backend vectorizes sequence attention and dense layers on CPU. The
reference backend remains available for environments without NumPy. No GPU,
quantized kernel, KV cache, or multimodal encoder is claimed in this release.
