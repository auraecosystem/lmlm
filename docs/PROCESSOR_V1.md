# LMLM Processor v1

LMLM exposes a versioned, loopback-only processor service consumed by
LMLMxtend. The protocol is independent of cloud providers.

## Architecture

```text
LMLM native CPU model assets
       |
       v
NativeCPUBackend -> LMLMEngine
       |                 |
       +------ Processor v1 service (127.0.0.1:8765)
                          |
                   HTTP + JSON
                 lmlm.processor.v1
                          |
             LMLMxtend HttpLmlmProcessor
                          |
            Extensions / agents / applications
```

The service supports two local modes:

- `native_cpu`: loads LMLM's tokenizer and CPU transformer directly from a
  local model directory. No separate inference server is required.
- `local_openai`: compatibility mode for an already-running local
  OpenAI-compatible inference endpoint.

Neither mode downloads model weights, calls remote providers, or falls back
to cloud inference.

## Native CPU mode (recommended)

Prepare a local model directory containing `lmlm-model.json`,
`tokenizer.json`, and `weights.json` using the format documented in
[Native CPU Inference v1](NATIVE_CPU_INFERENCE_V1.md).

From the repository root:

```bash
LMLM_BACKEND=native_cpu \
LMLM_MODEL_DIR=./model \
LMLM_PROCESSOR_HOST=127.0.0.1 \
LMLM_PROCESSOR_PORT=8765 \
python -m lmlm_processor.server
```

The service loads the model once at startup and serves generation requests
through the in-process native CPU engine.

## Compatibility mode

The existing local inference adapter remains available:

```bash
LMLM_BACKEND=local_openai \
LMLM_INFERENCE_URL=http://127.0.0.1:8080/v1 \
LMLM_PROCESSOR_HOST=127.0.0.1 \
LMLM_PROCESSOR_PORT=8765 \
python -m lmlm_processor.server
```

The local inference endpoint must be bound to loopback and implement
`GET /health` and `POST /v1/chat/completions`.

## Protocol

- Protocol identifier: `lmlm.processor.v1`
- Protocol version: `1`
- Health: `GET /v1/health`
- Process: `POST /v1/process`
- Request fields: `request_id`, `modality`, `content_type`,
  `payload_hex`, optional `prompt` and `parameters`
- Response fields: `request_id`, `content_type`, `payload_hex`

Payload bytes use hexadecimal encoding for lossless, dependency-free
transport. Both current execution modes support text only. Image, audio,
video, and mixed modalities are represented by the protocol but rejected
until their LMLM-native preprocessing and inference backends are implemented.

## Limits and security

- Service binds only to `127.0.0.1`.
- Compatibility mode accepts only plain HTTP loopback inference URLs.
- Request body is limited to 18 MiB; decoded payload is limited to 8 MiB.
- The native reference CPU engine limits model size to 5 million parameters
  and weights files to 64 MiB.
- Text generation is capped at 4096 output tokens by the compatibility
  backend; native mode applies the engine's 32768-token request ceiling.
- No authentication is configured because the service is loopback-only.
  Do not expose it through a tunnel or public reverse proxy.
- Model assets must be installed locally by the operator.

## Tests

```bash
python -m unittest discover -s tests -v
```
