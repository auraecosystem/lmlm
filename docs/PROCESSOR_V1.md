# LMLM Processor v1

LMLM now exposes a versioned, loopback-only processor service that can be consumed by LMLMxtend. The protocol is intentionally independent of any cloud provider.

## Architecture

```text
LMLM local model process (for example, a locally loaded GGUF model)
                 ^
                 | localhost OpenAI-compatible inference API
                 |
LMLM Processor v1 (Python, loopback only :8765)
                 ^
                 | HTTP + JSON, protocol lmlm.processor.v1
                 |
LMLMxtend HttpLmlmProcessor adapter
                 ^
                 |
LMLMxtend Extension Runtime / agents / applications
```

The service supports two local modes. The compatibility mode talks to an already-running loopback OpenAI-compatible inference endpoint. The native mode loads the LMLM tokenizer and CPU transformer directly from a local model directory. Neither mode downloads model weights, calls remote providers, or falls back to cloud inference.

## Start a local inference process

The repository's `Modelstore/llama.cpp` notes show a local llama-server workflow. Start that local server with the desired local model, bound to loopback, and expose its OpenAI-compatible API on port 8080. Do not bind it to a public interface.

## Start the LMLM processor service

From the repository root:

```bash
LMLM_INFERENCE_URL=http://127.0.0.1:8080/v1 \\
LMLM_PROCESSOR_HOST=127.0.0.1 \\
LMLM_PROCESSOR_PORT=8765 \\
python -m lmlm_processor.server
```

The processor service uses only Python's standard library. It expects the local inference server to implement `GET /health` and `POST /v1/chat/completions`.

## Protocol

- Protocol identifier: `lmlm.processor.v1`
- Protocol version: `1`
- Health: `GET /v1/health`
- Process: `POST /v1/process`
- Request fields: `request_id`, `modality`, `content_type`, `payload_hex`, optional `prompt` and `parameters`
- Response fields: `request_id`, `content_type`, `payload_hex`

Payload bytes use hexadecimal encoding to keep the wire format dependency-free and lossless. Both current execution modes support text only. Image, audio, video, and mixed modalities are represented by the protocol but rejected until their LMLM-native preprocessing and inference backends are implemented.

## Limits and security

- Service binds only to loopback addresses.
- In compatibility mode, the inference URL must be plain HTTP on loopback; TLS and remote endpoints are intentionally not enabled.
- Request body is limited to 16 MiB; decoded payload is limited to 8 MiB.
- Text generation is capped at 4096 output tokens and temperature 0–2.
- This service has no authentication because it is loopback-only. Do not expose it through a tunnel or public reverse proxy.
- Native mode uses the in-process reference CPU transformer and local model assets; see its documented 5-million-parameter and 64 MiB weight-file limits. Compatibility mode still requires a separately installed local inference runtime.

## Tests

```bash
python -m unittest discover -s tests -v
```
