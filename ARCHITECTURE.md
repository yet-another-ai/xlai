# XLAI workspace architecture

This document summarizes crate boundaries and data flow. The **stable public contract** for domain types and traits is `xlai-core`; other crates adapt or orchestrate around it.

## Crate roles

| Crate | Responsibility |
|--------|------------------|
| `xlai-core` | Shared types (`ChatRequest`, `ToolCall`, …), `XlaiError`, and async traits (`ChatModel`, `TtsModel`, …). |
| `xlai-runtime` | `RuntimeBuilder` / `XlaiRuntime`, `Chat` and `Agent` sessions, tool execution, streaming, embedded prompts/skills. |
| `xlai-backend-openai` | OpenAI-compatible HTTP client (chat, transcription, TTS). |
| `xlai-backend-llama-cpp` | Local GGUF inference via llama.cpp; maps `xlai-core` requests into local prompt/tool formats. |
| `xlai-backend-transformersjs` | Browser-side chat via a JS adapter (WASM only). |
| `xlai-backend-qts` | Native Qwen3 TTS: maps `TtsRequest` to `xlai-qts-core` (`TtsModel`); `VoiceSpec::Clone` (inline WAV) builds native `VoiceClonePromptV2` (x-vector / ICL). |
| `xlai-backend-qts-wasm` | Browser `TtsModel` for QTS (`qts-browser` provider): stub until GGML/ORT WASM engines land; WASM exports `qtsBrowserTts*`, manifest validation. |
| `xlai-qts-core` | Qwen3 TTS engine (GGUF + GGML talker, ONNX vocoder, optional ONNX reference-codec encoder for ICL `ref_code`, tokenizer, streaming). |
| `xlai-qts-browser` | QTS browser manifest and capability types (serde, no `xlai-sys` / `ort`). Optional `browser-manifest` feature on `xlai-qts-core` re-exports it as `xlai_qts_core::browser`. |
| `scripts/qts` (root `pyproject.toml`) | Python export: `uv run export-model-artifacts`, `uv run xlai-qts-hf-release` — see `docs/qts-export-and-hf-publish.md`. |
| `xlai-sys` | Vendored `llama.cpp` and/or standalone `ggml` (CMake + bindgen); enable `llama` and/or `qts-ggml`. |
| `xlai-qts-cli` | `synthesize` / `profile` / `tui` binary (`xlai-qts`) for local TTS workflows. |
| `xlai-local-common` | **Internal** helpers shared by local inference backends (prompt shapes, tool JSON schema). Not a public SDK surface; keep mappings thin and tested per backend. |
| `xlai-native` | Native Rust facade: re-exports runtime, backends, and core for applications. Enable optional `qts` feature for `QtsTtsModel` re-exports (avoids linking QTS/ggml unless needed). |
| `xlai-wasm` | `wasm-bindgen` entry points and JS-facing session factories. Default feature `qts` enables local QTS WASM surface (stub engine; see `docs/qts-wasm-browser-runtime.md`). |
| `xlai-ffi` | C ABI facade for future native interop. |

## Request flow (chat)

1. Application uses `xlai-native` or `xlai-wasm` to build a `RuntimeBuilder` with a `ChatBackend`.
2. `Chat` / `Agent` in `xlai-runtime` maintains history and tool registry, then calls `ChatModel::generate` or streaming APIs on the configured backend.
3. Backends translate `xlai_core::ChatRequest` into provider-specific payloads and map responses (and errors) back to `xlai_core` types.

## Errors

`xlai_core::XlaiError` carries a `kind`, human-readable `message`, and optional structured fields (`http_status`, `request_id`, `provider_code`, `retryable`, `details`) for observability. Backends should attach HTTP context when available (see `xlai-backend-openai::provider_response`).

## Tests

- **Unit tests** live next to code (`src/tests/` submodules where files grew large).
- **Ignored e2e** tests use real credentials or local model paths; see `README.md` and `.github/workflows/e2e.yml`.
