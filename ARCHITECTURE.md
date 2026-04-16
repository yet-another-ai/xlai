# XLAI workspace architecture

This document summarizes crate boundaries and data flow. The **stable public contract** for domain types and traits is `xlai-core`; other crates adapt or orchestrate around it.

## Crate roles

| Crate | Responsibility |
|--------|------------------|
| `xlai-core` | Shared types (`ChatRequest`, `ToolCall`, …), `XlaiError`, and async traits (`ChatModel`, `TtsModel`, …). |
| `xlai-local-common` | Local (non-API) chat prep: prompt templates, tool JSON envelope, `PreparedLocalChatRequest`. Used by `xlai-backend-llama-cpp` / `xlai-backend-transformersjs`; re-exported as `xlai_runtime::local_common` for compatibility. |
| `xlai-runtime` | `RuntimeBuilder` / `XlaiRuntime`, `Chat` and `Agent` sessions, tool execution, streaming, embedded prompts/skills. |
| `xlai-backend-openai` | OpenAI-compatible HTTP client (chat, transcription, TTS). |
| `xlai-backend-gemini` | Google Gemini HTTP client. |
| `xlai-backend-llama-cpp` | Local GGUF inference via llama.cpp; maps `xlai-core` requests into local prompt/tool formats. |
| `xlai-backend-transformersjs` | Browser-side chat via a JS adapter (WASM only). |
| `xlai-qts-manifest` | Browser QTS manifest / capability serde types (no GGML/ORT). Re-exported from `xlai_qts_core::browser`; `xlai-wasm` (feature `qts`) depends on it directly. |
| `xlai-qts-core` | Qwen3 TTS engine (GGUF + GGML talker, ONNX vocoder, optional ONNX reference-codec encoder for ICL `ref_code`, tokenizer, streaming) **and** native `TtsModel` bridge (`QtsTtsModel`): maps `TtsRequest` to the engine; `VoiceSpec::Clone` (inline WAV) builds native `VoiceClonePromptV2` (x-vector / ICL). |
| `scripts/qts` (root `pyproject.toml`) | Python export: `uv run export-model-artifacts`, `uv run xlai-qts-hf-release` — see `docs/qts/export-and-hf-publish.md`. |
| `xlai-sys-llama` | Vendored `llama.cpp` build (CMake + bindgen) for the llama backend. |
| `xlai-sys-ggml` | Vendored standalone `ggml` build (CMake + bindgen) for QTS. |
| `xlai-qts-cli` | `synthesize` / `profile` / `tui` binary (`xlai-qts`) for local TTS workflows. |
| `xlai-facade` | Internal: shared re-exports of `xlai-core`, `xlai-runtime`, OpenAI + Gemini + transformers.js backends, optional `llama` + `qts` for native. Consumed by `xlai-native` (with `llama` + accelerators) and `xlai-wasm` (no `llama`, no `qts` engine — WASM QTS stays stub + `xlai-qts-manifest` only). |
| `xlai-native` | Native Rust facade: thin re-export of `xlai-facade`. Enable optional `qts` for `QtsTtsModel` (avoids linking QTS/ggml unless needed). |
| `xlai-wasm` | `wasm-bindgen` entry points and JS-facing session factories. Default feature `qts` enables local QTS WASM surface (stub `TtsModel`, shared browser manifest types, `qtsBrowserTts*`; see `docs/qts/wasm-browser-runtime.md`). |
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
