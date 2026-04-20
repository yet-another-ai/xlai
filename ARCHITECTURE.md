# XLAI workspace architecture

This document summarizes crate boundaries and data flow. The **stable public contract** for domain types and traits is `xlai-core`; other crates adapt or orchestrate around it.

Workspace crates are grouped under `crates/core/`, `crates/runtime/`, `crates/backends/`, `crates/qts/`, `crates/sys/`, and `crates/platform/` (see [docs/development/crates-taxonomy.md](docs/development/crates-taxonomy.md)).

## Crate roles

| Crate | Responsibility |
|--------|------------------|
| `xlai-core` | Shared types (`ChatRequest`, `ToolCall`, …), `XlaiError`, and async traits (`ChatModel`, `TtsModel`, …). |
| `xlai-runtime` | `RuntimeBuilder` / `XlaiRuntime`, `Chat` and `Agent` sessions, tool execution, streaming, embedded prompts/skills. Includes `xlai_runtime::local_common` (prompt templates, tool JSON envelope, `PreparedLocalChatRequest`) for `xlai-backend-llama-cpp` / `xlai-backend-transformersjs`. |
| `xlai-backend-openai` | OpenAI-compatible HTTP client (chat, transcription, TTS). |
| `xlai-backend-gemini` | Google Gemini HTTP client. |
| `xlai-backend-llama-cpp` | Local GGUF inference via llama.cpp; maps `xlai-core` requests into local prompt/tool formats. |
| `xlai-backend-transformersjs` | Browser-side chat via a JS adapter (WASM only). |
| `xlai-qts-manifest` | Browser QTS manifest / capability serde types (no GGML/ORT). Re-exported from `xlai_qts_core::browser`; `xlai-wasm` (feature `qts`) depends on it directly. |
| `xlai-qts-core` | Qwen3 TTS engine (GGUF + GGML talker, ONNX vocoder, optional ONNX reference-codec encoder for ICL `ref_code`, tokenizer, streaming) **and** native `TtsModel` bridge (`QtsTtsModel`): maps `TtsRequest` to the engine; `VoiceSpec::Clone` (inline WAV) builds native `VoiceClonePromptV2` (x-vector / ICL). |
| `scripts/qts` (root `pyproject.toml`) | Python export: `uv run export-model-artifacts`, `uv run xlai-qts-hf-release` — see `docs/qts/export-and-hf-publish.md`. |
| `xlai-build-native` | Internal: shared CMake / OpenBLAS / Vulkan helpers for native `build.rs` scripts (`xlai-sys-*`). |
| `xlai-sys-llama` | Vendored `llama.cpp` build (CMake + bindgen) for the llama backend. Sources: `vendor/native/llama.cpp`. |
| `xlai-sys-ggml` | Vendored standalone `ggml` build (CMake + bindgen) for QTS. Sources: `vendor/native/ggml`. |
| `xlai-qts-cli` | `synthesize` / `profile` / `tui` binary (`xlai-qts`) for local TTS workflows. |
| `xlai-facade` | Internal (not on crates.io): native-only integration layer used by **`xlai-native`** (re-exports, default local `llama.cpp` wiring plus optional `qts`, Gemini + OpenAI + transformers.js for the aggregate). **`xlai-wasm` does not depend on it**; WASM re-exports come from `xlai-core`, `xlai-runtime`, and backends directly. |
| `xlai-native` | Native Rust entrypoint: **explicit** public re-exports (plus `gemini` submodule for workspace-only Gemini types). Uses `xlai-facade` for feature wiring. Local `llama.cpp` is included by default; enable optional `qts` for `QtsTtsModel` (avoids linking QTS/ggml unless needed). |
| `xlai-wasm` | `wasm-bindgen` entry points and JS-facing session factories. Public Rust types are re-exported from `xlai-core` / `xlai-runtime` / OpenAI + (on wasm32) transformers.js — no `xlai-facade` dependency. Default feature `qts` enables local QTS WASM surface (stub `TtsModel`, shared browser manifest types, `qtsBrowserTts*`; see `docs/qts/wasm-browser-runtime.md`). |
| `xlai-ffi` | C ABI facade for future native interop. |

## Request flow (chat)

1. Application uses `xlai-native` or `xlai-wasm` to build a `RuntimeBuilder` with a `ChatBackend`.
2. `Chat` / `Agent` in `xlai-runtime` maintains history and tool registry, then calls `ChatModel::generate` or streaming APIs on the configured backend.
3. Backends translate `xlai_core::ChatRequest` into provider-specific payloads and map responses (and errors) back to `xlai_core` types.

## Runtime execution hints (game / interactive workloads)

- **`xlai-core`** defines advisory `ChatExecutionOverrides` / `ChatExecutionConfig`, `TtsExecutionOverrides` / `TtsExecutionConfig`, and a shared `CancellationSignal`. `ChatRequest` / `TtsRequest` carry optional `execution` plus in-process-only `cancellation` (not serialized on the wire).
- **Merge order** for chat: `RuntimeBuilder::with_chat_execution_defaults` → `Chat::with_chat_execution_overrides` / `Agent::with_chat_execution_overrides` → optional per-call layer on `begin_stream` / `stream_with_options`. Session `Some(_)` fields override runtime defaults per field.
- **`xlai-runtime`** merges into every `Chat::build_request`, exposes `ChatExecutionHandle` (`next_event`, `cancel_on_drop`) alongside existing streams, and checks `cancellation` in `XlaiRuntime::stream_chat`. `with_default_max_tool_round_trips` seeds new `Agent` sessions unless overridden with `with_max_tool_round_trips`.
- **Local backends**: `PreparedLocalChatRequest` carries `execution` + `cancellation` for llama.cpp / transformers.js prep. **llama.cpp** honors cooperative cancel in the token loop, optional `max_tokens_per_second` pacing, and `ChatModel::warmup()` preloads weights. **QTS** checks cancellation on unary/stream paths and in streaming PCM chunks; `TtsModel::warmup()` ensures the worker is spawned.
- **WASM** session options accept camelCase `chatExecution`, `runtimeChatExecutionDefaults`, `runtimeTtsExecutionDefaults`, `defaultMaxToolRoundTrips`, and `ttsExecution` on TTS calls (see `crates/platform/xlai-wasm/src/types.rs`). Remote OpenAI-style backends ignore unsupported hints.

## Errors

`xlai_core::XlaiError` carries a `kind`, human-readable `message`, and optional structured fields (`http_status`, `request_id`, `provider_code`, `retryable`, `details`) for observability. Backends should attach HTTP context when available (see `xlai-backend-openai::provider_response`).

## Tests

- **Unit tests** live next to code (`src/tests/` submodules where files grew large).
- **Ignored e2e** tests use real credentials or local model paths; see `README.md` and `.github/workflows/e2e.yml`.
