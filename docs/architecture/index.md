# XLAI workspace architecture

This page summarizes **crate boundaries** and **data flow**. The stable public contract for domain types and traits is **`xlai-core`**; other crates adapt or orchestrate around it.

::: tip
The same content lives in the repository as [`ARCHITECTURE.md`](https://github.com/yetanother.ai/xlai/blob/main/ARCHITECTURE.md) for browsing on GitHub without the docs site.
:::

## Crate roles

| Crate                                 | Responsibility                                                                                                                                                                                                                                                                                                             |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `xlai-core`                           | Shared types (`ChatRequest`, `EmbeddingRequest`, `ToolCall`, …), `XlaiError`, and async traits (`ChatModel`, `EmbeddingModel`, `TtsModel`, …).                                                                                                                                                                             |
| `xlai-runtime`                        | `RuntimeBuilder` / `XlaiRuntime`, `Chat` and `Agent` sessions, tool execution, streaming, embedded prompts/skills. Includes `xlai_runtime::local_common` (prompt templates, tool JSON envelope, `PreparedLocalChatRequest`) for `xlai-backend-llama-cpp` / `xlai-backend-transformersjs`.                                  |
| `xlai-backend-openai`                 | OpenAI-compatible HTTP client (chat, embeddings, transcription, TTS).                                                                                                                                                                                                                                                      |
| `xlai-backend-openrouter`             | OpenRouter Responses API chat backend.                                                                                                                                                                                                                                                                                     |
| `xlai-backend-gemini`                 | Google Gemini HTTP backend (chat, embeddings, image generation).                                                                                                                                                                                                                                                           |
| `xlai-backend-llama-cpp`              | Local GGUF inference via llama.cpp; maps `xlai-core` requests into local prompt/tool formats and local embeddings.                                                                                                                                                                                                         |
| `xlai-backend-transformersjs`         | Browser-side chat plus embeddings via a JS adapter (WASM only).                                                                                                                                                                                                                                                            |
| `xlai-qts-manifest`                   | Browser QTS manifest / capability serde types (no GGML/ORT). Re-exported from `xlai_qts_core::browser`; `xlai-wasm` (feature `qts`) depends on it directly.                                                                                                                                                                |
| `xlai-qts-core`                       | Qwen3 TTS engine (GGUF + GGML talker, ONNX vocoder, optional ONNX reference-codec encoder for ICL `ref_code`, tokenizer, streaming) **and** native `TtsModel` bridge (`QtsTtsModel`).                                                                                                                                      |
| `scripts/qts` (root `pyproject.toml`) | Python export: `uv run export-model-artifacts`, `uv run xlai-qts-hf-release` — see [Export and Hugging Face](/qts/export-and-hf-publish).                                                                                                                                                                                  |
| `xlai-build-native`                   | Internal: shared CMake / OpenBLAS / Vulkan helpers for native `build.rs` scripts.                                                                                                                                                                                                                                          |
| `xlai-sys-llama`                      | Vendored `llama.cpp` build (CMake + bindgen) for the llama backend (`vendor/native/llama.cpp`).                                                                                                                                                                                                                            |
| `xlai-sys-ggml`                       | Vendored standalone `ggml` build (CMake + bindgen) for QTS (`vendor/native/ggml`).                                                                                                                                                                                                                                         |
| `xlai-qts-cli`                        | `synthesize` / `profile` / `tui` binary (`xlai-qts`) for local TTS workflows.                                                                                                                                                                                                                                              |
| `xlai-facade`                         | Internal (not on crates.io): **native-only** integration layer for `xlai-native` (optional `llama` / `qts`, Gemini + HTTP/local backends). `xlai-wasm` does not use this crate.                                                                                                                                            |
| `xlai-native`                         | Native Rust entrypoint: explicit re-exports and `gemini` submodule; uses `xlai-facade` for feature wiring. Enable optional `qts` for `QtsTtsModel`.                                                                                                                                                                        |
| `xlai-wasm`                           | `wasm-bindgen` entry points and JS-facing session factories. Re-exports from `xlai-core` / `xlai-runtime` / backends (no `xlai-facade`). Default feature `qts` enables local QTS WASM surface (stub `TtsModel`, shared browser manifest types, `qtsBrowserTts*`; see [Browser / WASM runtime](/qts/wasm-browser-runtime)). |
| `xlai-ffi`                            | C ABI facade for future native interop.                                                                                                                                                                                                                                                                                    |

## Request flow (chat)

1. Application uses `xlai-native` or `xlai-wasm` to build a `RuntimeBuilder` with a `ChatBackend`.
2. `Chat` / `Agent` in `xlai-runtime` maintains history and tool registry, then calls `ChatModel::generate` or streaming APIs on the configured backend.
3. Backends translate `xlai_core::ChatRequest` into provider-specific payloads and map responses (and errors) back to `xlai_core` types.

## Request flow (embeddings)

1. Application configures an embedding backend with `RuntimeBuilder::with_embedding_backend(...)` or `with_embedding_model(...)`.
2. `XlaiRuntime::embed` forwards `xlai_core::EmbeddingRequest` to the configured `EmbeddingModel`.
3. The backend maps provider-specific responses back into ordered `EmbeddingResponse { vectors, usage, metadata }`.

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
- **Ignored e2e** tests use real credentials or local model paths; see [CI and testing](/development/ci-and-testing) and `.github/workflows/e2e.yml` on GitHub.

## Related

- [Crate taxonomy](/development/crates-taxonomy) — published vs internal crates
- [QTS overview](/qts/) — TTS stack and browser direction
