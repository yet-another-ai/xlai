# XLAI

**Documentation site:** the VitePress site lives under [`docs/`](docs/). Build and preview locally with `pnpm docs:dev` and `pnpm docs:build`.

`xlai` is a Rust-first AI integration workspace for building reusable AI calling flows across native applications and the browser.
The name was inspired by [moeru-ai/xsai](https://github.com/moeru-ai/xsai), which focuses on extra-small AI SDK.

The project is designed around a unified API with pluggable backends. Today it includes:

The long-term goal is to support API-based models, tool integration, skill management, knowledge retrieval, vector search, and local/device inference behind the same overall API model.

## Status

This repository is in an early stage and the public API is still evolving.

Already implemented:

- Cargo workspace with native and `wasm32` build support
- OpenAI-compatible chat backend
- chat sessions with per-session tool registration
- streaming chat output
- configurable tool-call execution mode: concurrent by default, sequential optional
- unit/mock test lane and protected e2e test lane
- GitHub Actions for build, test, formatting, clippy, and e2e

## Workspace Layout

```text
xlai/
├── vendor/
│   └── native/          # git submodules: llama.cpp, ggml (for xlai-sys-*)
├── crates/
│   ├── core/
│   │   └── xlai-core/
│   ├── runtime/
│   │   └── xlai-runtime/
│   ├── backends/
│   │   ├── xlai-backend-openai/
│   │   ├── xlai-backend-transformersjs/
│   │   ├── xlai-backend-gemini/
│   │   └── xlai-backend-llama-cpp/
│   ├── qts/
│   │   ├── xlai-qts-core/
│   │   ├── xlai-qts-manifest/
│   │   └── xlai-qts-cli/
│   ├── sys/
│   │   ├── xlai-build-native/
│   │   ├── xlai-sys-llama/
│   │   └── xlai-sys-ggml/
│   └── platform/
│       ├── xlai-facade/
│       ├── xlai-native/
│       ├── xlai-wasm/
│       └── xlai-ffi/
├── packages/
│   └── xlai/
└── .github/workflows/
```

For crate boundaries and request flow, see [ARCHITECTURE.md](ARCHITECTURE.md). For publish vs internal crates, see [docs/development/crates-taxonomy.md](docs/development/crates-taxonomy.md).

### Crates And Packages

- `crates/core/xlai-core`
  Shared domain types and traits for chat, tools, embeddings, knowledge, and vector search.
- `crates/platform/xlai-ffi`
  Native C ABI facade crate for future FFI integrations.
- `crates/runtime/xlai-runtime`
  Runtime builder, chat session API, streaming, and tool-calling orchestration. Local-backend prompt prep (`PreparedLocalChatRequest`, tool JSON, templates) lives in **`xlai_runtime::local_common`** (used by llama.cpp and transformers.js backends).
- `crates/platform/xlai-facade`
  Internal native aggregate wiring and re-exports for `xlai-native` only (not used by `xlai-wasm`). Not published to crates.io.
- `crates/platform/xlai-native`
  Native Rust-facing entrypoint: explicit re-exports, optional `qts`, and `gemini` submodule for workspace-only Gemini types.
- `crates/platform/xlai-wasm`
  Browser-facing `wasm-bindgen` crate for web integration. Default Cargo feature `qts` enables local QTS entrypoints (`qtsBrowserTts`, manifest validation); the in-browser engine is still a stub until GGML/ORT WASM work lands (see `docs/qts/wasm-browser-runtime.md`).
- `crates/backends/xlai-backend-llama-cpp`
  Native `llama.cpp` chat backend for local GGUF inference.
- `crates/sys/xlai-build-native`
  Shared `build.rs` helpers for native CMake/OpenBLAS/Vulkan and llama.cpp CMake patches (build-dependency only).
- `crates/sys/xlai-sys-llama`
  Vendored `llama.cpp` native stack for the local chat backend.
- `crates/sys/xlai-sys-ggml`
  Vendored standalone `ggml` native stack for QTS.
- `packages/xlai`
  Vite-based TypeScript package published as `@yai-xlai/xlai`, built on top of `xlai-wasm`, with Vitest coverage.
- `crates/backends/xlai-backend-openai`
  OpenAI-compatible backend implementation using `reqwest`.
- `crates/backends/xlai-backend-transformersjs`
  Browser chat backend that delegates generation to a JavaScript adapter (WASM).
- `crates/backends/xlai-backend-gemini`
  Google Gemini HTTP backend (workspace-only; `publish = false`).
- `crates/qts/xlai-qts-manifest`
  Serde types for browser QTS model manifests and capability JSON (no GGML/ORT); used by `xlai-wasm` (feature `qts`) and re-exported as `xlai_qts_core::browser`.
- `crates/qts/xlai-qts-core`
  Qwen3 TTS engine; links standalone `ggml` through `xlai-sys-ggml`. Exposes native `TtsModel` (`QtsTtsModel`, WAV output; tuning via `TtsRequest` metadata `xlai.qts.*`). **`VoiceSpec::Clone`** uses the first reference sample (inline WAV only): x-vector and ICL prompts, with optional `xlai.qts.voice_clone_mode` (`xvector` \| `icl`). ICL needs `qwen3-tts-reference-codec.onnx` + preprocess JSON from `uv run export-model-artifacts` (see `docs/qts/export-and-hf-publish.md`). Pipelined vocoder chunking and overlap are documented in `docs/qts/vocoder-streaming.md`. `xlai_qts_core::browser` re-exports `xlai-qts-manifest`.
- `crates/qts/xlai-qts-cli`
  Binary `xlai-qts`: `synthesize`, `profile`, and interactive `tui`. Without voice-clone flags, `synthesize` uses `xlai-runtime` + `xlai-qts-core` (`QtsTtsModel`). With `--voice-clone-prompt` or `--ref-audio`, it uses the direct engine path. Run `cargo run -p xlai-qts-cli -- --help` (or `… synthesize --help`) for flags.

## Requirements

- Rust stable
- minimum supported Rust version: `1.94`

The workspace uses:

- edition `2024`
- Apache 2.0 license

## Quick Start

### Build

```bash
cargo build --workspace
```

### Test

```bash
cargo test --workspace
```

### Check `wasm32`

```bash
rustup target add wasm32-unknown-unknown
cargo check -p xlai-wasm --target wasm32-unknown-unknown --features qts
```

### Install JavaScript dependencies

```bash
pnpm install
```

### Build `@yai-xlai/xlai`

```bash
pnpm --filter @yai-xlai/xlai build
```

### Test `@yai-xlai/xlai`

```bash
pnpm --filter @yai-xlai/xlai test
```

### Run clippy

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

### Check formatting

```bash
cargo fmt --all -- --check
```

## Local Configuration

For local end-to-end testing, copy the template and fill in your values:

```bash
cp .env.example .env
```

Current variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `OPENAI_TRANSCRIPTION_MODEL`

Local `.env` files are ignored by Git.

## Example

This is the current native Rust usage style:

```rust
use xlai_native::core::{
    ToolCallExecutionMode, ToolDefinition, ToolParameter, ToolParameterType, ToolResult,
};
use xlai_native::{OpenAiConfig, RuntimeBuilder};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = RuntimeBuilder::new()
        .with_chat_backend(OpenAiConfig::new(
            std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_owned()),
            std::env::var("OPENAI_API_KEY")?,
            std::env::var("OPENAI_MODEL")
                .unwrap_or_else(|_| "gpt-4.1-mini".to_owned()),
        ))
        .build()?;

    let mut chat = runtime
        .chat_session()
        .with_system_prompt("Be concise.");

    chat.register_tool(
        ToolDefinition {
            name: "lookup_weather".into(),
            description: "Lookup current weather".into(),
            parameters: vec![ToolParameter {
                name: "city".into(),
                description: "City name".into(),
                kind: ToolParameterType::String,
                required: true,
            }],
            execution_mode: ToolCallExecutionMode::Concurrent,
        },
        |arguments| async move {
            let city = arguments["city"].as_str().unwrap_or("unknown");
            Ok(ToolResult {
                tool_name: "lookup_weather".into(),
                content: format!("weather for {city}: sunny"),
                is_error: false,
                metadata: Default::default(),
            })
        },
    );

    let response = chat.prompt("What's the weather in Paris?").await?;
    println!("{}", response.message.content);

    Ok(())
}
```

Agent sessions are available through the same runtime:

```rust
use futures_util::StreamExt;
use xlai_native::core::{
    ChatChunk, ToolCallExecutionMode, ToolDefinition, ToolParameter, ToolParameterType, ToolResult,
};
use xlai_native::{ChatExecutionEvent, OpenAiConfig, RuntimeBuilder};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = RuntimeBuilder::new()
        .with_chat_backend(OpenAiConfig::new(
            std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_owned()),
            std::env::var("OPENAI_API_KEY")?,
            std::env::var("OPENAI_MODEL")
                .unwrap_or_else(|_| "gpt-4.1-mini".to_owned()),
        ))
        .build()?;

    let mut agent = runtime
        .agent_session()?
        .with_system_prompt("Use tools when helpful.");
    // Streaming tool loop:
    // - `.with_max_tool_round_trips(n)` — cap tool round-trips (default 8).
    // - `.with_context_compressor(|msgs, est| async move { Ok(msgs) })` — optional Rust-only hook
    //   before each streamed model call (see README “Context compression hook”).

    agent.register_tool(
        ToolDefinition {
            name: "lookup_weather".into(),
            description: "Lookup current weather".into(),
            parameters: vec![ToolParameter {
                name: "city".into(),
                description: "City name".into(),
                kind: ToolParameterType::String,
                required: true,
            }],
            execution_mode: ToolCallExecutionMode::Concurrent,
        },
        |arguments| async move {
            let city = arguments["city"].as_str().unwrap_or("unknown");
            Ok(ToolResult {
                tool_name: "lookup_weather".into(),
                content: format!("weather for {city}: sunny"),
                is_error: false,
                metadata: Default::default(),
            })
        },
    );

    // Unary `prompt` / `execute` / `prompt_*` = one model call (no tool callbacks).
    // Use `stream_prompt` / `stream` / `stream_*` for the multi-round tool loop.
    let mut stream = agent.stream_prompt("What's the weather in Paris?");
    let mut response = None;
    while let Some(item) = stream.next().await {
        let item = item?;
        if let ChatExecutionEvent::Model(ChatChunk::Finished(resp)) = item {
            response = Some(resp);
        }
    }
    let response = response.expect("stream ended without a finished model response");
    println!("{}", response.message.content);

    Ok(())
}
```

Add `futures-util` to your crate’s dependencies (same major line as the workspace) so `StreamExt` is available.

For local native inference, the same builder can be pointed at `llama.cpp`:

```rust
use xlai_native::{LlamaCppConfig, RuntimeBuilder};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = RuntimeBuilder::new()
        .with_chat_backend(
            LlamaCppConfig::new(std::env::var("LLAMA_CPP_MODEL")?)
                .with_context_size(4096)
                .with_max_output_tokens(256),
        )
        .build()?;

    let response = runtime
        .chat_session()
        .with_system_prompt("Be concise.")
        .prompt("Reply with a short greeting.")
        .await?;

    println!("{}", response.message.content);
    Ok(())
}
```

## Tool Calling

`Chat` sessions can register tools directly with async callbacks. `Agent` sessions
also expose an `McpRegistry` via `agent.mcp_registry()` so MCP-provided tools can
be registered separately from built-in agent tools.

The WebAssembly package mirrors this split with `chat(...)`, `createChatSession(...)`,
`agent(...)`, and `createAgentSession(...)`.

Current behavior:

- tools are registered per chat or agent session
- tool calls are exposed to the model through the runtime request
- **`Chat`** performs one model call per `prompt` / `execute` / `stream`; it does not execute tool callbacks or run multiple model rounds—you get the model’s response (including any `tool_calls`) and can drive the next step yourself
- **`Agent`**
  - **Unary** `prompt` / `execute` / `prompt_parts` / `prompt_content`: exactly **one** model call; registered tools are **not** run by the runtime (same “no silent multi-minute loop” guarantee as chat).
  - **Streaming** `stream` / `stream_prompt` / `stream_prompt_content` / `stream_prompt_parts`: runs the automatic **tool loop**. After each assistant turn that finishes with tool calls, tools run and another model turn starts until a response has no tool calls or **`Agent::with_max_tool_round_trips`** (default `8`) is exceeded.
- when tools run (on **`Agent`**), local session tools are used before falling back to a runtime-level tool executor
- each tool’s `ToolDefinition::execution_mode` controls how its calls interact with other calls in the same model turn
- if any invoked tool in a turn is `Sequential`, all tool calls in that turn run sequentially in model order (no overlap)
- otherwise, tool calls in a turn run concurrently

### Tool loop

| Surface          | Control                            | Effect                                              |
| ---------------- | ---------------------------------- | --------------------------------------------------- |
| **Rust `Agent`** | `with_max_tool_round_trips(usize)` | Maximum model↔tool rounds per stream (default `8`). |

`Chat` never runs this loop. Unary agent calls never block on long multi-round tool execution without you choosing a streaming API.

### Context compression hook

On **`Agent`**, **`with_context_compressor`** (Rust) registers an **async** closure that runs **once per streaming tool-loop round**, immediately **before** each model call. It receives the full accumulated `ChatMessage` list for that stream and a **best-effort** `Option<u32>` input-token estimate (JSON serialization of the outgoing `ChatRequest`, bytes÷4 heuristic—not provider-tokenizer-accurate). It must return the message list to send for that round (often a compressed copy). The agent still appends assistant and tool messages to its **internal** history after each round; only the **outgoing** request for that step uses the returned list. Returning an empty list fails the stream with a provider error. **`Agent::register_context_compressor`** is the `&mut self` variant for the same hook.

The hook is **not** used for unary `prompt` / `execute`.

**JavaScript (`@yai-xlai/xlai`):** on **`AgentSession`**, call **`registerContextCompressor(async (messages, estimatedInputTokens) => messages)`** before **`streamPrompt`** / **`streamPromptWithContent`**. The callback receives the same semantics as Rust (message array in Rust JSON shape, `estimatedInputTokens` as `number | null`). WASM exports **`registerContextCompressor`**, **`streamPrompt`**, and **`streamPromptWithContent`** on **`AgentSession`**.

## Streaming

The runtime supports streamed chat output through `ChatChunk` and `ChatExecutionEvent`.

- **`Chat`**: each `stream` / `stream_prompt` / `stream_*` call performs **one** model run; events are deltas and a final `ChatChunk::Finished` for that turn.
- **`Agent`**: streaming uses the same chunk types, but the stream may include multiple model rounds while tools are still being requested. Between rounds you may see **`ChatExecutionEvent::ToolCall`** and **`ToolResult`** events after a finished assistant message that contained tool calls. An optional context-compressor hook may rewrite the message list before each of those model calls (Rust **`with_context_compressor`**, or JS **`AgentSession.registerContextCompressor`**). In JS, **`streamPrompt`** / **`streamPromptWithContent`** collect the full event list in order (one round-trip through the WASM bridge; not a browser `ReadableStream` yet).

Streaming currently includes:

- message start events
- content delta events
- tool call delta events
- final response events (per model turn; agent streams may emit several before the stream ends)

## Testing Model

The repository uses two test classes.

### 1. Unit and mock tests

These are the default tests and do not require API tokens.

Run locally with:

```bash
cargo test --workspace
```

### 2. End-to-end tests

These use real provider credentials and are ignored by default.

Run locally with:

```bash
cargo test --workspace -- --ignored --test-threads=1
```

The current OpenAI smoke tests will also load `.env` automatically for local runs.

For ASR/transcription end-to-end coverage, set `OPENAI_TRANSCRIPTION_MODEL` to a
transcription-capable model such as `gpt-4o-mini-transcribe`.

## CI

### Build workflow

`.github/workflows/build.yml` builds:

- Linux
- Windows
- macOS arm64
- macOS x86_64 via cross-target build
- `wasm32-unknown-unknown`
- the `@yai-xlai/xlai` package bundle through the `pnpm` workspace

### Test workflow

`.github/workflows/test.yml` runs:

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `pnpm --filter @yai-xlai/xlai test`

### Publish workflow

`.github/workflows/publish.yml` runs **crates.io** and **npm** checks:

- On **pull requests** and pushes to **`main`**: `cargo publish --dry-run` for the [publishable Rust crate subset](docs/development/publishing.md) (in dependency order) and `npm publish --dry-run` for `@yai-xlai/xlai` after a full package build.
- On pushes to tags matching **`v*`**: real `cargo publish` and `npm publish` using secrets on the GitHub **`publish`** environment (`CRATES_IO_TOKEN`, `NPM_TOKEN`).

See [`docs/development/publishing.md`](docs/development/publishing.md) for ordering, version bumps, and crates that are not on crates.io yet.

### E2E workflow

`.github/workflows/e2e.yml` runs ignored tests with provider credentials (and skips QTS model-dir tests until a CI fixture download exists).

It is intended to use a protected GitHub Environment such as `e2e`, with maintainer approval and environment secrets.

The OpenAI e2e environment currently expects:

- `OPENAI_API_KEY` as a GitHub secret
- `OPENAI_BASE_URL` as a GitHub environment variable
- `OPENAI_MODEL` as a GitHub environment variable
- `OPENAI_TRANSCRIPTION_MODEL` as a GitHub environment variable

The ignored native `llama.cpp` smoke test expects:

- `LLAMA_CPP_MODEL` pointing to a local GGUF model file, or the default fixture path `fixtures/llama.cpp/Qwen3.5-0.8B-Q4_0.gguf`

The repository intentionally ignores downloaded GGUF fixtures under `fixtures/llama.cpp/`.
CI downloads that fixture with the Hugging Face CLI and caches it between runs.

Ignored QTS integration tests in `xlai-qts-core` expect a full Qwen3 TTS model directory:

- `XLAI_QTS_MODEL_DIR` pointing at a folder containing the GGUF talker, ONNX vocoder (`qwen3-tts-vocoder.onnx`), tokenizer, and `config.json` (see `xlai_qts_core::ModelPaths`). For **ICL** voice clone, also add `qwen3-tts-reference-codec.onnx` and `qwen3-tts-reference-codec-psreprocess.json` (export with `uv run export-model-artifacts`; see `docs/qts/export-and-hf-publish.md`).

CI e2e currently **skips** those tests because no Hugging Face download step is wired yet; run them locally after downloading weights.

To build `xlai-native` with the QTS backend available, enable `--features qts` (optional; avoids linking QTS/ggml in default builds).

## Current Design Notes

- unified API across native and browser targets
- target-specific dependency wiring for native vs `wasm32`
- OpenAI-compatible backend uses `reqwest`
- public APIs aim to be async-first and streaming-capable

## Roadmap

Planned or expected next areas include:

- more provider backends
- typed tool argument helpers
- embeddings and retrieval backends
- skill management APIs
- local/device inference backends
- more browser-focused examples

## Support of LLM API Providers

### Chat API

- [x] OpenAI Backends
  - [x] [AIHubMix](https://aihubmix.com/?aff=OOiX)
  - [x] [OpenAI](https://platform.openai.com/docs/guides/gpt/chat-completions-api)
  - [x] [Azure OpenAI API](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference)
- [x] OpenRouter Backend
- [x] llama.cpp Backends
  - [x] CPU
  - [x] OpenBLAS
  - [x] Accelerate.framework
  - [x] Metal
  - [x] Vulkan
  - [ ] CUDA
  - [ ] HIP
  - [ ] OpenVINO

### ASR API

- [x] [OpenAI](https://developers.openai.com/api/docs/guides/speech-to-text)

### TTS API

- [x] [OpenAI](https://developers.openai.com/api/docs/guides/text-to-speech)
- [ ] QTS
  - Transformer
    - [x] CPU
    - [x] OpenBLAS
    - [x] Accelerate.framework
    - [x] Metal
    - [x] Vulkan
    - [ ] CUDA
    - [ ] HIP
    - [ ] OpenVINO
  - Vocoder (ORT default EPs)

## License

Apache License 2.0. See `LICENSE`.
