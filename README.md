# XLAI

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
├── crates/
│   ├── xlai-backend-llama-cpp/
│   ├── xlai-backend-openai/
│   ├── xlai-backend-transformersjs/
│   ├── xlai-core/
│   ├── xlai-ffi/
│   ├── xlai-llama-cpp-sys/
│   ├── xlai-local-common/
│   ├── xlai-native/
│   ├── xlai-runtime/
│   └── xlai-wasm/
├── packages/
│   └── xlai/
└── .github/workflows/
```

For crate boundaries and request flow, see [ARCHITECTURE.md](ARCHITECTURE.md).

### Crates And Packages

- `crates/xlai-core`
  Shared domain types and traits for chat, tools, embeddings, knowledge, and vector search.
- `crates/xlai-ffi`
  Native C ABI facade crate for future FFI integrations.
- `crates/xlai-runtime`
  Runtime builder, chat session API, streaming, and tool-calling orchestration.
- `crates/xlai-native`
  Native Rust-facing facade crate that re-exports the runtime API.
- `crates/xlai-wasm`
  Browser-facing `wasm-bindgen` facade crate for web integration.
- `crates/xlai-backend-llama-cpp`
  Native `llama.cpp` chat backend for local GGUF inference.
- `crates/xlai-llama-cpp-sys`
  Vendored `llama.cpp` submodule plus generated/raw FFI bindings and build integration.
- `packages/xlai`
  Vite-based TypeScript package published as `@yai-xlai/xlai`, built on top of `xlai-wasm`, with Vitest coverage.
- `crates/xlai-backend-openai`
  OpenAI-compatible backend implementation using `reqwest`.
- `crates/xlai-backend-transformersjs`
  Browser chat backend that delegates generation to a JavaScript adapter (WASM).
- `crates/xlai-local-common`
  Internal helpers shared by local inference backends (prompts, tool JSON envelope).

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
cargo check -p xlai-wasm --target wasm32-unknown-unknown
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

    let mut agent = runtime.agent_session()?.with_system_prompt("Use tools when helpful.");

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

    let response = agent.prompt("What's the weather in Paris?").await?;
    println!("{}", response.message.content);

    Ok(())
}
```

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

- tools are registered per chat session
- tool calls are exposed to the model through the runtime request
- local chat-session tools are executed before falling back to a runtime-level tool executor
- each tool’s `ToolDefinition::execution_mode` controls how its calls interact with other calls in the same model turn
- if any invoked tool in a turn is `Sequential`, all tool calls in that turn run sequentially in model order (no overlap)
- otherwise, tool calls in a turn run concurrently

## Streaming

The runtime supports streamed chat output through `ChatChunk` and `ChatExecutionEvent`.

Streaming currently includes:

- message start events
- content delta events
- tool call delta events
- final response events

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
cargo test --workspace -- --ignored
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

### E2E workflow

`.github/workflows/e2e.yml` runs ignored tests with provider credentials.

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

- [x] OpenAI-compatible Backends
  - [x] [AIHubMix](https://aihubmix.com/?aff=OOiX)
  - [x] [OpenRouter](https://openrouter.ai/)
  - [x] [OpenAI](https://platform.openai.com/docs/guides/gpt/chat-completions-api)
  - [x] [Azure OpenAI API](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference)
- [x] llama.cpp Backends
  - [x] CPU
  - [ ] BLAS
  - [x] Accelerate.framework
  - [x] Metal
  - [x] Vulkan
  - [ ] CUDA
  - [ ] HIP
  - [ ] OpenVINO

### ASR API

- [x] [OpenAI](https://developers.openai.com/api/docs/guides/speech-to-text)

## License

Apache License 2.0. See `LICENSE`.
