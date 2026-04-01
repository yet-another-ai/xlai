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
│   ├── xlai-backend-openai/
│   ├── xlai-core/
│   ├── xlai-ffi/
│   ├── xlai-native/
│   ├── xlai-runtime/
│   └── xlai-wasm/
└── .github/workflows/
```

### Crates

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
- `crates/xlai-backend-openai`
  OpenAI-compatible backend implementation using `reqwest`.

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

Local `.env` files are ignored by Git.

## Example

This is the current native Rust usage style:

```rust
use xlai_native::core::{ToolDefinition, ToolParameter, ToolParameterType, ToolResult};
use xlai_native::{OpenAiConfig, RuntimeBuilder, ToolCallExecutionMode};

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
        .with_system_prompt("Be concise.")
        .with_tool_call_execution_mode(ToolCallExecutionMode::Concurrent);

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

## Tool Calling

`Chat` sessions can register tools directly with async callbacks.

Current behavior:

- tools are registered per chat session
- tool calls are exposed to the model through the runtime request
- local chat-session tools are executed before falling back to a runtime-level tool executor
- multiple tool calls from the same model turn run concurrently by default
- sequential execution can be selected explicitly

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

The current OpenAI smoke test will also load `.env` automatically for local runs.

## CI

### Build workflow

`.github/workflows/build.yml` builds:

- Linux
- Windows
- macOS arm64
- macOS x86_64 via cross-target build
- `wasm32-unknown-unknown`

### Test workflow

`.github/workflows/test.yml` runs:

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`

### E2E workflow

`.github/workflows/e2e.yml` runs ignored tests with provider credentials.

It is intended to use a protected GitHub Environment such as `e2e`, with maintainer approval and environment secrets.

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

- [x] OpenAI-compatible Backends
  - [x] [AIHubMix](https://aihubmix.com/?aff=OOiX)
  - [x] [OpenRouter](https://openrouter.ai/)
  - [x] [OpenAI](https://platform.openai.com/docs/guides/gpt/chat-completions-api)
  - [x] [Azure OpenAI API](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference)
- [ ] llama.cpp Backends

## License

Apache License 2.0. See `LICENSE`.
