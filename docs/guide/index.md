# Introduction

`xlai` is a **Rust-first** AI integration workspace for building reusable AI calling flows across **native** applications and the **browser**. The design is inspired by [moeru-ai/xsai](https://github.com/moeru-ai/xsai).

The project is built around a **unified API** with **pluggable backends**. The long-term goal is to support API-based models, tool integration, skill management, knowledge retrieval, vector search, and local or device inference behind the same overall API model.

## What works today

- Cargo workspace with native and `wasm32` build support
- OpenAI-compatible chat backend
- Chat sessions with per-session tool registration
- Streaming chat output
- Configurable tool-call execution (concurrent by default, sequential optional)
- Unit/mock tests and ignored end-to-end tests
- GitHub Actions for build, test, formatting, clippy, and e2e

## Where to go next

1. [Getting started](/guide/getting-started) — clone, build, test, and optional JS package build
2. [Configuration](/guide/configuration) — `.env` and provider variables
3. [Architecture](/architecture/) — crates and request flow
4. [Rust SDK](/rust/) — `RuntimeBuilder`, chat, agents, tools
5. [JavaScript / WASM](/js/) — `@yai-xlai/xlai`

For the full narrative and roadmap in the repository, see the [README on GitHub](https://github.com/yetanother.ai/xlai/blob/main/README.md).
