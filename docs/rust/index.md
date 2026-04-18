# Rust SDK

Application code typically depends on **`xlai-native`** (or **`xlai-wasm`** in the browser). Both expose `RuntimeBuilder`, chat and agent sessions, and backend configuration types.

## Platform entrypoints

| Crate         | Use when                                                                                      |
| ------------- | --------------------------------------------------------------------------------------------- |
| `xlai-native` | Native binaries and servers on macOS, Linux, or Windows. Optional `llama` and `qts` features. |
| `xlai-wasm`   | `wasm32-unknown-unknown` builds and JavaScript interop.                                       |
| `xlai-ffi`    | C ABI / shared library embedding (wraps `xlai-native`).                                       |

Domain types and traits live in **`xlai-core`** (semver-stable on crates.io). Session APIs live in **`xlai-runtime`** and are re-exported through **`xlai-native`** / **`xlai-wasm`**. The **`xlai-facade`** crate is an internal workspace helper (not on crates.io) used **only by `xlai-native`** for native aggregate wiring; **`xlai-wasm` does not depend on it**.

## Minimal pattern

1. Build a `RuntimeBuilder` and register a chat backend (`OpenAiConfig`, `LlamaCppConfig`, …).
2. Call `.build()?` to obtain a runtime.
3. Use `chat_session()` or `agent_session()?` for typed conversations.

Concrete examples (OpenAI, llama.cpp, streaming agents) are in the repository [README](https://github.com/yetanother.ai/xlai/blob/main/README.md#example) — they are kept there so they stay in sync with the workspace.

## Topics

- [Chat, agents, and tools](/rust/chat-and-agents) — tool registration, execution modes, and the agent tool loop
- [Streaming](/rust/streaming) — `ChatChunk` and `ChatExecutionEvent` behavior

## Optional: QTS in native builds

Enable the **`qts`** feature on `xlai-native` when you need `QtsTtsModel` without taking the dependency in every build.
