---
layout: home

hero:
  name: XLAI
  text: Rust-first AI integration
  tagline: A unified API with pluggable backends for native apps and the browser — chat, tools, streaming, and more.
  actions:
    - theme: brand
      text: Get started
      link: /guide/
    - theme: alt
      text: View on GitHub
      link: https://github.com/yetanother.ai/xlai

features:
  - title: Pluggable backends
    details: OpenAI-compatible HTTP, Google Gemini, local llama.cpp, and browser transformers.js — behind the same runtime model.
  - title: Sessions that fit your app
    details: Chat and agent sessions with per-session tools, streaming, and configurable tool execution.
  - title: Native and WASM
    details: Use xlai-native in Rust services and @yai-xlai/xlai in the browser on top of xlai-wasm.
  - title: Apache 2.0
    details: Licensed for use in open and commercial projects.
---

## Status

This project is in an **early stage**; the public API is still evolving. See the [introduction](/guide/) for what is implemented today and how to try it.

## Quick links

| Topic                  | Link                                      |
| ---------------------- | ----------------------------------------- |
| Install and build      | [Getting started](/guide/getting-started) |
| Environment variables  | [Configuration](/guide/configuration)     |
| Crate layout           | [Architecture](/architecture/)            |
| Rust runtime and tools | [Rust SDK](/rust/)                        |
| npm / WASM package     | [JavaScript](/js/)                        |
| TTS (Qwen3)            | [QTS](/qts/)                              |
| Releases and CI        | [Development](/development/)              |
