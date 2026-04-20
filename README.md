# XLAI

`xlai` is a **Rust-first** AI integration workspace for building reusable AI calling flows across **native** applications and the **browser**, behind a unified API with pluggable backends. The name was inspired by [moeru-ai/xsai](https://github.com/moeru-ai/xsai).

> **Status:** early stage; the public API is still evolving.

## Documentation

The full documentation lives in the VitePress site under [`docs/`](docs/):

| Topic                  | Link                                                        |
| ---------------------- | ----------------------------------------------------------- |
| Introduction           | [`docs/guide/index.md`](docs/guide/index.md)                |
| Install and build      | [`docs/guide/getting-started.md`](docs/guide/getting-started.md) |
| Environment variables  | [`docs/guide/configuration.md`](docs/guide/configuration.md) |
| Architecture & crates  | [`docs/architecture/index.md`](docs/architecture/index.md)  |
| Crate taxonomy         | [`docs/development/crates-taxonomy.md`](docs/development/crates-taxonomy.md) |
| Rust SDK               | [`docs/rust/index.md`](docs/rust/index.md)                  |
| JavaScript / WASM      | [`docs/js/index.md`](docs/js/index.md)                      |
| Provider support       | [`docs/providers/index.md`](docs/providers/index.md)        |
| QTS (Qwen3 TTS)        | [`docs/qts/index.md`](docs/qts/index.md)                    |
| CI, testing, releases  | [`docs/development/index.md`](docs/development/index.md)    |

Preview the site locally:

```bash
pnpm install
pnpm docs:dev
```

## Quick start

Requirements: Rust stable (MSRV **1.94**, edition **2024**) and **pnpm**.

```bash
git clone https://github.com/yetanother.ai/xlai.git
cd xlai
cargo build --workspace
cargo test --workspace
```

JavaScript workspace:

```bash
pnpm install
pnpm --filter @yai-xlai/xlai build
pnpm --filter @yai-xlai/xlai test
```

For end-to-end provider tests, copy `.env.example` to `.env` and fill in credentials, then run:

```bash
cargo test --workspace -- --ignored --test-threads=1
```

See [`docs/guide/configuration.md`](docs/guide/configuration.md) for the full list of environment variables and [`docs/development/ci-and-testing.md`](docs/development/ci-and-testing.md) for the testing model.

## Support of LLM API Providers

### Chat API

- [x] OpenAI
  - [x] [OpenAI](https://developers.openai.com/api/docs/guides/text)
  - [x] [Azure OpenAI API](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference)
- [x] [OpenRouter](https://openrouter.ai/docs/api/api-reference/responses/create-responses)
- [x] llama.cpp
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

Apache License 2.0. See [`LICENSE`](LICENSE).
