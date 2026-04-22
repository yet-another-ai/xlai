# Supported providers

XLAI focuses on a **small set of first-party backends** with a shared `xlai-core` contract. Provider matrices in the repo README change as features land; this page summarizes the **intent** today.

## Chat

| Backend                   | Crate                         | Notes                                                                             |
| ------------------------- | ----------------------------- | --------------------------------------------------------------------------------- |
| OpenAI-compatible HTTP    | `xlai-backend-openai`         | OpenAI-style HTTP backend for OpenAI and other providers with matching APIs       |
| OpenRouter Responses API  | `xlai-backend-openrouter`     | Dedicated OpenRouter backend for `/responses`; avoids assuming full OpenAI parity |
| Google Gemini             | `xlai-backend-gemini`         | Gemini HTTP API                                                                   |
| llama.cpp (local GGUF)    | `xlai-backend-llama-cpp`      | Native inference via vendored `llama.cpp`                                         |
| transformers.js (browser) | `xlai-backend-transformersjs` | WASM-only; delegates generation to a JS adapter                                   |

## Embeddings

| Backend                   | Crate                         | Notes                                                                                          |
| ------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------- |
| OpenAI-compatible HTTP    | `xlai-backend-openai`         | Uses `/v1/embeddings`; supports `dimensions` when the upstream provider/model supports it      |
| Google Gemini             | `xlai-backend-gemini`         | Uses Gemini embeddings endpoints; supports output dimensionality                               |
| llama.cpp (local GGUF)    | `xlai-backend-llama-cpp`      | Local embeddings from embedding-capable GGUF models; custom `dimensions` are not supported     |
| transformers.js (browser) | `xlai-backend-transformersjs` | Uses feature extraction with pooled, normalized vectors; custom `dimensions` are not supported |

## Speech

| Capability                      | Status                                                                      |
| ------------------------------- | --------------------------------------------------------------------------- |
| OpenAI-compatible transcription | Supported via `xlai-backend-openai`                                         |
| OpenAI-compatible TTS           | Supported via `xlai-backend-openai`                                         |
| OpenRouter speech / media APIs  | Not yet exposed through a dedicated backend                                 |
| Local Qwen3 TTS (QTS)           | Native engine in `xlai-qts-core`; browser path is staged — see [QTS](/qts/) |

OpenRouter remains chat-only in XLAI today. The dedicated `xlai-backend-openrouter` backend targets `/responses` and does not expose a first-class embeddings surface.

## llama.cpp acceleration matrix

The README tracks the live native accelerator matrix for `llama.cpp` and QTS. Today the workspace defaults the local native stacks to request `openblas`, `cuda`, `hip`, and `openvino`, while Vulkan and Metal remain opt-in. The vendored `ggml` / `llama.cpp` core is built statically; CUDA / OpenVINO / ROCm runtimes are linked as **external system SDKs** discovered through `CUDA_PATH` / `OpenVINO_DIR` / `ROCM_PATH`. `hip` is still downgraded to a build warning on every static-core build because upstream `ggml` does not support `GGML_HIP=ON` together with static linking. See [Native vendor layout](../development/native-vendor) for the linking contract and [Support of LLM API providers](https://github.com/yetanother.ai/xlai/blob/main/README.md#support-of-llm-api-providers) on GitHub for the checklist.
