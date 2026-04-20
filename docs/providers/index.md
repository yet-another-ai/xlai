# Supported providers

XLAI focuses on a **small set of first-party backends** with a shared `xlai-core` contract. Provider matrices in the repo README change as features land; this page summarizes the **intent** today.

## Chat

| Backend                   | Crate                         | Notes                                                                        |
| ------------------------- | ----------------------------- | ---------------------------------------------------------------------------- |
| OpenAI-compatible HTTP    | `xlai-backend-openai`         | OpenAI-style HTTP backend for OpenAI and other providers with matching APIs |
| OpenRouter Responses API  | `xlai-backend-openrouter`     | Dedicated OpenRouter backend for `/responses`; avoids assuming full OpenAI parity |
| Google Gemini             | `xlai-backend-gemini`         | Gemini HTTP API                                                              |
| llama.cpp (local GGUF)    | `xlai-backend-llama-cpp`      | Native inference via vendored `llama.cpp`                                    |
| transformers.js (browser) | `xlai-backend-transformersjs` | WASM-only; delegates generation to a JS adapter                              |

## Speech

| Capability                      | Status                                                                      |
| ------------------------------- | --------------------------------------------------------------------------- |
| OpenAI-compatible transcription | Supported via `xlai-backend-openai`                                         |
| OpenAI-compatible TTS           | Supported via `xlai-backend-openai`                                         |
| OpenRouter speech / media APIs  | Not yet exposed through a dedicated backend                                 |
| Local Qwen3 TTS (QTS)           | Native engine in `xlai-qts-core`; browser path is staged — see [QTS](/qts/) |

## llama.cpp acceleration matrix

The README tracks CPU/GPU EP support (Metal, Vulkan, CUDA roadmap, etc.). See [Support of LLM API providers](https://github.com/yetanother.ai/xlai/blob/main/README.md#support-of-llm-api-providers) on GitHub for the live checklist.
