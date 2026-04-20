# Configuration

## Local environment file

For local development and end-to-end tests, copy the template from the repository root:

```bash
cp .env.example .env
```

Fill in values for the backends you use. `.env` is gitignored.

## Common variables

The root [`.env.example`](https://github.com/yetanother.ai/xlai/blob/main/.env.example) is the source of truth. It includes:

| Area                 | Examples                                                                                                       |
| -------------------- | -------------------------------------------------------------------------------------------------------------- |
| OpenAI-compatible    | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_TRANSCRIPTION_MODEL`, … |
| OpenRouter           | `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `OPENROUTER_MODEL`, optional app headers                          |
| Gemini               | `GEMINI_API_KEY`, `GEMINI_BASE_URL`, `GEMINI_MODEL`, `GEMINI_EMBEDDING_MODEL`, …                               |
| llama.cpp            | `LLAMA_CPP_MODEL` — path to a local GGUF file                                                                  |
| QTS                  | `XLAI_QTS_MODEL_DIR`, optional voice-clone inputs, vocoder tuning env vars                                     |
| Hugging Face release | `HF_TOKEN` (for maintainers running the HF workflow)                                                           |

Ignored OpenAI e2e tests load `.env` automatically when you run them locally.

## Embeddings

- OpenAI-compatible one-shot embeddings default to `OPENAI_EMBEDDING_MODEL` when set, otherwise they fall back to `OPENAI_MODEL`.
- Gemini embeddings use `GEMINI_EMBEDDING_MODEL` when set; the default in `.env.example` is `gemini-embedding-001`.
- `dimensions` is supported for remote OpenAI-compatible and Gemini embeddings when the selected provider/model supports it.
- Local `llama.cpp` and browser `transformers.js` embeddings ignore provider-specific projection and return `Unsupported` if `dimensions` is requested.

## Next steps

- [Rust SDK](/rust/) — wire `OpenAiConfig`, `LlamaCppConfig`, or other backends in code
- [Development / CI](/development/ci-and-testing) — how credentials are used in GitHub Actions
