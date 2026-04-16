# QTS (Qwen3 TTS)

**QTS** is the local **Qwen3 TTS** stack in XLAI: native synthesis in `xlai-qts-core`, a CLI in `xlai-qts-cli`, and browser-facing manifest types for future WASM integration.

## Native

- Engine and `TtsModel` bridge: `xlai-qts-core`
- CLI (`xlai-qts`): synthesize, profile, interactive TUI
- Export and Hugging Face packaging: Python `scripts/qts` — [Export and Hugging Face](/qts/export-and-hf-publish)
- Vocoder pipelining and overlap: [Vocoder streaming](/qts/vocoder-streaming)

## Browser / WASM

The in-browser engine is **not** complete yet; `xlai-wasm` exposes a stable surface and capability types while GGML/ORT browser work proceeds.

- [Browser / WASM runtime](/qts/wasm-browser-runtime) — feasibility and design direction
- [Browser model manifest](/qts/wasm-model-manifest) — versioned manifest for fetch and cache

## Hugging Face

- [Hugging Face model card](/qts/huggingface-qts-model-card) — template used when publishing artifact repos
