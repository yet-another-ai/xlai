# QTS crate split (future work)

`xlai-qts-core` currently combines:

- GGUF + GGML talker inference
- ONNX vocoder and optional reference-codec paths
- Native `TtsModel` bridge (`QtsTtsModel`)
- Re-exports of browser manifest types under `xlai_qts_core::browser`

This keeps one coherent product crate for native TTS, but it is a large conceptual surface.

After the main workspace and platform boundaries are stable, consider splitting along **dependency seams**, for example:

1. **`xlai-qts-engine`** — GGML talker + tokenizer + streaming pipeline internals (depends on `xlai-sys-ggml`).
2. **`xlai-qts-onnx`** (or feature-gated modules) — ORT session wiring and execution-provider features.
3. **`xlai-qts-adapter`** — thin `TtsModel` implementation mapping `xlai-core::TtsRequest` to the engine.

Keep **`xlai-qts-manifest`** as the browser-only serde surface; `xlai-wasm` should continue to avoid pulling native GGML/ORT until a real WASM engine path exists.

Any split should preserve the existing **feature names** on the outer crate or document a migration for accelerator flags (`cuda`, `metal`, …).
