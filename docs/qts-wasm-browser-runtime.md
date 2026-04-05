# QTS browser / WASM runtime

This document records feasibility findings and the intended runtime shape for local Qwen3 TTS in the browser (`wasm32-unknown-unknown`).

## Feasibility spikes (repo state)

### GGML talker (`xlai-sys` + vendored GGML)

- `cargo check -p xlai-sys --target wasm32-unknown-unknown --features qts-ggml` fails during CMake configuration: the default C toolchain cannot compile a trivial program for `wasm32-unknown-unknown` (no compatible clang target / CMake “unknown” platform).
- The current integration links **static** GGML libraries produced by CMake for **host** targets (see `crates/xlai-sys/build.rs`). That path assumes native linkers, pthread/OpenMP-style deps on Linux, frameworks on Apple, etc.
- The `webgpu` Cargo feature on `xlai-sys` forwards to `GGML_WEBGPU` in CMake. That is a **native** GGML backend build flag today, not a guarantee of a working **browser** WebGPU path from Rust `wasm32-unknown-unknown`.

**Conclusion:** End-to-end local GGML talker in the browser requires a **separate** build and integration story (for example Emscripten-based GGML, a prebuilt browser GGML WASM module, or a different inference stack). The existing `xlai-sys` CMake pipeline is not sufficient for `wasm32-unknown-unknown` without additional toolchain work.

### Vocoder (`ort` / ONNX Runtime)

- `cargo check -p ort --target wasm32-unknown-unknown` fails: `ort-sys` does not ship prebuilt ONNX Runtime binaries for `wasm32-unknown-unknown` with the default feature set.
- Browser deployments typically use **onnxruntime-web** (WASM + optional WebGPU EP) or a custom ORT build linked explicitly.

**Conclusion:** The current `xlai-qts-core` vocoder path (native `ort` sessions from filesystem paths) does not port directly. A browser vocoder needs either a custom ORT WASM build wired into `ort`, or a dedicated browser execution adapter.

## Design direction

1. **Capability-based degradation:** Expose structured capability objects from WASM (see `qtsBrowserTtsCapabilities`) so UIs can branch on CPU vs WebGPU without relying on native env vars.
2. **Asset manifest:** Use a versioned manifest (see `docs/qts-wasm-model-manifest.md`) for fetch, cache, and integrity checks instead of a POSIX `model_dir` on disk.
3. **Threading:** Avoid `std::thread` worker loops in browser builds; prefer single-threaded async or explicit Web Workers with message passing once an engine exists.
4. **Next implementation steps (when engines exist):** Plug a real `Qwen3TtsEngine` (or split talker/vocoder adapters) behind the same `TtsModel` / WASM entrypoints added for the stub phase.

## Browser matrix (all major browsers)

| Browser   | Talker (GGML) | Vocoder (ORT) | Notes                                      |
|----------|---------------|---------------|--------------------------------------------|
| Chromium | Pending       | Pending       | WebGPU most capable when engines land      |
| Firefox  | Pending       | Pending       | WebGPU / EP support may lag Chromium       |
| Safari   | Pending       | Pending       | Test WebGPU + WASM stack explicitly        |

Until engines are integrated, the shipped WASM API returns a stable **unsupported** error with `details.code = qts_wasm_engine_pending` so tests and UIs can detect the phase.
