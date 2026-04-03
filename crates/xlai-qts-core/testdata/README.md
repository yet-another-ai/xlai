# Test data

This directory is for mostly local fixtures. The root `.gitignore` ignores everything here except a small set of checked-in protobuf prompt fixtures used by integration tests.

- **`minimal.gguf`** (optional): tiny synthetic GGUF for parser experiments — generate with your preferred tooling and keep under ~512 KiB.
- **Golden vectors** (optional): for layer-B numerics, add `reference/*.bin` from an upstream Qwen3 TTS reference build (e.g. [predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp)) — not vendored in this repo.
- **`sample1.xvector.voice-clone-prompt.pb`**: checked-in xvector-only voice-clone prompt fixture generated from `testdata/sample1.wav` and `testdata/sample1.txt`.
- **`sample1.icl.voice-clone-prompt.pb`**: checked-in ICL voice-clone prompt fixture generated from `samples/sample1.wav`.

Integration tests use real checkpoints via `QWEN3_TTS_MODEL_DIR` (see `docs/testing.md`).
