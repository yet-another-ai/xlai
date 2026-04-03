# Test data

This directory is for mostly local fixtures. The root `.gitignore` ignores everything here except a small set of checked-in CBOR prompt fixtures used by integration tests.

- **`minimal.gguf`** (optional): tiny synthetic GGUF for parser experiments — generate with your preferred tooling and keep under ~512 KiB.
- **Golden vectors** (optional): for layer-B numerics, add `reference/*.bin` from an upstream Qwen3 TTS reference build (e.g. [predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp)) — not vendored in this repo.
- **`sample1.xvector.voice-clone-prompt.cbor`**: checked-in xvector-only voice-clone prompt fixture generated from `testdata/sample1.wav` and `testdata/sample1.txt`.
- **`sample1.icl.voice-clone-prompt.cbor`**: checked-in ICL voice-clone prompt fixture generated from `samples/sample1.wav`.
- **`sample1_ref.wav`** (optional): same reference clip as the golden prompts; if present, the ignored test `integration_native_xvector_prompt_parity_shape` compares native `create_voice_clone_prompt` metadata to the CBOR fixture.

Integration tests use real checkpoints via `XLAI_QTS_MODEL_DIR` (see `docs/testing.md`).
