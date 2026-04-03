# Qwen3-TTS: export artifacts and publish to Hugging Face

This repository owns the **non-protobuf** export pipeline for Qwen3-TTS runtime files used by `xlai-qts-core` and `xlai-backend-qts`.

Protobuf-based voice-clone prompt export (`.pb`) is **not** supported here; use CBOR prompts or Rust-native `create_voice_clone_prompt` instead.

## Prerequisites

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/)
- From the repo root: `uv sync`

## Export GGUF, vocoder ONNX, and reference-codec ONNX

```bash
uv run export-model-artifacts \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --out-dir /path/to/out
```

This writes:

| File | Purpose |
|------|---------|
| `qwen3-tts-0.6b-f16.gguf` (and optional `q8_0`) | Talker + code predictor + tokenizer metadata |
| `qwen3-tts-vocoder.onnx` | ONNX Runtime vocoder |
| `qwen3-tts-reference-codec.onnx` | Speech-tokenizer **encode** graph for ICL `ref_code` |
| `qwen3-tts-reference-codec-preprocess.json` | Preprocess metadata for the Rust reference-codec path |

Flags mirror the implementation in `scripts/qts/export_model_artifacts.py` (`--main-type`, `--vocoder-out`, `--ref-codec-out`, `--local-files-only`, etc.).

## Prepare a Hugging Face release directory

From the **repository root** (so paths and `git rev-parse` resolve correctly):

```bash
uv run xlai-qts-hf-release \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --hf-repo-dir /path/to/cloned/hf-model-repo
```

Defaults:

- Export output: `models/qwen3-tts-bundle`
- Staging output: `target/hf-qts-release` (unless `--hf-repo-dir` is the only path, then files are written directly into the clone)
- Model card template: `docs/huggingface-qts-model-card.md`

Use `--skip-export` to package artifacts that are already on disk.

## CI

Tagged releases use `.github/workflows/hf-release-qts.yml` (repository root). Push a tag matching `qts-v*` (for example `qts-v1.0.0`) so ordinary semver tags on the monorepo do not trigger the export. Configure the `HF_TOKEN` secret on the `publish` environment.
