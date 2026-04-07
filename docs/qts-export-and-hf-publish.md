# Qwen3-TTS: export artifacts and publish to Hugging Face

This repository owns the **non-protobuf** export pipeline for Qwen3-TTS runtime files used by `xlai-qts-core` (engine and native `TtsModel` bridge).

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

The Hugging Face model packaging job is **manual only**: run [`.github/workflows/hf-release-qts.yml`](../.github/workflows/hf-release-qts.yml) from the Actions tab (**workflow_dispatch**). Configure the `HF_TOKEN` secret on the `publish` environment.

Pushing a `v*` tag does **not** trigger that workflow; it **does** run [`.github/workflows/build.yml`](../.github/workflows/build.yml) (artifacts) and [`.github/workflows/publish.yml`](../.github/workflows/publish.yml) (crates.io + npm) if you use those release paths. See [`docs/publishing.md`](publishing.md) for registry releases.

## Future: stateful streaming vocoder ONNX

The native runtime currently calls `qwen3-tts-vocoder.onnx` in a **stateless** way: each ORT run takes a full `[1, n_frames, n_codebooks]` window. Chunk continuity is handled in Rust via a small **overlap-and-add** layer (`OverlapAddChunkDecoder`; see [`qts-vocoder-streaming.md`](./qts-vocoder-streaming.md)).

A **stateful** export (extra inputs/outputs for convolver or decoder cache tensors) could remove redundant re-decoding of context frames and shrink overlap further. That would require:

1. A new or alternate ONNX graph from the training/export stack (`scripts/qts/export_model_artifacts.py` and upstream checkpoints).
2. Matching changes in `crates/xlai-qts-core/src/pipeline/vocoder.rs` to bind and thread state across steps.
3. Versioning or manifest entries so HF bundles can declare which vocoder interface they ship.

Until then, tuning `xlai.qts.vocoder_chunk_size` and `QWEN3_TTS_VOCODER_OVERLAP_FRAMES` is the supported path.
