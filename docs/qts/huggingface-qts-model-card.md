---
license: apache-2.0
language:
  - zh
  - en
  - ja
  - ko
  - de
  - fr
  - ru
  - pt
  - es
  - it
base_model:
  - Qwen/Qwen3-TTS-12Hz-0.6B-Base
pipeline_tag: text-to-speech
quantized_by: dsh0416
tags:
  - audio
  - tts
  - voice-clone
---

# Qwen3-TTS-12Hz-0.6B-Base-QTS

`Qwen3-TTS-12Hz-0.6B-Base-QTS` is a distribution repository for model artifacts
produced by **[yetanother.ai/xlai](https://github.com/yetanother.ai/xlai)** (Qwen3 TTS native stack).

This Hugging Face repository is intended to contain stable, downloadable runtime
artifacts only:

- one shared `qwen3-tts-vocoder.onnx`
- `qwen3-tts-reference-codec.onnx` and `qwen3-tts-reference-codec-preprocess.json` for **ICL** voice clone in `xlai-qts-core`
- one or more GGUF variants such as `qwen3-tts-0.6b-f16.gguf`
- optional additional GGUF variants such as `qwen3-tts-0.6b-q8_0.gguf`

It is not the source-of-truth repository for code, export logic, or developer
documentation. Those live in **[yetanother.ai/xlai](https://github.com/yetanother.ai/xlai)**.

## Relationship To `xlai`

- GitHub [`yetanother.ai/xlai`](https://github.com/yetanother.ai/xlai): source code, export scripts (`scripts/qts`), Rust runtime (`xlai-qts-core`)
- Hugging Face [`dsh0416/Qwen3-TTS-12Hz-0.6B-Base-QTS`](https://huggingface.co/dsh0416/Qwen3-TTS-12Hz-0.6B-Base-QTS): exported runtime artifacts

Recommended maintenance flow:

1. Change behavior in the GitHub repository first.
2. Export artifacts from a known Git commit (`uv run export-model-artifacts`, or `uv run xlai-qts-hf-release`).
3. Publish only the built model files to this Hugging Face repository, preferably from the tagged GitHub Actions release workflow in `xlai`.
4. Keep this model card aligned with the GitHub docs, but do not treat this repository as a second source repository.

## Included Files

Expected root layout:

```text
{{ROOT_LAYOUT}}
README.md
SHA256SUMS
```

Notes:

- `qwen3-tts-vocoder.onnx` is shared across all GGUF variants in this repository.
- `qwen3-tts-reference-codec.onnx` and `qwen3-tts-reference-codec-preprocess.json` are required for **ICL** voice cloning in `xlai`; x-vector-only clone does not need them.
- The Rust runtime expects the GGUF, vocoder ONNX, and (for ICL) reference-codec files to live in the same directory by default (`xlai_qts_core::ModelPaths`).
- Not every release must ship every quantization variant.
- For the current artifact set, `q8_0` is the recommended default download and `f16` is the reference-quality export.

## Current Quantization Support

At the moment, the xlai exporter supports:

{{QUANTIZATION_LIST}}

Other quantization types may appear in future releases once the export and
validation pipeline is ready.

## Usage With `xlai`

See the source repository for current usage and export documentation:

- GitHub: [`yetanother.ai/xlai`](https://github.com/yetanother.ai/xlai)
- Export / publish: [`docs/qts/export-and-hf-publish.md`](https://github.com/yetanother.ai/xlai/blob/main/docs/qts/export-and-hf-publish.md)

Typical local layout:

```text
models/
  qwen3-tts-0.6b-f16.gguf
  qwen3-tts-vocoder.onnx
  qwen3-tts-reference-codec.onnx
  qwen3-tts-reference-codec-preprocess.json
```

Example CLI usage:

```bash
cargo run -p xlai-qts-cli -- synthesize \
  --model-dir /path/to/models \
  --text "hello" \
  --out target/hello.wav
```

## Provenance

Current source repository snapshot:

- GitHub commit: `{{SOURCE_COMMIT}}`

Current artifact checksums:

{{CHECKSUM_LIST}}

For future releases, it is recommended to record:

- source GitHub commit SHA from `yetanother.ai/xlai`
- exported file list
- SHA256 checksums
- any release-specific notes such as added or removed quantization variants

## Base Model

Base upstream model:

- [`Qwen/Qwen3-TTS-12Hz-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
