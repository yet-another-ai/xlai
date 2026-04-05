# QTS fixture bundle (local export)

This directory is for **locally generated** Qwen3-TTS runtime files. Large artifacts are gitignored (see `.gitignore`).

## Generate a q8_0 bundle (talker GGUF + vocoder ONNX)

From the repo root:

```sh
uv sync
uv run export-model-artifacts \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --out-dir fixtures/qts \
  --main-type q8_0
```

You should get at least:

- `qwen3-tts-0.6b-q8_0.gguf`
- `qwen3-tts-vocoder.onnx`

Point `XLAI_QTS_MODEL_DIR` at this folder (plus tokenizer / `config.json` from the same upstream snapshot — copy or symlink from the Hugging Face cache if needed).

## Reference codec (ICL voice clone)

Full export also writes `qwen3-tts-reference-codec.onnx` and `qwen3-tts-reference-codec-preprocess.json`. On some **PyTorch + transformers** combinations, reference-codec ONNX tracing can fail (known issue with newer `torch.onnx` / Mimi masking paths).

If that happens, you can still validate GGUF + vocoder export:

```sh
uv run export-model-artifacts \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --out-dir fixtures/qts \
  --main-type q8_0 \
  --skip-reference-codec
```

Install **SoX** on macOS (`brew install sox`) if you want to silence optional SoX warnings from audio deps.
