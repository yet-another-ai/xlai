# QTS 12Hz vocoder streaming (native)

This document describes how `xlai-qts-core` pipelines the ONNX vocoder with the GGML talker for **Qwen3-TTS-12Hz** checkpoints.

## Recommended settings

| Knob | Typical value | Notes |
|------|----------------|-------|
| `vocoder_chunk_size` (`SynthesizeRequest` / metadata `xlai.qts.vocoder_chunk_size`) | **4** | Matches the Qwen3-TTS paper packet size (4 codec frames ≈ 320 ms at 12.5 Hz). Use `0` for sequential decode (no pipelining). |
| Overlap (codec frames) | **1** (default) | Controlled by env `QWEN3_TTS_VOCODER_OVERLAP_FRAMES`. Values `2`–`8` are clamped; use **`3`** if you hear seams at chunk boundaries. |
| `vocoder_thread_count` | **4** (or half of `thread_count` when `0`) | ONNX Runtime CPU threads for the vocoder worker. |

Public constant: `xlai_qts_core::QTS12HZ_RECOMMENDED_VOCODER_CHUNK_FRAMES` (= `4`).

## Architecture

1. **Talker** (`TtsTransformer::rollout_codec_frames_kv_streaming`) emits codec frames and batches them into `VocoderChunk`s.
2. **Prefix warmup**: When ICL/x-vector reference frames exist, the first chunk is `prefix_warmup_only` — it only seeds overlap state; **no reference audio is written to the output PCM** (avoids a full prefix decode + proportional trim).
3. **Overlap-add** (`OverlapAddChunkDecoder` in `pipeline/vocoder_streaming.rs`): Each generated chunk may be decoded with the last *N* frames of the previous chunk prepended (*N* = overlap). The overlapping audio region is linearly crossfaded. The ONNX graph itself remains **stateless** per call.

## Benchmarking

Use the CLI:

```sh
cargo run -p xlai-qts-cli -- profile --text "..." --model-dir "$MODEL_DIR" --chunk-size 4 --runs 3
```

Vary `--chunk-size`, `QWEN3_TTS_VOCODER_OVERLAP_FRAMES`, and `xlai.qts.talker_kv_mode` to compare wall-clock rows in the printed table (`vocoder_decode`, `codec_rollout`, `pipeline_overlap`).

## Environment variables

| Variable | Effect |
|----------|--------|
| `QWEN3_TTS_VOCODER_OVERLAP_FRAMES` | Integer overlap in codec frames (default `1`, max `8`). |

See also: [`qts-export-and-hf-publish.md`](./qts-export-and-hf-publish.md) for artifact layout and future stateful-vocoder notes.
