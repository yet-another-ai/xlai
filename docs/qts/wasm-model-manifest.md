# QTS browser model manifest

Browser clients cannot rely on a single POSIX directory. Use a **versioned manifest** describing required blobs, optional URLs, and integrity metadata for caching.

## JSON schema (informal)

Top-level object:

| Field            | Type    | Required | Description                                           |
| ---------------- | ------- | -------- | ----------------------------------------------------- |
| `schema_version` | integer | yes      | Must be `1` for this revision.                        |
| `model_id`       | string  | yes      | Human-readable id (e.g. Hugging Face repo name).      |
| `revision`       | string  | yes      | Opaque version (commit, tag, or content hash prefix). |
| `files`          | array   | yes      | List of file descriptors (see below).                 |

Each element of `files`:

| Field          | Type    | Required | Description                                                                                                                                    |
| -------------- | ------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `logical_name` | string  | yes      | Stable key used by the loader (`main_gguf`, `vocoder_onnx`, `reference_codec_onnx`, `reference_codec_preprocess_json`, `tokenizer_config`, …). |
| `filename`     | string  | yes      | Basename on disk / in cache (e.g. `qwen3-tts-vocoder.onnx`).                                                                                   |
| `sha256`       | string  | no       | Hex digest for integrity after download.                                                                                                       |
| `size_bytes`   | integer | no       | Expected size; helps resumable fetch UIs.                                                                                                      |
| `url`          | string  | no       | Direct download URL; omit if bytes come from elsewhere (OPFS upload).                                                                          |

## Required logical names for full QTS

Align with native `ModelPaths` / export docs:

- `main_gguf` — talker GGUF (any supported quantized name).
- `vocoder_onnx` — `qwen3-tts-vocoder.onnx`.
- For ICL voice clone: `reference_codec_onnx`, `reference_codec_preprocess_json` (filename `qwen3-tts-reference-codec-preprocess.json`).

Tokenizer / `config.json` are typically embedded in GGUF for this stack; if split artifacts are added later, extend `logical_name` enums in `xlai-qts-manifest` (shared by `xlai_qts_core::browser` and `xlai-wasm`).

## Caching recommendations

1. Key cache entries by `(model_id, revision, logical_name, sha256)`.
2. If `sha256` is missing, use `revision` + `size_bytes` only for best-effort invalidation.
3. Prefer **Cache Storage** or **OPFS** for multi-hundred-MB GGUF; keep a small manifest in IndexedDB pointing to OPFS handles.

## Rust types

See `xlai_qts_manifest` / `xlai_qts_core::browser` for serde-compatible structs and `validate_required_files`; the `xlai-wasm` crate (feature `qts`) depends on `xlai-qts-manifest` for the same types.
