#!/usr/bin/env python3
"""Export Qwen3-TTS main weights to GGUF, 12Hz vocoder ONNX, and reference-codec (encode) ONNX."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import onnx
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from tqdm import tqdm

import gguf

from scripts.qts.dtype_utils import resolve_dtype

logger = logging.getLogger(__name__)


@contextmanager
def _transformers_drop_packed_sequence_mask_for_onnx_trace() -> Iterator[None]:
    """Avoid packed-sequence mask branch in ``create_causal_mask`` during ONNX trace.

    For ``position_ids`` + ``attention_mask is None``, recent transformers always
    builds ``packed_sequence_mask`` and combines it via vmap; that path hits
    functorch/dynamo ops that break ``torch.onnx.export(..., dynamo=False)`` (JIT trace).
    Single-sequence reference audio (batch 1) does not need that branch.
    """
    import transformers.masking_utils as masking_utils

    real_preprocess: Callable[..., Any] = masking_utils._preprocess_mask_arguments

    def _preprocess_drop_packed(*args: Any, **kwargs: Any) -> Any:
        early_exit, attention_mask, _packed, kv_length, kv_offset = real_preprocess(
            *args, **kwargs
        )
        return early_exit, attention_mask, None, kv_length, kv_offset

    masking_utils._preprocess_mask_arguments = _preprocess_drop_packed
    try:
        yield
    finally:
        masking_utils._preprocess_mask_arguments = real_preprocess


@contextmanager
def _patch_torch_cdist_p2_for_onnx_trace() -> Iterator[None]:
    """Replace ``torch.cdist(..., p=2)`` with a broadcast formulation for ONNX trace.

    Mimi / RVQ codebooks call ``torch.cdist`` in ``modeling_mimi``; the TorchScript ONNX
    symbolic for ``cdist`` can assert on dynamic shapes. Elementwise squared distance +
    ``sqrt`` lowers cleanly to standard ONNX ops.
    """
    real_cdist = torch.cdist

    def cdist_p2_onnx(
        x1: torch.Tensor,
        x2: torch.Tensor,
        p: float = 2,
        compute_mode: str = "use_mm_for_euclid_dist_if_necessary",
    ) -> torch.Tensor:
        if p != 2:
            return real_cdist(x1, x2, p, compute_mode)
        # x1: (..., N, D), x2: (..., M, D) -> (..., N, M)
        return torch.sqrt(
            torch.clamp(
                ((x1.unsqueeze(-2) - x2.unsqueeze(-3)) ** 2).sum(dim=-1),
                min=0.0,
            )
        )

    torch.cdist = cdist_p2_onnx  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.cdist = real_cdist  # type: ignore[method-assign]


def _qwen3_tts_tokenizer_cls():  # lazy: importing qwen_tts pulls torch/transformers
    from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

    return Qwen3TTSTokenizer

MODEL_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "*.safetensors",
    "speech_tokenizer/*.json",
    "speech_tokenizer/*.safetensors",
]

MAIN_TYPE_TO_QUANT = {
    "f16": gguf.GGMLQuantizationType.F16,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
}

MAIN_TYPE_TO_FILE_TYPE = {
    "f16": gguf.LlamaFileType.MOSTLY_F16,
    "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
}

SUPPORTED_MAIN_TYPES = ("f16", "q8_0")


def configure_stdio() -> None:
    """Use UTF-8 stdio so exporter status logs do not fail on Windows cp1252."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # Keep export behavior intact even if the host stream is not reconfigurable.
            pass

class Qwen3MainGgufExporter:
    """Export the talker, code predictor, and tokenizer metadata to GGUF."""

    TENSOR_MAP = {
        "talker.model.codec_embedding.weight": "talker.codec_embd.weight",
        "talker.model.text_embedding.weight": "talker.text_embd.weight",
        "talker.codec_head.weight": "talker.codec_head.weight",
        "talker.model.norm.weight": "talker.output_norm.weight",
        "talker.text_projection.linear_fc1.weight": "talker.text_proj.fc1.weight",
        "talker.text_projection.linear_fc1.bias": "talker.text_proj.fc1.bias",
        "talker.text_projection.linear_fc2.weight": "talker.text_proj.fc2.weight",
        "talker.text_projection.linear_fc2.bias": "talker.text_proj.fc2.bias",
        "talker.code_predictor.model.norm.weight": "code_pred.output_norm.weight",
    }

    TALKER_LAYER_PATTERNS = [
        (r"talker\.model\.layers\.(\d+)\.input_layernorm\.weight", "talker.blk.{}.attn_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight", "talker.blk.{}.attn_q.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight", "talker.blk.{}.attn_k.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight", "talker.blk.{}.attn_v.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight", "talker.blk.{}.attn_output.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.q_norm\.weight", "talker.blk.{}.attn_q_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.k_norm\.weight", "talker.blk.{}.attn_k_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.post_attention_layernorm\.weight", "talker.blk.{}.ffn_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight", "talker.blk.{}.ffn_gate.weight"),
        (r"talker\.model\.layers\.(\d+)\.mlp\.up_proj\.weight", "talker.blk.{}.ffn_up.weight"),
        (r"talker\.model\.layers\.(\d+)\.mlp\.down_proj\.weight", "talker.blk.{}.ffn_down.weight"),
    ]

    CODE_PREDICTOR_LAYER_PATTERNS = [
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.input_layernorm\.weight", "code_pred.blk.{}.attn_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight", "code_pred.blk.{}.attn_q.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight", "code_pred.blk.{}.attn_k.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight", "code_pred.blk.{}.attn_v.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight", "code_pred.blk.{}.attn_output.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.q_norm\.weight", "code_pred.blk.{}.attn_q_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.k_norm\.weight", "code_pred.blk.{}.attn_k_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.post_attention_layernorm\.weight", "code_pred.blk.{}.ffn_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight", "code_pred.blk.{}.ffn_gate.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.mlp\.up_proj\.weight", "code_pred.blk.{}.ffn_up.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.mlp\.down_proj\.weight", "code_pred.blk.{}.ffn_down.weight"),
    ]

    CODE_PREDICTOR_CODEBOOK_PATTERNS = [
        (r"talker\.code_predictor\.model\.codec_embedding\.(\d+)\.weight", "code_pred.codec_embd.{}.weight"),
        (r"talker\.code_predictor\.lm_head\.(\d+)\.weight", "code_pred.lm_head.{}.weight"),
    ]

    def __init__(self, input_dir: Path, output_path: Path, output_type: str):
        self.input_dir = input_dir
        self.output_path = output_path
        self.output_type = output_type
        self.config = self._load_config()
        self._extract_params()

    def _load_config(self) -> dict[str, Any]:
        config_path = self.input_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return json.loads(config_path.read_text(encoding="utf-8"))

    def _extract_params(self) -> None:
        talker_config = self.config.get("talker_config", {})
        code_predictor_config = talker_config.get("code_predictor_config", {})
        speaker_encoder_config = self.config.get("speaker_encoder_config", {})

        self.hidden_size = talker_config.get("hidden_size", 1024)
        self.intermediate_size = talker_config.get("intermediate_size", 3072)
        self.num_hidden_layers = talker_config.get("num_hidden_layers", 28)
        self.num_attention_heads = talker_config.get("num_attention_heads", 16)
        self.num_kv_heads = talker_config.get("num_key_value_heads", 8)
        self.head_dim = talker_config.get("head_dim", 128)
        self.vocab_size = talker_config.get("vocab_size", 3072)
        self.text_vocab_size = talker_config.get("text_vocab_size", 151936)
        self.text_hidden_size = talker_config.get("text_hidden_size", 2048)
        self.num_code_groups = talker_config.get("num_code_groups", 16)
        self.rms_norm_eps = talker_config.get("rms_norm_eps", 1e-6)
        self.rope_theta = talker_config.get("rope_theta", 1_000_000)
        self.mrope_section = talker_config.get("rope_scaling", {}).get("mrope_section", [24, 20, 20])
        self.code_predictor_num_layers = code_predictor_config.get("num_hidden_layers", 5)
        self.code_predictor_vocab_size = code_predictor_config.get("vocab_size", 2048)
        self.speaker_enc_dim = speaker_encoder_config.get("enc_dim", 1024)
        self.speaker_sample_rate = speaker_encoder_config.get("sample_rate", 24000)
        self.codec_pad_id = talker_config.get("codec_pad_id", 2148)
        self.codec_bos_id = talker_config.get("codec_bos_id", 2149)
        self.codec_eos_id = talker_config.get("codec_eos_token_id", 2150)
        self.model_name = "Qwen3-TTS-12Hz-0.6B"

    def _map_tensor_name(self, hf_name: str) -> str | None:
        if hf_name in self.TENSOR_MAP:
            return self.TENSOR_MAP[hf_name]
        for pattern, template in self.TALKER_LAYER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                return template.format(match.group(1))
        for pattern, template in self.CODE_PREDICTOR_LAYER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                return template.format(match.group(1))
        for pattern, template in self.CODE_PREDICTOR_CODEBOOK_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                return template.format(match.group(1))
        return None

    def _get_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        safetensor_files = sorted(self.input_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {self.input_dir}")
        for sf_path in safetensor_files:
            logger.info("Loading tensors from %s", sf_path.name)
            with safe_open(sf_path, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    yield name, handle.get_tensor(name)

    def _should_quantize(self, tensor_name: str) -> bool:
        if any(x in tensor_name for x in ["_embd", "codebook"]):
            return False
        if "_norm" in tensor_name:
            return False
        if ".bias" in tensor_name:
            return False
        if "lm_head" in tensor_name or "codec_head" in tensor_name:
            return False
        return True

    def _convert_dtype(
        self,
        tensor: torch.Tensor,
        tensor_name: str,
    ) -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
        data = tensor.float().numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
        if data.ndim <= 1:
            return data.astype(np.float32), gguf.GGMLQuantizationType.F32
        if self.output_type == "f16":
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16
        if not self._should_quantize(tensor_name):
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16
        quant = MAIN_TYPE_TO_QUANT[self.output_type]
        try:
            quantized = gguf.quants.quantize(data.astype(np.float32), quant)
            return quantized, quant
        except Exception as exc:
            logger.warning(
                "Quantization failed for %s with %s: %s; falling back to F16",
                tensor_name,
                self.output_type,
                exc,
            )
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16

    def _load_tokenizer(self) -> tuple[list[str], list[int], list[str]]:
        vocab_path = self.input_dir / "vocab.json"
        merges_path = self.input_dir / "merges.txt"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

        vocab_dict = json.loads(vocab_path.read_text(encoding="utf-8"))
        sorted_vocab = sorted(vocab_dict.items(), key=lambda item: item[1])

        tokens: list[str] = []
        toktypes: list[int] = []
        for token, _token_id in sorted_vocab:
            tokens.append(token)
            if token.startswith("<|") and token.endswith("|>"):
                toktypes.append(gguf.TokenType.CONTROL)
            else:
                toktypes.append(gguf.TokenType.NORMAL)

        while len(tokens) < self.text_vocab_size:
            tokens.append(f"[PAD{len(tokens)}]")
            toktypes.append(gguf.TokenType.UNUSED)

        merges: list[str] = []
        if merges_path.exists():
            for line in merges_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    merges.append(line)
        return tokens, toktypes, merges

    def _add_metadata(self, writer: gguf.GGUFWriter) -> None:
        arch = "qwen3-tts"
        writer.add_name(self.model_name)
        writer.add_type(gguf.GGUFType.MODEL)
        writer.add_file_type(MAIN_TYPE_TO_FILE_TYPE[self.output_type])
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

        writer.add_block_count(self.num_hidden_layers)
        writer.add_embedding_length(self.hidden_size)
        writer.add_feed_forward_length(self.intermediate_size)
        writer.add_head_count(self.num_attention_heads)
        writer.add_head_count_kv(self.num_kv_heads)
        writer.add_key_length(self.head_dim)
        writer.add_value_length(self.head_dim)
        writer.add_rope_freq_base(self.rope_theta)
        writer.add_layer_norm_rms_eps(self.rms_norm_eps)
        writer.add_vocab_size(self.vocab_size)

        writer.add_uint32(f"{arch}.text_vocab_size", self.text_vocab_size)
        writer.add_uint32(f"{arch}.text_hidden_size", self.text_hidden_size)
        writer.add_uint32(f"{arch}.num_code_groups", self.num_code_groups)
        writer.add_array(f"{arch}.rope.mrope_section", self.mrope_section)
        writer.add_uint32(f"{arch}.code_predictor.layer_count", self.code_predictor_num_layers)
        writer.add_uint32(f"{arch}.code_predictor.vocab_size", self.code_predictor_vocab_size)
        writer.add_uint32(f"{arch}.speaker_encoder.embedding_length", self.speaker_enc_dim)
        writer.add_uint32(f"{arch}.speaker_encoder.sample_rate", self.speaker_sample_rate)
        writer.add_uint32(f"{arch}.codec.pad_id", self.codec_pad_id)
        writer.add_uint32(f"{arch}.codec.bos_id", self.codec_bos_id)
        writer.add_uint32(f"{arch}.codec.eos_id", self.codec_eos_id)

    def _add_tokenizer(self, writer: gguf.GGUFWriter) -> None:
        tokens, toktypes, merges = self._load_tokenizer()
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("qwen2")
        writer.add_token_list(tokens)
        writer.add_token_types(toktypes)
        if merges:
            writer.add_token_merges(merges)

        tokenizer_config_path = self.input_dir / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
            vocab = json.loads((self.input_dir / "vocab.json").read_text(encoding="utf-8"))

            eos_token = tokenizer_config.get("eos_token")
            if isinstance(eos_token, dict):
                eos_token = eos_token.get("content")
            if eos_token in vocab:
                writer.add_eos_token_id(vocab[eos_token])

            pad_token = tokenizer_config.get("pad_token")
            if isinstance(pad_token, dict):
                pad_token = pad_token.get("content")
            if pad_token in vocab:
                writer.add_pad_token_id(vocab[pad_token])

            chat_template = tokenizer_config.get("chat_template")
            if chat_template:
                writer.add_chat_template(chat_template)

    def export(self) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = gguf.GGUFWriter(path=None, arch="qwen3-tts")
        self._add_metadata(writer)
        self._add_tokenizer(writer)

        tensor_count = 0
        skipped_count = 0
        for hf_name, tensor in tqdm(self._get_tensors(), desc="Converting GGUF"):
            ggml_name = self._map_tensor_name(hf_name)
            if ggml_name is None:
                skipped_count += 1
                logger.debug("Skipping unmapped tensor: %s", hf_name)
                continue
            data, dtype = self._convert_dtype(tensor, ggml_name)
            writer.add_tensor(ggml_name, data, raw_dtype=dtype)
            tensor_count += 1

        logger.info(
            "Prepared GGUF tensors: converted=%s skipped=%s output=%s",
            tensor_count,
            skipped_count,
            self.output_path,
        )
        writer.write_header_to_file(path=self.output_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        return self.output_path


class VocoderOnnxWrapper(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module, decode_upsample_rate: int):
        super().__init__()
        self.decoder = decoder
        self.decode_upsample_rate = int(decode_upsample_rate)

    def forward(self, audio_codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio_lengths = (audio_codes[..., 0] > -1).sum(dim=1) * self.decode_upsample_rate
        clamped = torch.clamp(audio_codes, min=0).to(dtype=torch.long)
        audio_values = self.decoder(clamped.transpose(1, 2)).squeeze(1)
        return audio_values, audio_lengths


class ReferenceCodecEncodeOnnxWrapper(torch.nn.Module):
    """ONNX-exportable slice of `Qwen3TTSTokenizerV2Model.encode` (batch size 1).

    Calls `model.encoder.encode` directly and applies the same length trim as the Python
    `encode` implementation: keep the first ``ceil(sum(padding_mask) / encode_downsample_rate)``
    code frames (see ``code[..., :-(-mask.sum() // d)]`` in ``modeling_qwen3_tts_tokenizer_v2``).
    This avoids list comprehensions / dynamic iteration that confuse the TorchScript exporter.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self._num_quantizers = int(model.encoder_valid_num_quantizers)
        self._encode_downsample = int(model.encode_downsample_rate)

    def forward(self, input_values: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        enc = self.model.encoder.encode(
            input_values.unsqueeze(1),
            return_dict=True,
        )
        # (1, Q, T_code)
        codes = enc.audio_codes[:, : self._num_quantizers, :]
        code = codes[0]
        # Scalar int64; match Python floor-division sign rule for the trim length.
        # Align with `codes[0]`: only batch index 0 (Rust and this export use batch == 1).
        length = padding_mask[0].sum()
        n_frames = -((-length) // self._encode_downsample)
        code = code[..., :n_frames]
        return code.transpose(0, 1).to(torch.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face repo id or local model directory.")
    parser.add_argument("--out-dir", required=True, help="Directory to write exported artifacts into.")
    parser.add_argument(
        "--main-type",
        action="append",
        default=None,
        help=(
            "Quantization/data type for the main GGUF export. "
            "Repeat the flag or pass a comma-separated list to export multiple GGUF variants."
        ),
    )
    parser.add_argument("--main-out", default=None, help="Override the main GGUF output path.")
    parser.add_argument("--vocoder-out", default=None, help="Override the vocoder ONNX output path.")
    parser.add_argument(
        "--ref-codec-out",
        default=None,
        help="Override the reference codec (speech tokenizer encode) ONNX output path.",
    )
    parser.add_argument(
        "--vocoder-dtype",
        default="float32",
        help="dtype used when loading the PyTorch vocoder before ONNX export.",
    )
    parser.add_argument(
        "--vocoder-opset",
        type=int,
        default=17,
        help="ONNX opset version for vocoder export.",
    )
    parser.add_argument(
        "--ref-codec-dtype",
        default="float32",
        help="dtype used when loading the PyTorch speech tokenizer for reference-codec ONNX export.",
    )
    parser.add_argument(
        "--ref-codec-opset",
        type=int,
        default=17,
        help="ONNX opset version for reference codec export.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download from Hugging Face; require all files to already exist locally.",
    )
    parser.add_argument(
        "--skip-reference-codec",
        action="store_true",
        help=(
            "Skip qwen3-tts-reference-codec.onnx and preprocess JSON (GGUF + vocoder only). "
            "Use when ONNX export fails on your PyTorch/transformers stack; ICL voice clone "
            "needs the reference codec artifacts from a machine where export succeeds."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def validate_main_type(main_type: str) -> None:
    if main_type not in SUPPORTED_MAIN_TYPES:
        choices = ", ".join(SUPPORTED_MAIN_TYPES)
        raise SystemExit(
            f"Unknown --main-type {main_type!r}. Valid values: {choices}."
        )


def resolve_main_types(raw_values: list[str] | None) -> list[str]:
    if not raw_values:
        return ["f16"]

    resolved: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in raw.split(","):
            main_type = part.strip()
            if not main_type:
                continue
            validate_main_type(main_type)
            if main_type not in seen:
                resolved.append(main_type)
                seen.add(main_type)

    if not resolved:
        raise SystemExit("At least one non-empty --main-type must be provided.")
    return resolved


def resolve_model_dir(model_name_or_path: str, local_files_only: bool) -> Path:
    path = Path(model_name_or_path).expanduser()
    if path.exists():
        return path.resolve()
    snapshot = snapshot_download(
        repo_id=model_name_or_path,
        allow_patterns=MODEL_ALLOW_PATTERNS,
        local_files_only=local_files_only,
    )
    return Path(snapshot)


def default_main_output(out_dir: Path, main_type: str) -> Path:
    return out_dir / f"qwen3-tts-0.6b-{main_type}.gguf"


def default_vocoder_output(out_dir: Path) -> Path:
    return out_dir / "qwen3-tts-vocoder.onnx"


def default_reference_codec_output(out_dir: Path) -> Path:
    return out_dir / "qwen3-tts-reference-codec.onnx"


def default_reference_codec_preprocess_output(out_dir: Path) -> Path:
    return out_dir / "qwen3-tts-reference-codec-preprocess.json"


def add_onnx_metadata(
    onnx_path: Path,
    *,
    source_model: str,
    speech_tokenizer_dir: Path,
    num_quantizers: int,
    decode_upsample_rate: int,
    output_sample_rate: int,
) -> None:
    # Load external tensor data eagerly so we can re-save the model as a single
    # self-contained ONNX file after attaching metadata.
    model = onnx.load(str(onnx_path), load_external_data=True)
    metadata = {
        "source_model": source_model,
        "speech_tokenizer_dir": str(speech_tokenizer_dir),
        "num_quantizers": str(num_quantizers),
        "decode_upsample_rate": str(decode_upsample_rate),
        "output_sample_rate_hz": str(output_sample_rate),
        "input_layout": "batch,frames,quantizers",
        "output_layout": "batch,samples",
    }
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    # Keep the vocoder artifact self-contained so one shared ONNX can be uploaded
    # alongside multiple GGUF variants without a duplicate .onnx.data payload.
    onnx.save_model(model, str(onnx_path), save_as_external_data=False)


def cleanup_stale_onnx_external_data(onnx_path: Path) -> None:
    data_path = Path(f"{onnx_path}.data")
    if not data_path.exists():
        return

    model = onnx.load(str(onnx_path), load_external_data=False)
    has_external_initializers = any(
        tensor.data_location == onnx.TensorProto.EXTERNAL
        for tensor in model.graph.initializer
    )
    if has_external_initializers:
        logger.info("Keeping external ONNX tensor data: %s", data_path)
        return

    data_path.unlink()
    logger.info("Removed stale ONNX external data file: %s", data_path)


def normalize_onnx_to_single_file(onnx_path: Path) -> None:
    data_path = Path(f"{onnx_path}.data")
    if not data_path.exists():
        return

    model = onnx.load(str(onnx_path), load_external_data=False)
    has_external_initializers = any(
        tensor.data_location == onnx.TensorProto.EXTERNAL
        for tensor in model.graph.initializer
    )
    if not has_external_initializers:
        cleanup_stale_onnx_external_data(onnx_path)
        return

    logger.info("Repacking ONNX external tensor data into %s", onnx_path)
    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save_model(model, str(onnx_path), save_as_external_data=False)
    cleanup_stale_onnx_external_data(onnx_path)


def export_vocoder_onnx(
    *,
    speech_tokenizer_dir: Path,
    output_path: Path,
    source_model: str,
    dtype_name: str,
    opset: int,
) -> Path:
    tokenizer = _qwen3_tts_tokenizer_cls().from_pretrained(
        str(speech_tokenizer_dir),
        device_map="cpu",
        dtype=resolve_dtype(dtype_name),
        attn_implementation="eager",
    )
    model = tokenizer.model
    model.eval()
    if model.get_model_type() != "qwen3_tts_tokenizer_12hz":
        raise SystemExit(
            f"Only the 12Hz speech tokenizer is supported for ONNX export, got {model.get_model_type()}"
        )

    num_quantizers = int(model.config.decoder_config.num_quantizers)
    decode_upsample_rate = int(model.get_decode_upsample_rate())
    output_sample_rate = int(model.get_output_sample_rate())
    wrapper = VocoderOnnxWrapper(model.decoder, decode_upsample_rate).cpu().eval()
    dummy_codes = torch.zeros((1, 16, num_quantizers), dtype=torch.long)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_codes,),
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
        input_names=["audio_codes"],
        output_names=["audio_values", "audio_lengths"],
        dynamic_axes={
            "audio_codes": {0: "batch", 1: "frames"},
            "audio_values": {0: "batch", 1: "samples"},
            "audio_lengths": {0: "batch"},
        },
    )
    add_onnx_metadata(
        output_path,
        source_model=source_model,
        speech_tokenizer_dir=speech_tokenizer_dir,
        num_quantizers=num_quantizers,
        decode_upsample_rate=decode_upsample_rate,
        output_sample_rate=output_sample_rate,
    )
    cleanup_stale_onnx_external_data(output_path)
    logger.info(
        "Exported vocoder ONNX: path=%s num_quantizers=%s decode_upsample_rate=%s",
        output_path,
        num_quantizers,
        decode_upsample_rate,
    )
    return output_path


def add_reference_codec_onnx_metadata(
    onnx_path: Path,
    *,
    source_model: str,
    speech_tokenizer_dir: Path,
    num_quantizers: int,
    encode_downsample_rate: int,
    input_sample_rate: int,
) -> None:
    model = onnx.load(str(onnx_path), load_external_data=True)
    metadata = {
        "artifact": "qwen3_tts_reference_codec_encode",
        "source_model": source_model,
        "speech_tokenizer_dir": str(speech_tokenizer_dir),
        "num_quantizers": str(num_quantizers),
        "encode_downsample_rate": str(encode_downsample_rate),
        "input_sample_rate_hz": str(input_sample_rate),
        "input_names": "input_values,padding_mask",
        "output_names": "audio_codes",
        "output_layout": "code_frames,num_quantizers",
    }
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    onnx.save_model(model, str(onnx_path), save_as_external_data=False)


def write_reference_codec_preprocess_json(
    *,
    speech_tokenizer_dir: Path,
    output_path: Path,
    num_quantizers: int,
    encode_downsample_rate: int,
    onnx_trace_audio_len: int,
) -> None:
    src = speech_tokenizer_dir / "preprocessor_config.json"
    if not src.exists():
        raise FileNotFoundError(f"Missing preprocessor config for speech tokenizer: {src}")
    data = json.loads(src.read_text(encoding="utf-8"))
    data["num_quantizers"] = int(num_quantizers)
    data["encode_downsample_rate"] = int(encode_downsample_rate)
    # TorchScript ONNX trace is specialized to this `input_values` time dimension; ORT
    # inference must use the same length after resampling (see Rust `ReferenceCodecEncoder`).
    data["onnx_trace_audio_len"] = int(onnx_trace_audio_len)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Wrote reference codec preprocess JSON: %s", output_path)


def export_reference_codec_onnx(
    *,
    speech_tokenizer_dir: Path,
    output_path: Path,
    preprocess_json_path: Path,
    source_model: str,
    dtype_name: str,
    opset: int,
) -> Path:
    tokenizer = _qwen3_tts_tokenizer_cls().from_pretrained(
        str(speech_tokenizer_dir),
        device_map="cpu",
        dtype=resolve_dtype(dtype_name),
        attn_implementation="eager",
    )
    model = tokenizer.model
    model.eval()
    if model.get_model_type() != "qwen3_tts_tokenizer_12hz":
        raise SystemExit(
            f"Only the 12Hz speech tokenizer is supported for reference codec export, got {model.get_model_type()}"
        )

    num_quantizers = int(model.encoder_valid_num_quantizers)
    encode_downsample_rate = int(model.get_encode_downsample_rate())
    input_sr = int(model.get_input_sample_rate())
    # Must match the ONNX trace example; ORT is not shape-general for this graph.
    trace_audio_len = max(400, input_sr // 2)

    write_reference_codec_preprocess_json(
        speech_tokenizer_dir=speech_tokenizer_dir,
        output_path=preprocess_json_path,
        num_quantizers=num_quantizers,
        encode_downsample_rate=encode_downsample_rate,
        onnx_trace_audio_len=trace_audio_len,
    )

    wrapper = ReferenceCodecEncodeOnnxWrapper(model).cpu().eval()
    half_sec = trace_audio_len
    dummy_iv = torch.zeros((1, half_sec), dtype=torch.float32)
    dummy_mask = torch.ones((1, half_sec), dtype=torch.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _transformers_drop_packed_sequence_mask_for_onnx_trace(), _patch_torch_cdist_p2_for_onnx_trace():
        torch.onnx.export(
            wrapper,
            (dummy_iv, dummy_mask),
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
            input_names=["input_values", "padding_mask"],
            output_names=["audio_codes"],
            dynamic_axes={
                "input_values": {0: "batch", 1: "audio_len"},
                "padding_mask": {0: "batch", 1: "audio_len"},
                "audio_codes": {0: "code_frames", 1: "num_quantizers"},
            },
        )
    add_reference_codec_onnx_metadata(
        output_path,
        source_model=source_model,
        speech_tokenizer_dir=speech_tokenizer_dir,
        num_quantizers=num_quantizers,
        encode_downsample_rate=encode_downsample_rate,
        input_sample_rate=input_sr,
    )
    cleanup_stale_onnx_external_data(output_path)
    logger.info(
        "Exported reference codec ONNX: path=%s num_quantizers=%s encode_downsample_rate=%s",
        output_path,
        num_quantizers,
        encode_downsample_rate,
    )
    return output_path


def main() -> None:
    configure_stdio()
    args = parse_args()
    configure_logging(args.verbose)
    main_types = resolve_main_types(args.main_type)

    model_dir = resolve_model_dir(args.model, local_files_only=args.local_files_only)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.main_out and len(main_types) != 1:
        raise SystemExit("--main-out can only be used when exporting exactly one --main-type.")

    vocoder_out = Path(args.vocoder_out).expanduser().resolve() if args.vocoder_out else default_vocoder_output(out_dir)
    ref_codec_out = (
        Path(args.ref_codec_out).expanduser().resolve()
        if args.ref_codec_out
        else default_reference_codec_output(out_dir)
    )
    ref_preprocess_out = default_reference_codec_preprocess_output(out_dir)

    logger.info("Resolved model dir: %s", model_dir)
    logger.info("Main GGUF types: %s", ", ".join(main_types))
    logger.info("Vocoder ONNX output: %s", vocoder_out)
    if args.skip_reference_codec:
        logger.info("Reference codec export: skipped (--skip-reference-codec)")
    else:
        logger.info("Reference codec ONNX output: %s", ref_codec_out)

    gguf_outputs: list[Path] = []
    for main_type in main_types:
        main_out = (
            Path(args.main_out).expanduser().resolve()
            if args.main_out
            else default_main_output(out_dir, main_type)
        )
        logger.info("Main GGUF output for %s: %s", main_type, main_out)
        if main_out.exists():
            logger.info("Reusing existing main GGUF: %s", main_out)
            gguf_out = main_out
        else:
            gguf_out = Qwen3MainGgufExporter(
                input_dir=model_dir,
                output_path=main_out,
                output_type=main_type,
            ).export()
        gguf_outputs.append(gguf_out)

    if vocoder_out.exists():
        logger.info("Reusing existing vocoder ONNX: %s", vocoder_out)
        normalize_onnx_to_single_file(vocoder_out)
        onnx_out = vocoder_out
    else:
        onnx_out = export_vocoder_onnx(
            speech_tokenizer_dir=model_dir / "speech_tokenizer",
            output_path=vocoder_out,
            source_model=args.model,
            dtype_name=args.vocoder_dtype,
            opset=args.vocoder_opset,
        )

    speech_tok_dir = model_dir / "speech_tokenizer"
    ref_onnx_out: Path | None
    if args.skip_reference_codec:
        ref_onnx_out = None
    elif ref_codec_out.exists():
        logger.info("Reusing existing reference codec ONNX: %s", ref_codec_out)
        normalize_onnx_to_single_file(ref_codec_out)
        ref_onnx_out = ref_codec_out
        if not ref_preprocess_out.exists():
            tokenizer = _qwen3_tts_tokenizer_cls().from_pretrained(
                str(speech_tok_dir),
                device_map="cpu",
                dtype=resolve_dtype(args.ref_codec_dtype),
                attn_implementation="eager",
            )
            m = tokenizer.model
            isr = int(m.get_input_sample_rate())
            write_reference_codec_preprocess_json(
                speech_tokenizer_dir=speech_tok_dir,
                output_path=ref_preprocess_out,
                num_quantizers=int(m.encoder_valid_num_quantizers),
                encode_downsample_rate=int(m.get_encode_downsample_rate()),
                onnx_trace_audio_len=max(400, isr // 2),
            )
    else:
        ref_onnx_out = export_reference_codec_onnx(
            speech_tokenizer_dir=speech_tok_dir,
            output_path=ref_codec_out,
            preprocess_json_path=ref_preprocess_out,
            source_model=args.model,
            dtype_name=args.ref_codec_dtype,
            opset=args.ref_codec_opset,
        )

    if ref_onnx_out is not None:
        ref_summary = (
            f"reference_codec_onnx={ref_onnx_out} reference_codec_preprocess={ref_preprocess_out}"
        )
    else:
        ref_summary = "reference_codec_onnx=skipped reference_codec_preprocess=skipped"
    print(
        f"exported artifacts: main_ggufs={','.join(str(path) for path in gguf_outputs)} "
        f"vocoder_onnx={onnx_out} {ref_summary} main_types={','.join(main_types)}"
    )


if __name__ == "__main__":
    main()
