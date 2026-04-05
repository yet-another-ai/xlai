"""Torch dtype helpers shared by ONNX export scripts."""

from __future__ import annotations

from typing import Any


def resolve_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise SystemExit(
            f"unknown dtype {name!r} (expected one of: auto, float16, bfloat16, float32)"
        ) from exc
