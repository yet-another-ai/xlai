#!/usr/bin/env python3
"""Prepare a Hugging Face release directory for Qwen3-TTS QTS artifacts (xlai).

Runs ``uv run export-model-artifacts`` (or uses existing files with ``--skip-export``),
then writes ``README.md`` from a template, ``SHA256SUMS``, and ``.gitattributes`` (Xet LFS
for ``.gguf`` / ``.onnx``), optionally syncing into a cloned HF git repo root.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Repository root (this file lives under ``scripts/qts/``)."""
    return Path(__file__).resolve().parent.parent.parent


def _default_model_dir(root: Path) -> Path:
    return root / "models" / "qwen3-tts-bundle"


def _release_main_types() -> tuple[str, ...]:
    return ("f16", "q8_0")


def _resolve_release_main_types(raw_values: list[str]) -> list[str]:
    supported = _release_main_types()
    if not raw_values:
        return list(supported)

    resolved: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in raw.split(","):
            main_type = part.strip()
            if not main_type:
                continue
            if main_type not in supported:
                choices = ", ".join(supported)
                raise SystemExit(
                    f"unknown --main-type {main_type!r}. Valid values: {choices}."
                )
            if main_type not in seen:
                resolved.append(main_type)
                seen.add(main_type)

    if not resolved:
        raise SystemExit("At least one non-empty --main-type must be provided.")
    return resolved


def _git_rev_parse_head(cwd: Path) -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def _ensure_git_repo_root(dir_path: Path) -> None:
    if not dir_path.is_dir():
        raise SystemExit(f"hf repo dir does not exist: {dir_path}")
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=dir_path,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"hf repo dir is not a git repository: {dir_path}")
    top = Path(out.stdout.strip()).resolve()
    if dir_path.resolve() != top:
        raise SystemExit(
            f"--hf-repo-dir must point at the repository root, got {dir_path}"
        )


def _same_canonical(left: Path, right: Path) -> bool:
    if not left.exists() or not right.exists():
        return left == right
    return left.resolve() == right.resolve()


def _is_managed_release_file_name(name: str) -> bool:
    if name in (
        ".gitattributes",
        "README.md",
        "SHA256SUMS",
        "qwen3-tts-vocoder.onnx",
        "qwen3-tts-reference-codec.onnx",
        "qwen3-tts-reference-codec-preprocess.json",
    ):
        return True
    return name.startswith("qwen3-tts-0.6b-") and name.endswith(".gguf")


def _remove_managed_release_files(dir_path: Path) -> None:
    for entry in dir_path.iterdir():
        if entry.is_file() and _is_managed_release_file_name(entry.name):
            entry.unlink()


def _collect_release_artifacts(dir_path: Path) -> list[Path]:
    artifacts: list[Path] = []
    for path in dir_path.iterdir():
        if not path.is_file():
            continue
        n = path.name
        if n == "qwen3-tts-vocoder.onnx":
            artifacts.append(path)
        elif n == "qwen3-tts-reference-codec.onnx":
            artifacts.append(path)
        elif n == "qwen3-tts-reference-codec-preprocess.json":
            artifacts.append(path)
        elif n.startswith("qwen3-tts-0.6b-") and n.endswith(".gguf"):
            artifacts.append(path)
    artifacts.sort(key=lambda p: p.name)

    if not artifacts:
        raise SystemExit(f"no release artifacts found in {dir_path}")
    if not any(p.name == "qwen3-tts-vocoder.onnx" for p in artifacts):
        raise SystemExit(f"missing qwen3-tts-vocoder.onnx in {dir_path}")
    if not any(p.suffix == ".gguf" for p in artifacts):
        raise SystemExit(f"missing qwen3-tts-0.6b-*.gguf in {dir_path}")
    if not any(p.name == "qwen3-tts-reference-codec.onnx" for p in artifacts):
        raise SystemExit(f"missing qwen3-tts-reference-codec.onnx in {dir_path}")
    if not any(p.name == "qwen3-tts-reference-codec-preprocess.json" for p in artifacts):
        raise SystemExit(
            f"missing qwen3-tts-reference-codec-preprocess.json in {dir_path}"
        )
    return artifacts


def _sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _render_sha256sums(rows: list[tuple[str, str]]) -> str:
    lines = [f"{digest}  {name}\n" for name, digest in rows]
    return "".join(lines)


def _quantization_name(path: Path) -> str | None:
    name = path.name
    if not name.startswith("qwen3-tts-0.6b-") or not name.endswith(".gguf"):
        return None
    return name.removeprefix("qwen3-tts-0.6b-").removesuffix(".gguf")


def _render_hf_model_card(
    template: str,
    source_commit: str,
    copied_files: list[Path],
    quantizations: list[str],
    checksums: list[tuple[str, str]],
) -> str:
    root_layout = "\n".join(p.name for p in copied_files)
    quantization_list = "\n".join(f"- `{q}`" for q in quantizations)
    checksum_list = "\n".join(
        f"- `{name}`\n  `{digest}`" for name, digest in checksums
    )
    return (
        template.replace("{{SOURCE_COMMIT}}", source_commit)
        .replace("{{ROOT_LAYOUT}}", root_layout)
        .replace("{{QUANTIZATION_LIST}}", quantization_list)
        .replace("{{CHECKSUM_LIST}}", checksum_list)
    )


def _hf_xet_gitattributes() -> str:
    return (
        "*.gguf filter=lfs diff=lfs merge=lfs -text\n"
        "*.onnx filter=lfs diff=lfs merge=lfs -text\n"
    )


def _run_export_model_artifacts(
    repo_root: Path,
    model: str,
    artifacts_dir: Path,
    main_types: list[str],
    local_files_only: bool,
    verbose: bool,
) -> None:
    cmd = [
        "uv",
        "run",
        "export-model-artifacts",
        "--model",
        model,
        "--out-dir",
        str(artifacts_dir),
    ]
    for mt in main_types:
        cmd.extend(["--main-type", mt])
    if local_files_only:
        cmd.append("--local-files-only")
    if verbose:
        cmd.append("--verbose")

    print(
        f"exporting release artifacts: model={model} out_dir={artifacts_dir}",
        file=sys.stderr,
    )
    r = subprocess.run(cmd, cwd=repo_root)
    if r.returncode != 0:
        raise SystemExit("uv run export-model-artifacts failed")


def _sync_release_to_hf_repo(staged_release_dir: Path, hf_repo_dir: Path) -> None:
    _ensure_git_repo_root(hf_repo_dir)
    staged = staged_release_dir.resolve()
    hf_root = hf_repo_dir.resolve()
    if staged == hf_root:
        return
    _remove_managed_release_files(hf_root)
    for path in staged.iterdir():
        if path.is_file():
            shutil.copy2(path, hf_root / path.name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="Upstream HF id or local dir (required unless --skip-export).")
    p.add_argument(
        "--main-type",
        action="append",
        default=None,
        help="GGUF variant (repeat or comma-separated). Default: f16 and q8_0.",
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Directory export-model-artifacts writes into (default: models/qwen3-tts-bundle).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Prepared release directory (default: target/hf-qts-release).",
    )
    p.add_argument(
        "--readme-template",
        type=Path,
        default=None,
        help="Markdown template with {{SOURCE_COMMIT}}, {{ROOT_LAYOUT}}, etc.",
    )
    p.add_argument("--source-commit", default=None, help="Override provenance commit SHA.")
    p.add_argument(
        "--hf-repo-dir",
        type=Path,
        default=None,
        help="Optional cloned Hugging Face model repo root to sync managed files into.",
    )
    p.add_argument(
        "--local-files-only",
        action="store_true",
        help="Pass through to export-model-artifacts.",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="Package artifacts already present in --artifacts-dir.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()

    artifacts_dir = (
        args.artifacts_dir.expanduser().resolve()
        if args.artifacts_dir
        else _default_model_dir(root)
    )
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else (root / "target" / "hf-qts-release")
    )
    readme_template = (
        args.readme_template.expanduser().resolve()
        if args.readme_template
        else (root / "docs" / "huggingface-qts-model-card.md")
    )

    main_types = _resolve_release_main_types(args.main_type or [])

    if not args.skip_export and not args.model:
        raise SystemExit("--model is required unless --skip-export is used")

    hf_repo_dir = args.hf_repo_dir.expanduser().resolve() if args.hf_repo_dir else None
    artifacts_dir_explicit = args.artifacts_dir is not None
    out_dir_explicit = args.out_dir is not None

    if hf_repo_dir is not None:
        _ensure_git_repo_root(hf_repo_dir)
        if not artifacts_dir_explicit and not out_dir_explicit:
            artifacts_dir = hf_repo_dir
            out_dir = hf_repo_dir
        if (
            not args.skip_export
            and _same_canonical(artifacts_dir, hf_repo_dir)
        ):
            _remove_managed_release_files(hf_repo_dir)

    if not args.skip_export:
        _run_export_model_artifacts(
            root,
            args.model,
            artifacts_dir,
            main_types,
            args.local_files_only,
            args.verbose,
        )

    source_commit = args.source_commit or _git_rev_parse_head(root)
    artifacts = _collect_release_artifacts(artifacts_dir)
    gguf_files = [p for p in artifacts if p.suffix == ".gguf"]
    quantizations = [q for p in gguf_files if (q := _quantization_name(p))]

    package_in_place = _same_canonical(artifacts_dir, out_dir)
    if package_in_place:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for a in artifacts:
            shutil.copy2(a, out_dir / a.name)

    (out_dir / ".gitattributes").write_text(_hf_xet_gitattributes(), encoding="utf-8")

    copied = _collect_release_artifacts(out_dir)
    checksums = [(p.name, _sha256_hex(p)) for p in copied]
    (out_dir / "SHA256SUMS").write_text(
        _render_sha256sums(checksums), encoding="utf-8"
    )

    template = readme_template.read_text(encoding="utf-8")
    readme = _render_hf_model_card(
        template, source_commit, copied, quantizations, checksums
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    if hf_repo_dir is not None:
        _sync_release_to_hf_repo(out_dir, hf_repo_dir)
        if package_in_place:
            print(
                f"prepared release files directly in git repo: {hf_repo_dir}",
                file=sys.stderr,
            )
        else:
            print(
                f"synced prepared release files into git repo: {hf_repo_dir}",
                file=sys.stderr,
            )

    print(
        f"prepared Hugging Face release directory: {out_dir}\n"
        f"artifacts from: {artifacts_dir}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
