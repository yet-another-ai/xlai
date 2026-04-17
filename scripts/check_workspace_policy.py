#!/usr/bin/env python3
"""Validate workspace layout and publish policy against repo docs and CI.

Run from repo root: python3 scripts/check_workspace_policy.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALLOWED_MEMBER_PREFIXES = (
    "crates/core/",
    "crates/runtime/",
    "crates/backends/",
    "crates/qts/",
    "crates/sys/",
    "crates/platform/",
)


def load_toml(path: Path) -> dict:
    import tomllib

    with path.open("rb") as f:
        return tomllib.load(f)


def markdown_section(md: str, heading: str, until: str | None) -> str:
    i = md.find(heading)
    if i == -1:
        return ""
    start = i + len(heading)
    if until is None:
        return md[start:]
    j = md.find(until, start)
    if j == -1:
        return md[start:]
    return md[start:j]


def crate_names_in_table_column(md_chunk: str) -> list[str]:
    return re.findall(r"^\|\s*`(xlai-[^`]+)`", md_chunk, flags=re.MULTILINE)


def publish_order_from_publishing_md() -> list[str]:
    text = (ROOT / "docs/development/publishing.md").read_text(encoding="utf-8")
    section_start = text.find("### Rust (crates.io)")
    if section_start == -1:
        return []
    chunk = text[section_start : section_start + 900]
    return re.findall(r"^\d+\.\s*`(xlai-[^`]+)`", chunk, flags=re.MULTILINE)


def publish_order_from_publish_yml() -> list[str]:
    lines = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith("  PUBLISH_CRATES_ORDER:"):
            names: list[str] = []
            j = i + 1
            while j < len(lines) and lines[j].startswith("    "):
                names.extend(lines[j].split())
                j += 1
            return names
    return []


def main() -> int:
    errors: list[str] = []

    root_data = load_toml(ROOT / "Cargo.toml")
    members: list[str] = root_data["workspace"]["members"]

    for m in members:
        if not any(m.startswith(p) for p in ALLOWED_MEMBER_PREFIXES):
            errors.append(
                f"workspace member {m!r} is not under an allowed prefix {ALLOWED_MEMBER_PREFIXES}"
            )

    taxonomy = (ROOT / "docs/development/crates-taxonomy.md").read_text(encoding="utf-8")
    pub_chunk = markdown_section(
        taxonomy,
        "## Published to crates.io",
        "## Internal only (`publish = false`)",
    )
    internal_chunk = markdown_section(
        taxonomy,
        "## Internal only (`publish = false`)",
        "## CI setup",
    )
    published_names = crate_names_in_table_column(pub_chunk)
    internal_names = crate_names_in_table_column(internal_chunk)

    if not published_names:
        errors.append("could not parse published crate names from crates-taxonomy.md")
    if not internal_names:
        errors.append("could not parse internal crate names from crates-taxonomy.md")

    order_md = publish_order_from_publishing_md()
    order_yml = publish_order_from_publish_yml()
    if order_md != order_yml:
        errors.append(
            "publish order mismatch:\n"
            f"  docs/development/publishing.md: {order_md}\n"
            f"  .github/workflows/publish.yml:  {order_yml}"
        )

    if order_md and published_names and order_md != published_names:
        errors.append(
            "publish order / taxonomy published table mismatch:\n"
            f"  publishing.md order: {order_md}\n"
            f"  taxonomy table:       {published_names}"
        )

    pkg_paths: dict[str, str] = {}
    for rel in members:
        data = load_toml(ROOT / rel / "Cargo.toml")
        name = data["package"]["name"]
        pkg_paths[name] = rel

    taxonomy_all = set(published_names) | set(internal_names)
    for name, rel in pkg_paths.items():
        if name not in taxonomy_all:
            errors.append(
                f"crate {name!r} ({rel}) is not listed in docs/development/crates-taxonomy.md"
            )

    for name, rel in pkg_paths.items():
        data = load_toml(ROOT / rel / "Cargo.toml")
        publish = data["package"].get("publish", True)
        is_unpublished = publish is False

        if name in published_names and is_unpublished:
            errors.append(f"published crate {name!r} has publish = false in {rel}/Cargo.toml")
        if name in internal_names and not is_unpublished:
            errors.append(
                f"internal crate {name!r} must set publish = false in {rel}/Cargo.toml"
            )

    internal_set = set(internal_names)
    for pub in published_names:
        if pub not in pkg_paths:
            continue
        data = load_toml(ROOT / pkg_paths[pub] / "Cargo.toml")
        for section in ("dependencies", "build-dependencies", "dev-dependencies"):
            deps = data.get(section, {})
            for dep_name, spec in deps.items():
                if dep_name in ("xlai-core", "xlai-runtime"):
                    continue
                if isinstance(spec, dict) and spec.get("workspace") is True:
                    if dep_name in internal_set:
                        errors.append(
                            f"published crate {pub!r} must not depend on internal {dep_name!r} "
                            f"({section})"
                        )

    if errors:
        print("workspace policy check failed:\n", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("workspace policy check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
