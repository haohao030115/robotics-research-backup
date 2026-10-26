#!/usr/bin/env python3
"""Create a metadata-only inventory of every workspace entry.

The inventory deliberately does not hash large artifacts or import code. It
records every reachable file/directory/symlink and every permission error so
cleanup decisions remain reversible and auditable.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import stat
from pathlib import Path


SOURCE_ROOTS = {
    "contracts", "research", "data_factory", "execution", "va_research",
    "experiments", "tests", "scripts", "docs",
}
COMPAT_ROOTS = {"canonical_va", "grasp_pipeline", "va_training", "dl-group2"}
EXTERNAL_ROOTS = {"kimodo-t6", "GR00T-WholeBodyControl"}
ARTIFACT_ROOTS = {"outputs", "expert_datasets", "grasp_workspace"}
HISTORY_ROOTS = {"archive", "workspace", "handoffs"}
ENV_ROOTS = {"miniconda3", "torch_musa_post1", "ardy_musa_env", "kimodo_musa_env"}


def category(relative: str) -> str:
    first = relative.split("/", 1)[0]
    if first.startswith("."):
        return "tooling_runtime"
    if first in SOURCE_ROOTS:
        return "maintained_source"
    if first in COMPAT_ROOTS:
        return "compatibility_repository"
    if first in EXTERNAL_ROOTS:
        return "external_repository"
    if first in ARTIFACT_ROOTS:
        return "generated_artifacts"
    if first in HISTORY_ROOTS:
        return "historical_archive"
    if first in ENV_ROOTS:
        return "runtime_environment"
    if first in {"AGENTS.md", "README.md", "config.yaml"}:
        return "workspace_control"
    return "unclassified"


def record(root: Path, path: Path, kind: str, error: str | None = None) -> dict:
    rel = path.relative_to(root).as_posix()
    item = {"path": rel, "kind": kind, "category": category(rel)}
    try:
        st = path.lstat()
        item.update({
            "size": st.st_size,
            "mode": stat.filemode(st.st_mode),
            "uid": st.st_uid,
            "gid": st.st_gid,
        })
        if kind == "symlink":
            item["target"] = os.readlink(path)
    except OSError as exc:
        item["error"] = f"{type(exc).__name__}: {exc}"
    if error:
        item["error"] = error
    return item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    errors: list[dict] = []
    with gzip.open(args.output, "wt", encoding="utf-8") as stream:
        for current, dirs, files in os.walk(root, topdown=True, onerror=lambda e: errors.append({"path": str(e.filename), "error": str(e)}), followlinks=False):
            current_path = Path(current)
            dirs.sort()
            files.sort()
            for name in dirs + files:
                path = current_path / name
                try:
                    mode = path.lstat().st_mode
                    kind = "symlink" if stat.S_ISLNK(mode) else ("directory" if stat.S_ISDIR(mode) else "file")
                    item = record(root, path, kind)
                except OSError as exc:
                    item = record(root, path, "unknown", f"{type(exc).__name__}: {exc}")
                    errors.append(item)
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                key = item.get("category", "unknown")
                counts[key] = counts.get(key, 0) + 1
        for item in errors:
            stream.write(json.dumps({"kind": "walk_error", **item}, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "root": str(root),
        "record_format": "JSONL gzip; one metadata record per workspace entry",
        "counts": counts,
        "walk_errors": len(errors),
        "output": str(args.output),
        "content_hashes": False,
    }
    summary_path = args.output.with_suffix("").with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
