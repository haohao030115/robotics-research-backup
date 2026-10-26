"""Pure command builders for the canonical data factory."""

from __future__ import annotations

from pathlib import Path
import sys


def dataset_command(canonical_root: str | Path, output: str | Path, *, val_count: int | None = None, seed: int | None = None, all_train: bool = False) -> list[str]:
    command = [
        sys.executable,
        "-B",
        "-m",
        "data_factory.dataset.cli",
        "dataset",
        "build",
        "--canonical-root",
        str(Path(canonical_root).resolve()),
        "--output-root",
        str(Path(output).resolve()),
    ]
    if all_train and val_count is not None:
        raise ValueError("all_train and val_count are mutually exclusive")
    if val_count is not None:
        if seed is None:
            raise ValueError("seed is required with val_count")
        command += ["--val-count", str(val_count), "--seed", str(seed)]
    elif all_train:
        command += ["--all-train"]
    else:
        raise ValueError("an explicit split is required: all_train or val_count plus seed")
    return command
