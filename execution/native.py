"""Build a native canonical command without starting it."""

from __future__ import annotations

from pathlib import Path
import sys


def canonical_command(reference: str | Path, output: str | Path, *, gpu: int = 0, mode: str = "production") -> list[str]:
    if mode not in {"audit", "production"}:
        raise ValueError("mode must be audit or production")
    # Native DDS traffic must run in the same per-process isolated namespace
    # used by the validated batch runner.  Keep the Python pipeline itself
    # unchanged; only its process boundary is wrapped here.
    return [
        "unshare",
        "--net",
        "bash",
        "-c",
        "ip link set lo up && exec \"$@\"",
        "canonical-data-factory",
        sys.executable,
        "-B",
        "-m",
        "data_factory.runtime.pipeline",
        "--reference",
        str(Path(reference).resolve()),
        "--output",
        str(Path(output).resolve()),
        "--backend",
        "native",
        "--gpu",
        str(gpu),
        "--mode",
        mode,
    ]
