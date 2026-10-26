"""Unified experiment planning without implicit execution."""

from __future__ import annotations

from pathlib import Path
import shlex
from typing import Iterable

from contracts import RunManifest


def plan(run_id: str, family: str, commands: dict[str, Iterable[str]], *, output: str | Path) -> Path:
    manifest = RunManifest(
        run_id=run_id,
        experiment_family=family,
        command=[shlex.join(list(value)) for value in commands.values()],
        configuration={"stages": list(commands)},
        provenance={"workspace": str(Path.cwd().resolve()), "execution": "planned; no subprocess started"},
        gates={"schema": None, "dataset": None, "offline": None, "runtime": None, "physical_task": None},
        outputs=[str(Path(output).resolve())],
    )
    return manifest.write(Path(output) / "run_manifest.json")
