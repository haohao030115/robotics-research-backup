"""Execute delegated research workflows with immutable run bookkeeping.

The runner is intentionally small: existing validated scripts remain the
algorithmic implementation, while this module owns configuration resolution,
stage ordering, logs, provenance and failure reporting.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable

from contracts import RunManifest


class WorkflowError(RuntimeError):
    """A workflow stage failed or produced an invalid run artifact."""


def _env(root: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if extra:
        env.update(extra)
    # Apply workflow-specific additions first, then prepend the workspace
    # root.  ``experiments.__main__`` supplies a family-specific PYTHONPATH;
    # updating it after the root insertion previously hid the ``research``
    # package from unified adapter exports.
    entries = [str(root)]
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def execute_stages(
    *,
    run_id: str,
    family: str,
    commands: dict[str, Iterable[str]],
    output: str | Path,
    root: str | Path,
    cwd: str | Path | None = None,
    environment: dict[str, str] | None = None,
    start_at: str | None = None,
) -> int:
    """Execute stages in order, retaining logs and a structured failure.

    Output directories are immutable: an existing run manifest causes an
    immediate failure rather than mixing retry output with an earlier run.
    ``start_at`` supports an explicit resume into a fresh output directory.
    """

    root = Path(root).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise WorkflowError(f"refusing to reuse non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    manifest = RunManifest(
        run_id=run_id,
        experiment_family=family,
        status="RUNNING",
        command=[" ".join(map(str, command)) for command in commands.values()],
        configuration={"stages": list(commands), "start_at": start_at},
        provenance={"workspace": str(root), "runner": "experiments.execute", "python": sys.version},
        gates={"schema": None, "dataset": None, "offline": None, "runtime": None, "physical_task": None},
        outputs=[],
    )
    manifest_path = manifest.write(output / "run_manifest.json")
    stage_results: list[dict] = []
    active = start_at is None
    try:
        for name, command_iter in commands.items():
            if not active:
                active = name == start_at
                if not active:
                    continue
            command = [str(x) for x in command_iter]
            log_path = output / f"{name}.log"
            started = time.monotonic()
            with log_path.open("x") as log:
                log.write("$ " + " ".join(command) + "\n")
                log.flush()
                result = subprocess.run(
                    command,
                    cwd=str(Path(cwd).resolve()) if cwd else str(root),
                    env=_env(root, environment),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record = {
                "name": name,
                "command": command,
                "exit_code": result.returncode,
                "elapsed_seconds": time.monotonic() - started,
                "log": str(log_path),
            }
            stage_results.append(record)
            (output / "stages.json").write_text(json.dumps(stage_results, indent=2) + "\n")
            if result.returncode != 0:
                manifest.status = "FAILED"
                manifest.provenance["failure"] = record
                manifest.write(manifest_path)
                return result.returncode or 2
        manifest.status = "SUCCEEDED"
        manifest.outputs = [str(path) for path in sorted(output.iterdir())]
        manifest.provenance["stages"] = stage_results
        manifest.write(manifest_path)
        return 0
    except BaseException as exc:
        manifest.status = "FAILED"
        manifest.provenance["failure"] = {"type": type(exc).__name__, "error": str(exc)}
        manifest.write(manifest_path)
        raise
