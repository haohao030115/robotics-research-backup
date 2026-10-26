"""Executable commands for all seven logical workflow areas."""

from __future__ import annotations

from pathlib import Path
import sys

from data_factory.workflow import dataset_command
from execution.native import canonical_command
from research.ardy.workflow import build_commands as ardy_commands
from research.kimodo.workflow import build_commands as kimodo_commands


def expert_commands(family: str, root: Path, output: Path, **options) -> dict[str, list[str]]:
    if family == "kimodo":
        return kimodo_commands(
            project_root=root,
            prompt=options.get("prompt", "grasp"),
            output=output,
            seed=options.get("seed", 0),
            case=options.get("case", "center"),
            adapter=options.get("adapter", "legacy"),
            hand_profile=options.get("hand_profile", "default"),
        )
    if family == "ardy":
        built = ardy_commands(
            project_root=root,
            config=options["config"],
            output=output / "artifact",
            seed=options.get("seed", 0),
            case=options.get("case", "center"),
            gpu=options.get("gpu", 0),
            adapter=options.get("adapter", "legacy"),
            hand_profile=options.get("hand_profile", "ardy"),
        )
        return {"pipeline": built["pipeline"]}
    raise ValueError(f"unsupported expert family: {family}")


def canonical_commands(reference: Path, output: Path, *, gpu: int, mode: str) -> dict[str, list[str]]:
    return {"canonical": canonical_command(reference, output / "artifact", gpu=gpu, mode=mode)}


def dataset_commands(canonical_root: Path, output: Path, *, val_count: int | None, seed: int | None, all_train: bool) -> dict[str, list[str]]:
    return {"dataset": dataset_command(canonical_root, output, val_count=val_count, seed=seed, all_train=all_train)}


def va_commands(
    family: str,
    operation: str,
    *,
    checkpoint: Path | None = None,
    manifest: Path | None = None,
    dataset: Path | None = None,
    output: Path,
    config: Path | None = None,
    device: str = "musa:0",
) -> dict[str, list[str]]:
    if operation == "train":
        if family != "v4" or manifest is None:
            raise ValueError("executable training currently supports v4 with --manifest")
        # The data-factory adapter owns dataset serialization and provenance;
        # it invokes the authoritative va_research V4 trainer after validating
        # the complete dataset directory. This prevents training from
        # silently consuming a raw canonical manifest.
        command = [sys.executable, "-B", "-m", "data_factory.dataset.cli", "train",
                   "--dataset", str(Path(manifest).resolve().parent),
                   "--output", str(output / "artifact")]
        if config:
            command += ["--config", str(config)]
        return {"train": command}
    if operation == "offline":
        if checkpoint is None:
            raise ValueError("--checkpoint is required")
        if family != "v4":
            raise ValueError("executable offline evaluation currently supports v4")
        return {"offline": [sys.executable, "-B", "-m", "va_research.models.v4.evaluation", "--checkpoint", str(checkpoint), "--output", str(output / "artifact")]}
    if operation == "closed":
        if checkpoint is None:
            raise ValueError("--checkpoint is required")
        module = "va_research.models.v4.one_shot_rollout" if family == "bcz_hand" else "va_research.models.v4.rollout"
        return {"closed_loop": [sys.executable, "-B", "-m", module, "--checkpoint", str(checkpoint), "--output", str(output / "artifact")]}
    raise ValueError(f"unsupported VA operation: {operation}")
