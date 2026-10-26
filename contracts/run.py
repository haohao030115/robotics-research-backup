"""Immutable run provenance helpers for experiment orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class RunManifest:
    run_id: str
    experiment_family: str
    status: str = "PLANNED"
    command: list[str] = field(default_factory=list)
    configuration: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, bool | None] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)

    def validate(self) -> "RunManifest":
        if not self.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in self.run_id):
            raise ValueError("run_id must be a filesystem-safe identifier")
        if not self.experiment_family:
            raise ValueError("experiment_family is required")
        if self.status not in {"PLANNED", "RUNNING", "SUCCEEDED", "FAILED", "BLOCKED"}:
            raise ValueError(f"unsupported run status: {self.status}")
        return self

    def write(self, path: str | Path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n")
        return path

    @classmethod
    def read(cls, path: str | Path) -> "RunManifest":
        return cls(**json.loads(Path(path).read_text())).validate()
