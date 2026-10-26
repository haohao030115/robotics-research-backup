"""Build a small, read-only index of reproducibility artifacts.

The catalog deliberately records only metadata from JSON manifests and paths;
it never imports experiment code or rewrites generated outputs.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable


_FIELDS = (
    "run_id", "experiment_family", "status", "experiment_id", "milestone",
    "schema_version", "artifact_type", "dataset_id", "model", "sample_count",
    "artifact_id", "checkpoint", "gate",
)

_REPRESENTATIVES = {
    "kimodo_canonical_episode": [
        "research/kimodo/workflow.py",
        "outputs/canonical_va/kimodo_batch_20261008_v4/kimodo_episode_0004/training_manifest.json",
        "outputs/canonical_va/kimodo_batch_20261008_v4/kimodo_episode_0004/va_dataset/samples.npz",
        "outputs/canonical_va/kimodo_batch_20261008_v4/kimodo_episode_0004/validation/dataset_validation.json",
    ],
    "ardy_canonical_episode": [
        "research/ardy/workflow.py",
        "outputs/canonical_va/diverse_block_xy_20/attempts/ardy_near_seed2/unified_native_episode/training_manifest.json",
        "outputs/canonical_va/diverse_block_xy_20/attempts/ardy_near_seed2/unified_native_episode/va_dataset/samples.npz",
        "outputs/canonical_va/diverse_block_xy_20/attempts/ardy_near_seed2/unified_native_episode/canonical_replay/metadata.json",
    ],
    "rule_based_production": [
        "dl-group2/scripts/expert_trajectory.py",
        "expert_datasets/rule_reference_right100_v3/manifest.json",
        "outputs/rule_reference_right100_v3_remaining_0002/selection_manifest.json",
    ],
    "canonical_data_factory": [
        "data_factory/runtime/pipeline.py",
        "outputs/canonical_va/kimodo_19_merged/training_manifest.json",
        "outputs/canonical_va/kimodo_19_merged/va_dataset/samples.npz",
    ],
    "va_training": [
        "va_research/models/v4/training.py",
        "outputs/grasp_pipeline_kimodo_rule_combined_v4_20261009/training_run/manifest.json",
        "outputs/grasp_pipeline_kimodo_rule_combined_v4_20261009/training_run/config.json",
        "outputs/grasp_pipeline_kimodo_rule_combined_v4_20261009/training_run/checkpoint.pt",
        "outputs/grasp_pipeline_kimodo_rule_combined_v4_20261009/training_run/training_status.json",
    ],
    "va_offline_evaluation": [
        "va_research/models/v4/evaluation.py",
        "outputs/grasp_pipeline_kimodo_rule_combined_v4_20261009/training_run/offline_metrics.json",
    ],
    "va_closed_loop": [
        "va_research/models/v4/rollout.py",
        "outputs/grasp_pipeline_kimodo_rule_combined_v4_20261009/closed_eval_attempt_0003/provenance.json",
        "outputs/grasp_pipeline_kimodo_rule_combined_v4_20261009/closed_eval_attempt_0003/report.json",
    ],
}


def _resolve_recorded(value: str | Path, root: Path) -> Path:
    workspace = root.parent if root.name in {"outputs", "expert_datasets"} else root
    path = Path(value)
    text = str(path)
    if text.startswith("/workspace/group2/"):
        return workspace / text.removeprefix("/workspace/group2/")
    if path.is_absolute():
        return path
    return workspace / path if str(path).startswith(("outputs/", "expert_datasets/")) else root / path


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _category(path: Path, data: dict[str, Any]) -> str:
    if path.name == "training_manifest.json" and "episodes" in data:
        return "canonical_episode"
    if data.get("artifact_type") in {"v4_dataset", "dataset"}:
        return "compiled_dataset"
    if data.get("sources") and data.get("output"):
        return "compiled_dataset"
    if data.get("artifact_type") == "training" or path.name.endswith("checkpoint_manifest.json"):
        return "training_checkpoint"
    if path.name == "manifest.json" and "episodes" in data:
        return "expert_reference"
    if path.name == "experiment_manifest.json":
        return "intermediate_experiment"
    if path.name == "run_manifest.json":
        return "workflow_run"
    return "manifest"


def _evidence_category(path: Path) -> str | None:
    if path.name in {"checkpoint.pt", "latest.pt"}:
        return "training_checkpoint"
    if path.name in {"offline_metrics.json", "metrics.json", "report.json"}:
        return "evaluation_report"
    if path.name in {"validation_summary.json", "dataset_validation.json"}:
        return "validation_report"
    if path.name == "pipeline.json":
        return "intermediate_experiment"
    return None


def _record(path: Path, root: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"path": str(path.relative_to(root)), "kind": path.name, "readable": False}
    row = {key: data[key] for key in _FIELDS if key in data}
    row["category"] = _category(path, data)
    if isinstance(data.get("episodes"), list):
        row["episode_count"] = len(data["episodes"])
        row["episode_ids"] = [item.get("episode_id") for item in data["episodes"] if isinstance(item, dict) and item.get("episode_id")]
    row.update({"path": str(path.relative_to(root)), "kind": path.name, "readable": True})
    references: list[str] = []
    if isinstance(data.get("sources"), list):
        references.extend(str(item) for item in data["sources"])
    if isinstance(data.get("output"), str):
        references.append(data["output"])
    missing = [str(_resolve_recorded(item, root)) for item in references if not _resolve_recorded(item, root).exists()]
    row["missing_references"] = missing
    if row["category"] == "canonical_episode":
        rows = data.get("episodes", [])
        row["verified_episode_count"] = sum(bool(item.get("expert_valid")) for item in rows if isinstance(item, dict))
        row["status"] = "verified" if rows and not missing and row["verified_episode_count"] == len(rows) else "incomplete"
    elif row["category"] == "compiled_dataset":
        row["status"] = "verified" if not missing else "incomplete"
    elif row["category"] == "expert_reference":
        row["status"] = "verified" if data.get("episodes") and not missing else "incomplete"
    elif row["category"] == "training_checkpoint":
        checkpoint = path.parent / "checkpoint.pt"
        row["status"] = "verified" if checkpoint.is_file() and not missing else "incomplete"
    elif "status" not in row:
        row["status"] = "incomplete"
    return row


def build_catalog(roots: Iterable[str | Path]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    root_rows: list[dict[str, Any]] = []
    for raw in roots:
        root = Path(raw).expanduser().resolve()
        root_rows.append({"path": str(root), "exists": root.is_dir()})
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            if path.is_file() and path.name in {"run_manifest.json", "manifest.json", "training_manifest.json", "experiment_manifest.json"}:
                entries.append(_record(path, root))
    evidence: list[dict[str, Any]] = []
    for raw in roots:
        root = Path(raw).expanduser().resolve()
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            category = _evidence_category(path)
            if category and path.is_file():
                evidence.append({"category": category, "path": str(path.relative_to(root)), "root": str(root), "exists": True, "sha256": _sha256(path)})
    entries.sort(key=lambda row: (row.get("category", ""), row.get("status", ""), row["path"]))
    identities: dict[str, list[str]] = {}
    for row in entries:
        for episode_id in row.get("episode_ids", []):
            identities.setdefault(episode_id, []).append(row["path"])
    duplicate_groups = {key: sorted(paths) for key, paths in identities.items() if len(paths) > 1}
    root = Path.cwd().resolve()
    representatives = {}
    for name, paths in _REPRESENTATIVES.items():
        files = []
        for raw in paths:
            path = root / raw
            files.append({"path": raw, "exists": path.is_file(), "sha256": _sha256(path)})
        representatives[name] = {"verified": all(item["exists"] for item in files), "files": files}
    categories: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for row in entries:
        categories[row.get("category", "manifest")] = categories.get(row.get("category", "manifest"), 0) + 1
        statuses[row.get("status", "unclassified")] = statuses.get(row.get("status", "unclassified"), 0) + 1
    return {
        "schema_version": 2,
        "roots": root_rows,
        "artifact_count": len(entries),
        "evidence_count": len(evidence),
        "category_counts": categories,
        "status_counts": statuses,
        "duplicate_episode_groups": duplicate_groups,
        "artifacts": entries,
        "evidence": evidence,
        "representatives": representatives,
    }


def write_catalog(roots: Iterable[str | Path], output: str | Path) -> Path:
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(build_catalog(roots), indent=2, sort_keys=True) + "\n")
    return destination
