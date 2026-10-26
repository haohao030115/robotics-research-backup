# Reproducibility and delivery index

This index points to representative artifacts that are present in the
workspace. The machine-readable catalog at [artifact_catalog.json](artifact_catalog.json)
records SHA-256 hashes, category, status, missing references, and duplicate
episode identities. Re-run `python3 -m experiments catalog` after adding a
result.

## Representative chains

| Area | Source/config evidence | Execution and validation | Downstream artifact | Current status |
|---|---|---|---|---|
| Kimodo | `research/kimodo/workflow.py` | `outputs/canonical_va/kimodo_batch_20261008_v4/kimodo_episode_0004/validation/dataset_validation.json` | `.../training_manifest.json`, `.../va_dataset/samples.npz` | Canonical episode verified; fresh Kimodo export remains blocked by its existing IK gate |
| ARDY | `research/ardy/workflow.py` | `outputs/canonical_va/diverse_block_xy_20/attempts/ardy_near_seed2/unified_native_episode/canonical_replay/metadata.json` | `.../training_manifest.json`, `.../va_dataset/samples.npz` | Runtime/canonical artifact verified; physical grasp result remains separate |
| Rule-based production | `dl-group2/scripts/expert_trajectory.py` | `outputs/rule_reference_right100_v3_remaining_0002/selection_manifest.json` | `expert_datasets/rule_reference_right100_v3/manifest.json` | Reference catalog present; source repository retains uncommitted research changes |
| Canonical Data Factory | `data_factory/runtime/pipeline.py` | `outputs/canonical_va/kimodo_19_merged/training_manifest.json` and its source list | `outputs/canonical_va/kimodo_19_merged/va_dataset/samples.npz` | 19 episodes and 3,059 samples; one episode explicitly excluded |
| VA training | `va_research/models/v4/training.py`, recorded training config | `outputs/grasp_pipeline_kimodo_rule_combined_v4_20261009/training_run/training_status.json` | `.../checkpoint.pt` | Finished with `train_gate: true` |
| VA offline evaluation | `va_research/models/v4/evaluation.py` | `.../training_run/offline_metrics.json` | Metrics are tied to the checkpoint manifest and dataset hash | Offline metrics present and checkpoint provenance is recorded |
| VA closed loop | `va_research/models/v4/rollout.py` | `.../closed_eval_attempt_0003/provenance.json` and `report.json` | 810/810 recorded ticks, zero deadline or uncovered ticks | Runtime integrity passed; physical grasp was false |

The `...` entries above are shorthand for the full paths recorded in
`artifact_catalog.json`; they are not additional path aliases.

## Interpretation rules

- A canonical episode is counted once by its episode identity. Repeated
  manifests from retries, snapshots, or merged views are listed under
  `duplicate_episode_groups`.
- `verified` means the files and manifest fields needed for that artifact were
  present at the audit. It does not turn a runtime pass into physical task
  success.
- A successful training or offline metric file does not certify a closed-loop
  task. The closed-loop report must be read for scheduler integrity and the
  physical grasp fields.
- Paths embedded by the original container runs are normalized from
  `/workspace/group2` to this workspace when checking file existence. The
  original recorded path and its hash remain in the manifest.

## Reproduction checks

Run from the workspace root after `bash scripts/check_environment.sh`:

```bash
python3 -m experiments run --family kimodo --run-id CHECK_K --output /tmp/check_kimodo --dry-run
python3 -m experiments run --family ardy --run-id CHECK_A --output /tmp/check_ardy --dry-run
python3 -m experiments canonical --run-id CHECK_C --reference REFERENCE --output /tmp/check_canonical --dry-run
python3 -m experiments dataset --run-id CHECK_D --canonical-root CANONICAL --output /tmp/check_dataset --all-train --dry-run
python3 -m experiments va offline --family v4 --run-id CHECK_O --checkpoint CHECKPOINT --output /tmp/check_offline --dry-run
python3 -m experiments va closed --family v4 --run-id CHECK_L --checkpoint CHECKPOINT --output /tmp/check_closed --dry-run
```

The `va-train` container is required for MUSA training, native replay, GPU
rendering, and closed-loop execution. The final verification audit records
the exact successful and blocked runs in
[FINAL_VERIFICATION_20261015.md](architecture_audit/FINAL_VERIFICATION_20261015.md).
