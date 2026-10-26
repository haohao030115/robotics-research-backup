# Report Evidence Index

## Final checkpoint and runtime copies

| artifact | absolute path | SHA-256 / key evidence |
|---|---|---|
| full V4.1 checkpoint | `/home/group2/outputs/va_v4_1_kimodo_rule_20261009/checkpoint.pt` | `92de206b703d66cea50dfd8b92296778d2400d9969e3e964527a8bbd2328143f`; `v41=true`, epoch 50 |
| native rollout copy | `/home/group2/outputs/va_v4_1_kimodo_rule_20261009/rollout_checkpoint.pt` | `1beb45426b9b072873cc8e48f6ab2f2676cc355f916065482eb824660ed90e67`; V4-compatible extra-head-stripped copy |
| compatibility input | `/home/group2/outputs/va_v4_1_kimodo_rule_20261009/runtime_input/checkpoint_compat.pt` | `f78969b9bafe1c1c2e7da4b72616fc47e058acd71e5c44f29cce800a27326cc4` |
| training status | `/home/group2/outputs/va_v4_1_kimodo_rule_20261009/training_status.json` | finished=true; 50 epochs; 7,530/1,958 samples; 3,502,977 parameters |

The full checkpoint embeds the historical path `/workspace/group2/outputs/grasp_pipeline_kimodo_rule_combined_v4_20261009/dataset`; the rollout copy also embeds training-provenance hashes matching the retained `expert_datasets/train_dataset_v0` export. The report uses the retained export because it is present, byte-hashed and contains the exact 9,488-sample tensors.

## Dataset hashes and composition

| artifact | path | SHA-256 |
|---|---|---|
| manifest | `/home/group2/expert_datasets/train_dataset_v0/manifest.json` | `cc507f87571703906c3c6e8eb38a6e9dc8daee51109a64c014d51645786ffa2c` |
| samples | `/home/group2/expert_datasets/train_dataset_v0/samples.npz` | `b7ffb8437b913369d0202f3f024376182c764ebb0b445fb1e98e1db32a623fd9` |
| split | `/home/group2/expert_datasets/train_dataset_v0/split.json` | `263761da8e7afb0b13a0cf2bf1e1f2d9f2bce33d9cf0060d582cb15ae350f4a9` |
| index | `/home/group2/expert_datasets/train_dataset_v0/index.jsonl` | `3c8bad5e4bef8133112c062abaec0dd8f1894b127995ec6f3429e07702d9f13c` |
| completion record | `/home/group2/expert_datasets/train_dataset_v0/COMPLETED.json` | records sealed file hashes |

The manifest contains 63 episodes: 19 Kimodo and 44 rule-based; 50 train episodes/7,530 samples and 13 validation episodes/1,958 samples. The later 21-episode Unified Adapter V2 manifest is a subsequent production resource and is not the V4.1 training source.

## Model source

- [`va_research/models/v4/model.py`](../../va_research/models/v4/model.py): visual encoder, cross-attention, body/hand heads and base auxiliary heads.
- [`va_research/models/v4_1/model.py`](../../va_research/models/v4_1/model.py): readiness head and V4.1 forward extension.
- [`va_research/models/v4_1/losses.py`](../../va_research/models/v4_1/losses.py): readiness, velocity and reopening terms.
- [`va_research/models/v4_1/train_mixed.py`](../../va_research/models/v4_1/train_mixed.py): seed, optimizer, schedule, normalization, checkpoint writing.
- [`va_research/models/v4/runtime.py`](../../va_research/models/v4/runtime.py): absolute-tick planning and coverage checks.

## Offline evidence

- `outputs/va_v4_1_kimodo_rule_20261009/offline_metrics.json`: aggregate metrics, body/hand horizon arrays, confusion matrix, overlap flag and offline gate.
- `outputs/va_v4_1_kimodo_rule_20261009/history.json`: six recorded validation checkpoints.
- `outputs/va_v4_ablation/*/offline_metrics.json`: exploratory same-dataset ablations.

## Native learned-policy evidence

- `outputs/va_v4_1_kimodo_rule_20261009/closed_loop/report.json`: task criterion, 810/810 runtime, scheduler counters and physical failure.
- `outputs/va_v4_1_kimodo_rule_20261009/closed_loop/provenance.json`: checkpoint hash, native binary/controller hashes, MTT renderer provenance, camera, timing and `expert_future_injected=false`.
- `outputs/va_v4_1_kimodo_rule_20261009/closed_loop/inference_timing.json`: per-plan latency.
- `outputs/va_v4_1_kimodo_rule_20261009/closed_loop/control_timeline.json`: every absolute tick and the one fallback.
- `outputs/va_v4_1_kimodo_rule_20261009/closed_loop/physical_metrics.json` and `outputs/va_v4_1_kimodo_rule_20261009/closed_loop/contacts.json`: per-tick contact/lift/hold values.
- `outputs/va_v4_1_kimodo_rule_20261009/video/video_metadata.json`, `outputs/va_v4_1_kimodo_rule_20261009/video/failure_localization.json`, `outputs/va_v4_1_kimodo_rule_20261009/video/expert_vs_v4.mp4`: diagnostic comparison and divergence evidence.

## Canonical Data Factory and motion-generation evidence

- `data_factory/runtime/schema.py`, `pipeline.py`, `native_truth.py`, `native_forensic.py`, `production.py`, `compiler.py`.
- `data_factory/dataset/dataset.py`, `contracts.py`: accepted-episode compilation and tensor checks.
- `canonical_va/README.md`, `canonical_va/NATIVE_EXECUTION_AUDIT.md`: native identity, source policy and execution semantics.
- `outputs/canonical_production/production_summary.json`: physically successful deterministic Kimodo/ARDY expert examples.
- `outputs/unified_adapter_v2/joint_optimization_20261017/ARDY_KIMODO_COMPARISON_REPORT.md`: later adapter diversity and spatial results; not V4.1 training provenance.

## Evidence limitations

The aggregate offline evaluation is not an independent test because it records `validation_overlap=true`. The complete historical V4 prior path is not present as a standalone current output. Physical expert success does not imply learned success. Any claim that exceeds these evidence boundaries is intentionally excluded from the report.
