# Authoritative implementation and caller audit

Audit date: 2026-10-15. The complete metadata inventory is
`FULL_FILE_INVENTORY_20261015.jsonl.gz`; this document records executable
ownership rather than treating every historical file as a competing source.

| Functionality | Authoritative maintained source | Active callers | Legacy/external status |
|---|---|---|---|
| Kimodo generation, export, adaptation, qualification | `research/kimodo/` | `research.kimodo.workflow`, `experiments run --family kimodo` | `kimodo-t6/` is retained external source/checkpoint/provenance; old scripts are archived or compatibility inputs |
| ARDY segmented generation, overlap, adaptation, qualification | `research/ardy/` | `research.ardy.workflow`, `experiments run --family ardy` | `workspace/fuyuhan/ardy` remains model/checkpoint dependency; Qwen source is isolated under `research/ardy/online/qwen` |
| Canonical contracts and dataset production | `contracts/`, `data_factory/` | `experiments dataset`, `data_factory.dataset.cli`, `data_factory.runtime.*` | `canonical_va/` and `grasp_pipeline/` are compatibility/reproduction trees; compiler subprocesses now resolve through `data_factory.runtime` |
| SONIC/MuJoCo/native execution | `execution/sonic_mujoco/` and `data_factory/runtime/` | `execution.native`, canonical/data factory commands, VA rollouts | `GR00T-WholeBodyControl/` is protected upstream; `dl-group2/` supplies frozen scene/controller dependencies |
| VA V4/BCZ model, training, offline evaluation, rollout | `va_research/models/v4/`, `va_research/models/v4_0_bcz_hand/`, `va_research/models/v4_1/` | `experiments va`, `data_factory.dataset.training/evaluation` | `va_training/` remains historical compatibility/checkpoint reproduction; V4 active callers no longer import its modules |
| VA V2/V3 historical families | `va_research/models/v2/`, `va_research/models/v3/` for migrated model/data code | registry inspection only; no current unified workflow dispatches them | Their coupled legacy scripts remain in `va_training/scripts/` until a dependency-complete migration; this is explicitly non-authoritative compatibility code |
| Online Qwen/ARDY experiments | `research/ardy/online/qwen/` | explicit online scripts only | Separate experimental workflow; never selected by default ARDY qualification |
| Experiment provenance and execution | `experiments/` | all new workflow commands | old handoffs and one-off runners are under `archive/` with compatibility links where needed |

The active path audit found no current unified command that dispatches the
legacy `canonical_va.batch`, `grasp_pipeline.training`, or `va_training/v4`
implementations. Compatibility imports remain only where a historical family
or protected runtime contract explicitly requires them.
