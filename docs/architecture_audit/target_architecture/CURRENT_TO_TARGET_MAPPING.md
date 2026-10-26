# Current-to-target mapping

Migration operations are additive wrappers or extraction only. No operation below authorizes moving frozen upstream code or changing algorithms.

| Current path / entry point | Target module | Proposed path | Operation | Dependencies | Risk | Validation |
|---|---|---|---|---|---|---|
| `kimodo-t6/kimodo-main/kimodo/**`, `kimodo/scripts/generate.py` | Kimodo / Generation | `research/kimodo/generation/` wrapper | expose command and provenance adapter; retain model checkout | Kimodo checkpoint/text encoder | Medium | generated motion metadata and source hashes |
| `kimodo_to_t7_reference.py`, `kimodo/exports/mujoco.py` | Kimodo / Motion Export | `research/kimodo/export/` | wrapper around existing converter | Kimodo, G1 XML | High | 50 Hz arrays, joint permutation, wxyz and qpos equality |
| `dl-group2/scripts/adapt_model_reference.py` | Kimodo / Reference Adaptation | `research/kimodo/adaptation/` | move only after wrapper parity; initially delegate | MuJoCo, SONIC model | High | per-frame IK error and candidate hash |
| Kimodo qualification snapshots and `qualification_once.py` | Kimodo / Qualification | `research/kimodo/qualification/` | report adapter, no replay rewrite | native runtime | High | existing qualification reports and physical gates |
| `workspace/fuyuhan/ardy/**` | ARDY / Segmented Generation | `research/ardy/generation/` wrapper | retain external model and checkpoint | ARDY model/text encoder | Medium | native frame count, seed and segment log |
| `generate_segmented.py` | ARDY / History / Overlap | `research/ardy/overlap/` | wrapper/documentation first | ARDY motion representation | High | overlap frames, native history fields, qpos equality |
| `adapt_ardy.py`, `planned_grasp_reference.py` | ARDY / IK / Calibration | `research/ardy/adaptation/` | delegate wrapper; keep calibration evidence | MuJoCo, ARDY, calibration | High | wrist position/orientation and source hash checks |
| `replay_ardy_isolated.py`, `validate_grasp_run.py` | ARDY / Qualification | `research/ardy/qualification/` | report adapter | isolated DDS/native SONIC | High | replay timeline and grasp metrics |
| `qwen_ardy_closed_loop/qwen_ardy/**` | ARDY / Online/staged variants | `research/ardy/experiments/qwen/` | classify as experimental wrappers | Qwen, ARDY | Medium | reproduce existing staged reports |
| `canonical_va/schema.py`, `runtime_contract.json`, `source_policy.json` | Canonical Reference Interface | `contracts/reference/` | extract read-only contract after parity | none at runtime | Medium | fixture acceptance/rejection and byte-stable metadata |
| `canonical_va/native_replay.py`, `native_truth.py`, `native_build/` | SONIC + MuJoCo Execution | `execution/canonical_runtime/` | package boundary around existing implementation | SONIC, DDS, MUSA, MuJoCo | Very high | native replay, hash policy, exact initial state |
| `dl-group2/scripts/run_sonic_grasp.py`, `hand_trajectory_controller.py` | SONIC + MuJoCo / bridge and hand PD | `execution/canonical_runtime/bridge/` | delegate wrapper; no controller rewrite | SONIC, MuJoCo | Very high | 200/50/500 Hz and Kp/Kd parity |
| `canonical_va/compiler.py`, `pipeline.py`, `forensic.py` | Canonical Data Factory | `data_factory/{compiler,audit}/` | extract facade, retain old entry points | canonical trace | High | `validate_export`, ID joins and forensic flags |
| `grasp_pipeline/dataset.py`, `training.py`, `evaluation.py` | Research Experiments / orchestration facade | `experiments/cli/` | keep as compatibility CLI, delegate gradually | canonical/data/VA packages | Medium | dry runs, immutable output behavior |
| `va_training/v2,v3,v4,v4_1,v4_0_bcz_hand,v4_oracle` | VA Research / Models | `va_research/models/` | registry and import wrappers; no model edits initially | PyTorch/MUSA | Medium | checkpoint load and output shape parity |
| `va_training/*training.py`, `train_*.py` | VA Research / Training | `va_research/training/` | entry-point wrappers first | datasets, MUSA | Medium | training provenance and offline metrics |
| `va_training/*rollout*.py`, `sonic/` | VA Research / Closed loop | `va_research/rollout/` | backend interface around current native runners | native runtime, EGL, DDS | Very high | scheduler and token-consumption traces |
| `outputs/`, `expert_datasets/`, run snapshots | Artifacts/manifests | `artifacts/` and `manifests/` | registry references only; do not move initially | filesystem/object store | Low | hash and report re-openability |
| `GR00T-WholeBodyControl/**` | External upstream/frozen SONIC | remains in place | dependency manifest and patch overlay only | MUSA/C++/DDS | Very high | release hashes and native smoke test |

RDK-related components were not verified as a distinct active source boundary in this workspace. They remain **UNKNOWN** and must be added to the mapping only after launch/process evidence identifies their entry points.
