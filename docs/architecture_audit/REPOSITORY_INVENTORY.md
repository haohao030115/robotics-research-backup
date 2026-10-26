# Repository inventory

Audit date: 2026-10-14. **VERIFIED** means confirmed from source, Git metadata, or an existing report; **INFERRED** means supported by naming and call sites; **UNKNOWN** means not safe to decide from this workspace.

## Boundaries

| Tree | Actual role | Type/status | Evidence |
|---|---|---|---|
| `GR00T-WholeBodyControl/` | Upstream SONIC/GR00T code, Python training/support packages, C++ native G1 deployment, ONNX release bundle, robot model and SDKs | External upstream checkout with local `musa-sonic` branch; **VERIFIED active/frozen runtime dependency** | `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:176-188,2195-2200,2627-2635`; `canonical_va/common.py:8-14` |
| `dl-group2/` | Local tabletop MuJoCo scene, expert/reference producers, adaptation, hand PD, native SONIC bridge scripts and dataset utilities | Local Git repo; **VERIFIED active**; contains a Git submodule at `third_party/GR00T-WholeBodyControl` | `scripts/run_sonic_grasp.py:156-260`; `scripts/expert_trajectory.py:148-215`; `scripts/project_paths.py` |
| `canonical_va/` | Canonical reference schema, native replay/compiler, provenance/hash policy, renderer bootstrap, forensic validation and batch orchestration | Local application package; **VERIFIED active and protected** | `pipeline.py:40-145`; `native_replay.py:49-115`; `schema.py:19-251`; `source_policy.json` |
| `va_training/` | Successive VA implementations (baseline/flow, V2, V3, V4, V4.1, oracle, BCZ hand), loaders, losses, trainers, offline evaluation and native closed-loop rollouts | Separate local Git repo; **VERIFIED active but heavily versioned/untracked** | `v4/model.py:24-58`; `v4/dataset.py:43-125`; `v4/rollout.py:57-217`; Git status shows V4 and diagnostics as untracked |
| `grasp_pipeline/` | Artifact-contract wrapper around canonical compilation, dataset serialization/audit, training and evaluation | Local orchestration package; **VERIFIED active** | `cli.py:150-180`; `dataset.py:13-90`; `training.py:24-42` |
| `kimodo-t6/kimodo-main/` and integration snapshots | Kimodo model/library, motion conversion, G1 export and qualification snapshots | External/local experiment repositories and immutable run snapshots; **VERIFIED active as expert source, exact deployed checkout varies by run** | `kimodo/scripts/generate.py:286-403`; `g1-integration/task8_qualification_v2/code/generate_grasp_motion.py:56-96`; `kimodo_to_t7_reference.py:163-244` |
| `workspace/fuyuhan/ardy/` and `ardy_tabletop_segmented_20261008/` | ARDY model/checkpoints plus segmented generation, adaptation, wrist IK, isolated SONIC replay and validation | External/local experiment checkout and run snapshot; **VERIFIED active for ARDY path** | `code/run_pipeline.py:58-74`; `code/generate_segmented.py:95-150`; `code/adapt_ardy.py:40-113` |
| `research/ardy/online/qwen/` | Qwen/ARDY staged and online method-B experiments | Experiment repository; **VERIFIED separate/experimental** | `qwen_ardy/closed_loop.py`, `method_b_staged.py`, `rolling_planner.py` |
| `outputs/`, `expert_datasets/`, `handoffs/`, `kimodo-t6/*/outputs`, `workspace/*/runs` | Checkpoints, canonical episodes, videos, logs, manifests, audits and handoff snapshots | Generated artifacts; **VERIFIED not reusable source** | Directory contents and manifests; `outputs/v4_0_bcz_hand_rule100_full_20261009/closed_loop/report.json` |

## Build and runtime dependencies

* Python training/evaluation uses the existing `va-train` container and MUSA stack; host inspection reports `torch/MUSA import unavailable` (**VERIFIED**, `scripts/check_environment.sh`).
* Native SONIC uses CMake/C++20, Unitree SDK, ZMQ, YAML, Eigen, Python embedding and MUSA (`canonical_va/native_build/cmake/MusaController.cmake:1-21`).
* MuJoCo is loaded by `dl-group2` and `canonical_va`; the frozen contract requires MuJoCo 3.14.0, model dimensions `(nq,nv,nu,neq)=(57,55,43,0)` and 200/50 Hz (`canonical_va/runtime.py:44-80`).
* Native runs need an isolated network namespace and loopback; batch orchestration invokes `unshare --net` (`canonical_va/batch.py:12-38`).
* Release ONNX contracts are encoder `[1,1762] -> [1,64]` and decoder `[1,994] -> [1,29]`; MUSA graph execution is explicit and CPU fallback is rejected (`GR00T-WholeBodyControl/gear_sonic_deploy/scripts/musa_onnx.py:71-117`).

## Artifact inventory

The checked release bundle contains `model_encoder.onnx` (~50 MB), `model_decoder.onnx` (~41 MB), and `observation_config.yaml`. Kimodo and ARDY checkpoints are safetensors (~1.13 GB and ~0.77 GB respectively). Native and VA checkpoints, ONNX files, videos, logs and experiment folders are generated or external inputs and must remain outside reusable library packages.

## Unknowns

Several workspace-owned repositories cannot be inspected through Git because their directories have different ownership and Git refuses them as dubious. Their source files are readable, but commit identity and clean status are **UNKNOWN**. Permission-denied episode directories also prevent a complete artifact-level inventory. No claim of dead code is made from absence of imports.
