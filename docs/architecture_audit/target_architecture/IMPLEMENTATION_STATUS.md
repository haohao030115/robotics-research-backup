# Implementation status

The reorganized layout now contains executable local implementations for the seven target areas. Historical repositories and generated outputs remain intact as compatibility and provenance sources; protected SONIC/model internals were not rewritten.

## Source ownership

- `research/kimodo/`: generation, export, adaptation support, qualification, and runtime support scripts copied from the locally developed Kimodo integration. `experiments run --family kimodo` executes the staged workflow and records logs/provenance.
- `research/ardy/`: generation, segmented overlap, adaptation, qualification, online/replay support, and environment-resolved pipeline. The former Qwen/ARDY source under `workspace/ljy/qwen_ardy_closed_loop` is now physically under `research/ardy/online/qwen`; generated runs remain in the workspace archive. `experiments run --family ardy` executes one authoritative pipeline stage whose internal records contain prepare/generate/adapt/replay/verify results.
- `contracts/`: canonical reference, run-manifest, and policy contracts used by producers and consumers.
- `execution/sonic_mujoco/`: local runtime, native scene, rendering, trace, and build integration. `execution/native.py` now targets the copied canonical runtime and remains native-only; protected SONIC/DDS code stays in `GR00T-WholeBodyControl`.
- `data_factory/runtime/`: canonical trace, compiler, production, batch, schema, forensic, and versioning implementation.
- `data_factory/dataset/`: V4 split, provenance, immutable artifact, and tensor serialization implementation migrated from `grasp_pipeline`; the old package remains a compatibility entry point.
- `va_research/models/v2`, `v3`, `v4`, `v4_1`, `v4_0_bcz_hand`, `v4_oracle`: model, rollout, training, and evaluation implementations organized by version. Shared datasets, losses, SONIC release/runtime, configs, trainers, and utilities live under `va_research/`.
- `experiments/`: executable orchestration, structured manifests, stage logs, failure capture, and configuration resolution. Historical Kunqi batch source is catalogued under `experiments/archive/kimodo_batches`.
- `workspace/`: now has an archive index; Qwen/ARDY source was physically moved to `research/ardy/online/qwen`, while checkpoints, external checkouts, SDKs, videos, logs, and generated runs remain in place as immutable provenance artifacts.

## Working entry points

```bash
python3 -m experiments run --family kimodo --run-id ID --output OUTPUT --case center
python3 -m experiments run --family ardy --run-id ID --output OUTPUT --config ARDY_CONFIG --gpu 0
python3 -m experiments dataset --run-id ID --canonical-root CANONICAL --output OUTPUT --all-train
python3 -m experiments va offline --family v4 --run-id ID --checkpoint CHECKPOINT --output OUTPUT
python3 -m experiments va train --family v4 --run-id ID --manifest MANIFEST --output OUTPUT --device musa:0
python3 -m experiments va closed --family v4 --run-id ID --checkpoint CHECKPOINT --output OUTPUT
```

Training, native replay, rendering, and closed-loop commands remain explicit; ordinary command planning and import tests do not start expensive workloads.

## Evidence

- `bash scripts/check_environment.sh`: PASS; host warnings correctly direct MUSA/native work to `va-train`.
- `va-train` reorganized import smoke: PASS.
- `va-train` reorganization tests: **11 passed** with the standard-library runner.
- Focused V4 compatibility suite: **5 passed**.
- Reorganized canonical dataset production using an existing accepted episode: **SUCCEEDED**, producing samples, index, manifest, split, and completion marker.
- Reorganized V4 offline evaluation using the existing checkpoint: **SUCCEEDED**; metrics artifact and run manifest were written.
- Reorganized native canonical SONIC/MuJoCo workflow: **SUCCEEDED** in an isolated DDS namespace in 75.1 seconds; the original native binary, MTT S4000 renderer, MuJoCo scene, independent hand PD, trace joins and forensic audit all completed with `READY_FOR_UNIFIED_DATASET_PRODUCTION`.
- Reorganized rendering smoke: **SUCCEEDED** with `GL_VENDOR=Moore Threads`, `GL_RENDERER=MTT S4000`, 20 frames, and software rendering rejected.
- Host bytecode compilation for all reorganized packages: PASS.
- Qwen/ARDY online source relocation: syntax and functional smoke passed; source scripts now resolve the new root through `RESEARCH_ROOT`, `SONIC_ROOT`, `ARDY_REPO`, `QWEN_MODEL`, and `RUNS_ROOT`.
- Workspace archive inventory: `workspace/README.md` and `workspace/ARCHIVE_INDEX.json` added; dated ARDY snapshots are explicitly marked historical.

## Remaining verification limits

Physical Kimodo/ARDY qualification, VA training, and closed-loop rollout were not started in this pass because they are timing-sensitive workloads. Native SONIC/MuJoCo replay and canonical forensic execution are now verified through the reorganized entry point. Their entry points and environment contracts are resolved and import-tested, but runtime equivalence remains unverified until those workloads are deliberately scheduled in the isolated `va-train` runtime.

Legacy `canonical_va`, `grasp_pipeline`, `va_training`, Kimodo task directories, ARDY checkout, and protected SONIC source remain as compatibility/historical trees. Their active callers have been redirected where validated; they are not deleted because they contain uncommitted research, external dependencies, checkpoints, or runtime-critical assets.
