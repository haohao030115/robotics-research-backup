# Phase C — Dataset Audit + V4 Training Integration (2026-10-08)

Implemented `dataset audit` and explicit `train` commands without changing canonical, SONIC/DDS/MuJoCo, V4 model, loss, target, optimizer or normalization semantics.

## Added

- `python -m grasp_pipeline dataset audit --dataset DATASET [--output DIR]`: deterministic local `summary.json`, `report.html`, RGB gallery and `figures/`. It reports episode/split/producer/task-arm composition, cube XY when present, RGB/body/hand/token/hand-target statistics, phase/progress distributions and explicit unavailable fields. It does not score, filter, deduplicate or reject data.
- `python -m grasp_pipeline train --dataset DATASET --output RUN [--config CONFIG]`: explicit training boundary. Uses existing `V4Policy`, `objectives`, metric definitions, train-only mean/std clamp `.05`, seed, AdamW, gradient clip, LR schedule and offline gate/checkpoint criterion. It writes immutable run manifest, config, checkpoints, history, status and deterministic `report.html`. Existing `run` and `resume` remain preparation-only.
- `grasp_pipeline/reports.py` generates the report from saved training artifacts; no LLM or plotting dependency is required. Missing values are shown as unavailable.

## Verification

- Existing two-episode Phase B dataset was audited successfully: `outputs/grasp_pipeline_phase_b/dataset_audit/{summary.json,report.html,figures/}`.
- Phase B dataset and state tests remain passing; final Phase C suite has 21 tests passing (`outputs/grasp_pipeline_phase_b/tests_phase_c.log`).
- Historical full V4 artifacts were used for exact tensor/split checks. No new training, native replay, GPU training workload, offline evaluation, or closed-loop evaluation was launched.
- The explicit train command supports a dry run and has not been executed for a full training run in this phase, per scope. The training adapter is ready for a separately authorized run.

## Contracts and limits

The dataset manifest stores dataset checksum, split IDs, episode ranges and baseline tensor metadata. Training records seed, config, normalization rule, checkpoint paths and metric history. A training output directory is immutable and cannot be overwritten. Phase C does not add left-hand label semantics; existing right-hand phase/progress definitions are preserved. Audit statistics mark missing wrist/cube fields unavailable rather than reconstructing them.
