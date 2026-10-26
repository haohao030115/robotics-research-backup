# Experiment orchestration

Research Experiments & Orchestration is a cross-cutting management layer. It schedules and records Kimodo, ARDY, canonical, dataset, training, offline evaluation and closed-loop runs; it does not implement their algorithms.

## Run record

```text
run_id, experiment_family, variant, status
source_repositories/commits_or_hashes
checkpoint_hashes, reference_id, dataset_id, model_id
configuration + configuration_hash
execution_command + container/runtime manifest
renderer/DDS mode and hardware identity
inputs/outputs and artifact hashes
gates: schema, reference, dataset, offline, runtime, physical_task
logs, report paths, failure class, parent run IDs
```

Run IDs are immutable. A retry gets a new attempt under the same experiment declaration; outputs are never overwritten. Historical snapshots remain addressable even when Git identity is unknown.

## Configuration ownership

Experiment configs select producer, case, task arm, seed, runtime profile, dataset split, model family, checkpoint, evaluation mode and resource allocation. They do not duplicate joint order, camera, timing, hand gains, FSQ or grasp thresholds; those come from versioned contracts/runtime manifests.

## Commands and gates

The orchestration CLI should expose wrappers for:

* `kimodo generate|export|adapt|qualify`
* `ardy generate|convert|adapt|qualify`
* `canonical run|audit|compile`
* `dataset build|audit`
* `va train|evaluate|rollout`
* `run status|report|reproduce`

Initially these delegate to existing commands (`kimodo/scripts/generate.py`, ARDY `run_pipeline.py`, `canonical_va.batch`, `grasp_pipeline.cli`, V4 trainers/rollouts). Dry-run output must show the exact delegated command and manifest without starting native work.

## Reporting principles

Kimodo/ARDY reports may include motion quality, IK and producer metrics. Canonical reports include trace/forensic validity. VA reports include offline metrics and checkpoint provenance. Runtime reports include scheduler, renderer, DDS and completion evidence. Physical reports include contacts, lift, hold and fall criteria. A runtime-successful but task-unsuccessful run is recorded as such, never promoted as a successful grasp.

## Reproduction

`reproduce(run_id)` resolves the stored source/checkpoint/runtime manifests, verifies hashes, selects the validated container and creates a fresh output directory. It refuses to run on mismatched contracts or stale critical assets. Reproduction of historical runs can use recorded traces for audit, but production qualification requires the policy’s declared native execution mode.
