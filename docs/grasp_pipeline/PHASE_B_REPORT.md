# Phase B implementation report — 2026-10-08

Implemented the ordinary Python CLI through **READY_FOR_TRAINING**. It does not implement or invoke training/evaluation/report visualizations. Canonical compiler, native SONIC/DDS, existing V4 dataset/model/loss/training files and source policy were not modified.

## Retention decision

**Case B, no compiler change.** Rechecked current production cleanup code and actual latest accepted production artifact, not merely its five-step tensor shapes. The accepted production batch episode retains only samples/manifest/summary/provenance; full canonical sequences are not embedded or linked as a same-run companion. Compiler internally creates them, and supported audit mode retains them. Existing audit output provides full 816-row token/reference/observation tables. New V4 workflow selects audit retention; compact-only inputs fail with `INSUFFICIENT_CANONICAL_RETENTION`. Historical full outputs sharing an episode ID are not substituted for compact outputs from another replay. AUDIT.md has an additive recheck section; original audit evidence was preserved.

## Added interfaces/files

- `grasp_pipeline/{__init__,__main__,cli}.py`: compile, dataset build, run, resume, status, explicit JSON config, dry-run, machine-readable JSON status/errors. No train/eval commands.
- `contracts.py`: versioned checksummed contracts, atomic JSON writes, completion seal and validation; readable artifact permissions in shared Docker workspace.
- `state.py`: process-held lock, attempts, UTC timestamps, immutable finalized records, interruption recovery, checksum validation on reuse, dependency milestones separate from execution status.
- `adapters.py`: existing compiler subprocess invocation and authoritative accepted-artifact validation; explicit partial batch handling. Interruption cleanup includes descendants that create separate sessions; it does not alter robotics control code.
- `dataset.py`: invokes existing `v4.dataset.CanonicalV4` for every episode; serializes its exact seven tensors, original sample mapping/indices, source hashes/metadata, per-episode ranges and explicit split. Existing canonical schema normalizes legacy metadata; no new label/indexing implementation.
- `tests/test_pipeline.py`, `test_adapters.py`, `test_v4_saved.py`: lightweight state/CLI/mock compiler and real saved-fixture tests.
- `PHASE_B_USAGE.md`, `experiment.schema.json`, `dataset.schema.json`, `attempt.schema.json`: usage and formal v1 contracts.

Only the new package, Phase B docs, additive AUDIT.md clarification and new verification outputs were written. No expensive data duplication: immutable canonical episodes are referenced and checksummed; model-ready tensors are materialized.

## Behavior

Explicit split is mandatory: seeded validation count, explicit episode ID lists, or explicit all-train. No 19-episode or four-validation-episode assumptions. End tails drop exactly as CanonicalV4 does. Left-hand V4 semantics default to error; explicit exclusion records the incompatibility. Right-hand phase/progress definitions remain untouched.

Completed datasets are sealed and never overwritten. Resume validates completion hashes and contracts, then reuses the existing attempt. Changed completed artifacts block reuse; retry creates a new output/attempt rather than altering historical results. Abandoned RUNNING attempts become interrupted FAILED records only after the process lock is free. Normal interruption cleans up the compiler child tree.

Compiler partial batches remain FAILED, with original per-episode errors. Default policy blocks dataset. Explicit accepted-subset policy can build the valid subset while preserving compile FAILED and overall exit 2. Task-unsuccessful, validation-failed and execution-failed outcomes are separately recorded when compiler evidence exposes them; an unsuccessful task is never invented from an exception string.

Training and dataset audit remain NOT_STARTED, training_authorized=false. Resume cannot grant training authorization. READY_FOR_TRAINING denotes completion of preparation only.

## Verification actually executed

- CLI help, argument errors and compile/run dry-runs; no heavy subprocess on dry-run.
- **19 automated tests passed** (final log: `outputs/grasp_pipeline_phase_b/tests.log`). Coverage: manifest/state validation, immutable outputs, checksum tampering, busy lock, interrupted recovery, fresh retry, explicit split, compiler success/partial rejection/explicit subset via mocks, resume, synthetic process-tree cleanup, real V4 tensors, compact rejection and left-task rejection/explicit exclusion.
- Actual CLI on current full accepted `audit_regression`: 152 usable samples from 161 canonical anchors; READY_FOR_TRAINING with no training.
- Actual CLI on **two saved accepted episodes** (`kimodo_episode_0001` and `kimodo_episode_0004`): **304 samples**, deterministic seed 4 / val-count 1; train 0001 = 152, validation 0004 = 152. No duplicate or near-duplicate diversity audit was introduced; these are preparation fixtures, not evidence of dataset diversity.
- Independent baseline construction for each episode compared every serialized RGB/body/hand/body_tokens/hand_ref/phase/progress element and dtype using exact equality (**tolerance 0**). Original keep order, hand start rows and all mapping records also match exactly. Shape checks alone were not used.
- Actual resume of the multi-episode experiment reused attempt 1 and left all completed dataset file hashes unchanged. Training remained NOT_STARTED.
- Current canonical source policy check passed with no source warnings; core compiler/runtime was not changed.

Evidence: `outputs/grasp_pipeline_phase_b/verification.json`, `verification.log`, `tests.log`, `multi_final_cli.log`. Final dataset:

```
outputs/grasp_pipeline_phase_b/workspace/artifacts/datasets/dataset_saved_multi_v4_final_0001/
```

Experiment:

```
outputs/grasp_pipeline_phase_b/workspace/experiments/saved_multi_v4_final/experiment_manifest.json
```

## Not executed / limitations

- New reference → physical compiler batches were **mocked/dry-run only**. No new native SONIC rollout, GPU workload, VA training, offline inference or closed-loop evaluation was launched.
- Compiler adapter wiring is tested against mocked accepted/partial outputs; a fresh physical end-to-end run remains separately authorized work.
- Full audit retention is larger and performs full QA. Supporting compact-only V4 sources would require a separate retention contract; Phase B intentionally makes no compiler change.
- Input selection supports one full episode or direct batch episode directories, including explicit directory symlinks. Already merged canonical manifests are not treated as full episodes.
- Builder materializes CPU tensors in memory; no streaming/massive-scale optimization in this phase.
- Compiler retries currently run into a fresh batch attempt rather than automatically merging earlier physical attempts. Completed whole stages are reused; explicit accepted-subset policy avoids hiding partial failures.
- Left-hand-specific V4 label definitions, training loader integration, visual dataset audits and evaluation/report stages remain outside Phase B. Per-episode moving-joint masks are recorded for future trainer integration.
