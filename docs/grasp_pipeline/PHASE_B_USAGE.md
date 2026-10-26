# Phase B CLI and contracts

Run from the repository root in the existing `va-train` environment. CLI help/dry-run/state use Python standard library only; V4 build additionally uses existing numpy, torch and Pillow. No LLM, service, database, new model, or training dependency is introduced.

## Prepare existing canonical data (safe, no physical execution)

```bash
python -m grasp_pipeline run \
  --canonical-root /path/to/full_accepted_batch \
  --workspace /path/to/workspace --experiment exp_001 \
  --val-count 4 --seed 4 --stop-before train

python -m grasp_pipeline status --workspace /path/to/workspace --experiment exp_001
python -m grasp_pipeline resume --workspace /path/to/workspace --experiment exp_001 --through dataset
```

Split is mandatory and explicit: choose `--val-count N --seed S`, `--split /path/to/split.json`, or `--all-train`. The last is an explicit overfit/no-held-out-validation dataset, not a generalization split. `--split` JSON contains `train_episodes` and `val_episodes` lists covering included IDs exactly. No 19/4 hardcoding. Split is episode-disjoint; seeded selection shuffles sorted episode IDs. Baseline tensors are kept unnormalized; later baseline training applies train-only mean/std with clamp_min(.05).

## Compiler adapter and full workflow

These are operational commands that **will launch existing native compiler batches**; they were only mocked/dry-run during Phase B verification. Normal production use does not need an LLM:

```bash
python -m grasp_pipeline compile --references-root /path/to/references \
  --output-root /path/to/canonical --workspace /path/to/workspace \
  --experiment compile_001 --mode audit --dry-run

python -m grasp_pipeline run --references-root /path/to/references \
  --canonical-output-root /path/to/canonical --dataset-output-root /path/to/datasets \
  --workspace /path/to/workspace --experiment exp_002 \
  --val-count 4 --seed 4 --stop-before train --dry-run
```

Remove `--dry-run` only when a real compiler batch is intended. Adapter invokes `python -B -m canonical_va.batch`; the existing batch owns native semantics/network namespaces. V4 `run` uses audit retention. Standalone `compile --mode production` is supported but compact output is explicitly rejected by V4 build.

```bash
python -m grasp_pipeline dataset build --canonical-root /path/to/full_canonical \
  --model v4 --output-root /path/to/datasets --workspace /path/to/workspace \
  --experiment dataset_001 --split /path/to/split.json
```

Every output-root is a parent, not an overwrite target: compiler writes `<experiment>_attempt_0001`, dataset writes `dataset_<experiment>_0001`. Retry uses attempt 0002. Existing incomplete outputs/logs remain. Use a fresh experiment ID to change configuration. `resume` reuses stored configuration, has only compile/dataset endpoints and cannot authorize training. Standalone compile experiments remain compile-only; use a new dataset command to consume them. There is deliberately no train/eval/report/visualization command in Phase B.

## Policies and errors

- `--partial-policy error` default: any compiler batch failure keeps compile FAILED and blocks dataset. The original batch summary includes all episode reasons.
- `--partial-policy accepted`: explicitly consume a **validated accepted subset** despite partial batch failure. Compile remains FAILED, dataset may succeed and reach READY_FOR_TRAINING, overall exit remains 2. The decision and original batch summary are preserved. Zero-exit malformed output never counts as partial success.
- `--incompatible-policy error` default: left/non-right task metadata rejects V4 label incompatibility. `exclude` explicitly excludes such episodes with recorded reason. It does not redefine right-hand labels or swallow other validation failures. No compatible episodes is an error.
- Insufficient full retention always errors. No padding/interpolation/repetition of future labels. End-of-episode samples are dropped exactly as existing CanonicalV4 does, with dropped-tail counts retained.
- Only compiler-accepted episodes are eligible. Duplicate canonical IDs error rather than silently combining them. This is identity consistency, not a diversity/near-duplicate audit. Input is a single full episode or a batch's direct episode directories, not an already merged canonical manifest.
- Exit 0: successful execution/artifact validation or dry-run. Exit 2: CLI/config/artifact/process failure or explicitly consumed partial batch. Exit 3: experiment busy. Interrupted attempts record exit 130. Exceptions and logs remain in attempt records.

## Explicit JSON config

All create commands accept `--config config.json`; CLI values override it. Paths in the config are resolved relative to its directory; CLI paths relative to current directory. Unknown config keys fail. Split contents and original config checksum are stored, so resume does not silently reread changed split rules. Example:

```json
{
  "canonical_root": "../canonical/full_batch",
  "workspace": "../workspace",
  "experiment": "exp_003",
  "model": "v4",
  "val_count": 2,
  "seed": 4,
  "partial_policy": "error",
  "incompatible_policy": "error"
}
```

`python -m grasp_pipeline run --config config.json --dry-run`

## Artifact contracts v1

Experiment `experiment_manifest.json`: artifact_type=experiment, schema_version=1, independent experiment_id, pipeline_version/code hashes, immutable recorded configuration, workflow milestone, training_authorized=false, and compile/dataset_build/dataset_audit/training stages. The experiment state index is mutable; finalized attempt records and sealed datasets are not rewritten.

Each `stages/<stage>/attempt_NNNN/attempt.json`: start/end UTC timestamps, inputs with content-derived identity and hashes, config, command/API entrypoint, state, exit code, output paths/checksums, errors with kind, and execution.log. Lock protects the entire invocation. Compiler child inherits the process lock; resume cannot start while it remains alive. Once an abandoned RUNNING attempt has no lock owner, recovery records interruption and a retry gets a new directory. Ordinary interruption terminates the spawned compiler process group before releasing the lock. Compiler retries currently launch a fresh batch; they do not relabel failed historical attempts or copy previous physical evidence as new execution.

Dataset `manifest.json`: artifact_type=v4_dataset, schema_version=1, dataset_id, builder version/source hashes/config, baseline source metadata, canonical IDs/content fingerprints, per-episode ranges, full source checksums (including RGB), renderer provenance where available, excluded records, and checksums of samples/split/index. `COMPLETED.json` seals all four files and is written only after construction; resume validates checksums and tensor contracts before reuse.

`samples.npz` contains **exactly the existing `CanonicalV4.data` keys**:

| Key | Shape | dtype |
|---|---|---|
| rgb | [N,3,128,128] | float32 |
| body | [N,68] | float32 |
| hand | [N,28] | float32 |
| body_tokens | [N,10,64] | float32 |
| hand_ref | [N,50,14] | float32 |
| phase | [N] | int64 |
| progress | [N] | float32 |

`index.jsonl` holds row, episode_id, canonical_artifact_id, original sample_id, original CanonicalV4 keep index and hand start row, plus original mapping unchanged. No new tensor semantics. `split.json` contains the exact split policy/seed/IDs/sample counts. Source canonical artifacts are referenced and hashed, not copied; model-ready tensors are materialized as the dataset artifact. One immutable dataset can be referenced by multiple future experiments/trainers. Training remains NOT_STARTED; READY_FOR_TRAINING is preparation completion, not authorization or a guarantee of diverse/generalizable data.
