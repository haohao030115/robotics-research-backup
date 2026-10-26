# Phase E — Kimodo-19 end-to-end acceptance (2026-10-09)

The real Kimodo-19 workflow was executed in a new versioned root:
`outputs/grasp_pipeline_phase_e_20261009_v1/`.

## Dataset compatibility

The merged input contains 19 source sample artifacts in `training_manifest.json`. All 19 source IDs are present and resolved to their original accepted canonical episode directories; no canonical sequence was fabricated and no index was changed. The upstream manifest explicitly lists `kimodo_episode_0015` as excluded, so it is not silently treated as a missing member of this 19-episode dataset.

The smallest adapter was added to `grasp_pipeline/dataset.py`: an old merged manifest with `sources` is expanded to its original episode roots, source sample IDs are checked against the merged NPZ row-for-row, and source hashes are retained in the new manifest. The resulting dataset has 19 episodes and 2,888 usable V4 samples. The exact historical split was available and used:

- train: 15 episodes, 2,280 samples
- validation: 4 episodes, 608 samples
- producer: Kimodo (19)
- task arm: right (19)
- cube XY: available; all `[0.4, -0.15]`

Dataset build and audit succeeded. Audit is descriptive only and marks unavailable wrist fields explicitly.

## Training

Command:

```bash
python3 -B -m grasp_pipeline train \
  --dataset outputs/grasp_pipeline_phase_e_20261009_v1/datasets/dataset_kimodo19_0001 \
  --output outputs/grasp_pipeline_phase_e_20261009_v1/training_run \
  --config va_training/configs/token_v4.json
```

The adapter now supplies the dataset to the existing `v4.train_merged` optimization loop. Model, forward pass, objectives, AdamW, gradient clipping, train-only normalization, seed, scheduler, checkpoint score and gate are unchanged. The run used MUSA `MTT S4000`, random initialization, and stopped at epoch 50 because the original training gate passed.

Authoritative offline metrics from the saved checkpoint:

- token MAE: `0.00371823`
- hand MAE: `0.00364718`
- phase accuracy: `1.0`
- progress MAE: `0.00335266`
- offline gate: `true`

Training report: `outputs/grasp_pipeline_phase_e_20261009_v1/training_run/report.html`.

## Open-loop evaluation

The new checkpoint was evaluated through the CLI and the existing V4 evaluator. The open-loop stage succeeded and wrote metrics, predictions and HTML. Validation overlap is explicitly recorded; this is not an independent held-out claim because the report evaluates all 19 episode tensors.

Artifacts:

- `open_eval/metrics.json`
- `open_eval/predictions.npz`
- `open_eval/report.html`

## Closed-loop evaluation

Because the newly trained checkpoint passed the existing offline gate, one real isolated-network closed-loop attempt was made with an explicit Kimodo scenario. Native SONIC, DDS and MuJoCo initialized with hardware EGL (`Moore Threads`, `MTT S4000`). The runtime then failed closed at tick 289/810:

`RuntimeError: Native token consumption mismatch at tick 289; strict abort`

The failure is recorded as a runtime failure, separate from grasp success. The saved partial physics trace has 289 recorded ticks, `max_lift_m=0.0`, `max_continuous_hold_seconds=0.0`, and `physical_grasp_success=false`. No retry was made. The wrapper status was corrected to `FAILED` for this incomplete runtime, while all logs and evidence remain preserved.

Artifacts:

- `closed_eval/execution_status.json`
- `closed_eval/report.json`
- `closed_eval/metrics.json`
- `closed_eval/report.html`
- `closed_eval/physics.npz`
- `closed_eval/sonic.log`, `dds_python.log`, `failure_traceback.txt`
- `closed_eval/renderer_worker.json`
- `closed_comparison/expert_vs_v4.mp4` (partial, exact available ticks only)
- `closed_comparison/video_metadata.json`

The partial video is evidence of the saved trace; it is not a completed 810-tick rollout.

## Experiment summary and verification

Summary: `outputs/grasp_pipeline_phase_e_20261009_v1/experiment_summary/index.html`.

The lightweight regression suite ran after the adapter and status fix: 21 tests passed, 3 existing environment-dependent tests skipped. No Codex or LLM service is needed for normal CLI operation. Historical canonical data, checkpoints and prior experiment outputs were not overwritten; all new artifacts are under the versioned Phase E root.

Remaining issue: the new model’s closed-loop runtime has a native token-consumption mismatch at tick 289. Resolving that requires a separately scoped native scheduler/runtime investigation; this phase does not alter DDS semantics or retry until grasp success.
