# Phase F.1 — 50-reference fast production (2026-10-09)

The numeric even IDs were selected from `reference_episode_002` through `reference_episode_100` (step 2), with all four required source files present. The immutable selection manifest is at `outputs/grasp_pipeline_phase_f_even50/selection/selection_manifest.json`.

## Production

The real command used was:

```bash
python3 -B -m canonical_va.batch \
  --references-root outputs/grasp_pipeline_phase_f_even50/selection/references \
  --output-root outputs/grasp_pipeline_phase_f_even50/production_training_attempt_0002 \
  --backend native --gpu 0 --mode production --retention training \
  --render-profile auto --qa-rate 0
```

All 50 references were enumerated serially with per-episode native isolation and failure continuation. Results: **44 PASS, 6 FAIL, 0 SKIP**. Failed IDs were `004, 036, 050, 052, 074, 090`; each completed native execution but failed the unchanged physical/dataset gate, and its evidence remains in the episode directory. No failed episode was retried. Aggregate recorded episode time was 3348.91 seconds (44 accepted episodes); batch wall-clock from first log creation to final summary was approximately 3837 seconds.

Production output: `outputs/grasp_pipeline_phase_f_even50/production_training_attempt_0002/`.

## Dataset and audit

With explicit accepted-subset authorization, the existing builder consumed the 44 accepted training-retained episodes using a deterministic 9-episode validation split (`seed=0`): 35 train episodes / 9 validation episodes, 5250 / 1350 samples, 6600 total V4 samples. Exact IDs are recorded in `dataset_exp_df29637d0357_0001/split.json`; the split is a deterministic count split, not a historical Kimodo split.

Dataset: `outputs/grasp_pipeline_phase_f_even50/dataset_attempt_0002/dataset_exp_df29637d0357_0001/`.

Audit succeeded:

`outputs/grasp_pipeline_phase_f_even50/dataset_audit_attempt_0002/report.html`

## Training and evaluation continuation

After explicit authorization, baseline V4 training ran on the accepted subset and stopped at epoch 50 with the original gate true. The checkpoint is `outputs/grasp_pipeline_phase_f_even50/training_attempt_0001/checkpoint.pt`; the report is `training_attempt_0001/report.html`.

Open-loop evaluation succeeded. Its report and metrics are under `outputs/grasp_pipeline_phase_f_even50/open_eval_attempt_0001/`; validation overlap is recorded in `metrics.json`.

One closed-loop attempt was made with the new checkpoint and the compatible Phase D.1 tabletop scenario. Hardware EGL initialized as Moore Threads / MTT S4000, but native SONIC exited with code `-6` before the first tick (`recorded_ticks=0/800`, `runtime_succeeded=false`, `grasp_success=false`). The run is marked FAILED and all logs/evidence are retained under `outputs/grasp_pipeline_phase_f_even50/closed_eval_attempt_0003/`. No retry was made.
