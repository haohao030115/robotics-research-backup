# Phase D — Real training smoke test + evaluation integration

2026-10-08. Phase D adapters were added without changing canonical/native/V4 model or loss semantics.

## Implemented

- `python -m grasp_pipeline train --dataset DATASET --output RUN --config CONFIG` now performs the existing V4 model/loss/AdamW loop with explicit config, train-only normalization, checkpoint history and deterministic HTML report.
- `python -m grasp_pipeline eval offline --checkpoint CHECKPOINT --dataset DATASET --output DIR` reuses `V4Policy` and existing `v4.training.metrics`; writes predictions, JSON metrics, validation-overlap metadata, body/hand horizon metrics, phase confusion matrix and HTML.
- `python -m grasp_pipeline eval closed --checkpoint CHECKPOINT --scenarios SCENARIOS --output DIR` passes explicit scenario/checkpoint/output provenance to the existing native `v4.rollout` path. It does not edit checkpoints or replace SONIC/DDS/TabletopDDS. Runtime success and grasp success are stored separately.
- `summary.py` provides deterministic local index generation for saved reports.

## Real smoke verification

A real short GPU run was executed against the Phase B dataset in:

`outputs/grasp_pipeline_phase_d_smoke2/`

Configuration was an explicitly separate 2-epoch smoke budget. MUSA execution initialized and performed updates. The run saved:

- `config.json` with the 2-epoch budget and baseline loss weights
- `history.json` with train/validation metrics at epoch 1 and 2
- `status.json` with `finished=true`, `epochs=2`, `offline_gate=false`
- `checkpoints/checkpoint_epoch_0001.pt`, reloadable checkpoint
- `report.html` and training manifest/provenance

The smoke checkpoint did not pass the original offline gate. The gate was not weakened and this checkpoint was not used for closed-loop.

A real offline evaluation was then run successfully against the smoke checkpoint (temporary evaluation output `/tmp/d_off`; the CLI returned 0 and produced `metrics.json`, `predictions.npz`, `report.html`). Evaluation explicitly records validation/train episode overlap and does not call it independent held-out generalization.

## Closed-loop status

The required qualified-checkpoint substitution was prepared explicitly using the existing checkpoint from `outputs/va_v4_kimodo19`, its sibling `offline_metrics.json`, and an explicit scenario file. The wrapper entered the existing native path in an isolated network namespace, but this bounded attempt did not produce a runtime report or rollout artifact; only `scenario.json` and `evaluation_provenance.json` were written. It is therefore recorded as **not verified**, not as a successful rollout. No native controller or checkpoint was modified. The historical qualified closed-loop artifacts remain available under `outputs/va_v4_kimodo19/normal_gpu` and are not relabeled as a new run.

## Tests and limits

Phase C regression tests remain passing after the adapter changes. The smoke run exposed and fixed only integration issues (module path and hand moving-mask dimension); V4 architecture, forward pass, objective definitions, optimizer, normalization and target semantics were unchanged. No new compiler/native replay was launched. Closed-loop integration still needs a separately diagnosed launcher/runtime attempt before a video can honestly be claimed as produced by this CLI.
