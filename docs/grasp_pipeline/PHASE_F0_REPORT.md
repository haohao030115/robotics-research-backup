# Phase F.0 — Production training retention (2026-10-09)

Implemented an explicit model-agnostic retention mode for the existing native canonical compiler:

```bash
python3 -B -m canonical_va.pipeline \
  --reference <reference> --output <fresh-output> \
  --mode production --retention training --gpu 0
```

`compact` remains the default. The new `training` mode is also passed by the batch runner and `grasp_pipeline` compiler adapter.

## Change

`canonical_va.production.training_success()` stages and validates the retained files before pruning. It keeps the source reference package, canonical `sonic_encoder.npz`, `references.npz`, `observations.npz`, metadata/provenance, V4 sample archive and mapping, and every RGB frame referenced by the mapping. Forensic controller/DDS/physics traces are removed only after staging succeeds. The retention mode is recorded in `provenance.json` and `validation_summary.json`.

No SONIC, DDS, MuJoCo, `TabletopDDS.step`, indexing, target mapping, QA policy or physical criterion was changed.

## Verification

The complete canonical unittest suite passes: **21 tests**. It includes a synthetic atomic-retention test. An isolated copy of the historical full episode preserved byte-identical canonical tables, samples and mapping (`exact_preservation=true`) while removing forensic traces.

One real native production run was executed against the known-compatible Kimodo reference. It completed with `expert_valid=true`, `physical_success=true`, 816 encoder/reference/observation rows and 161 samples. Hardware EGL was used: `GL_VENDOR=Moore Threads`, `GL_RENDERER=MTT S4000`, `GL_VERSION=4.5 build 57493aa1-dirty`; software rendering was false. Total runtime was 88.286 s.

The retained episode was consumed successfully by the existing V4 dataset builder and dataset audit CLI. The audit produced `report.html` and `summary.json`; no training, 50-reference batch, or closed-loop evaluation was run in this phase.

## Artifacts and storage

- Real run: [`production_training`](../../outputs/grasp_pipeline_phase_f0_verification/production_training)
- Dataset build: `outputs/grasp_pipeline_phase_f0_verification/dataset/dataset_exp_c5994a0acd20_0001`
- Dataset audit: `outputs/grasp_pipeline_phase_f0_verification/audit/report.html`
- Native log: `outputs/grasp_pipeline_phase_f0_verification/native.log`
- Training-retained size: 8,825,243 bytes
- Historical full-audit size: 128,798,929 bytes
- Existing compact accepted example: 1,746,908 bytes

The previously started even-50 attempt was stopped and its partial logs/artifacts remain untouched; it was not resumed.
