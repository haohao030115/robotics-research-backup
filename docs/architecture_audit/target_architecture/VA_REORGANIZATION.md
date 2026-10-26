# VA reorganization

VA Research owns model definitions, training, closed-loop rollout adapters and evaluation. It must not own MuJoCo scene construction, DDS setup or canonical reference generation.

## Target structure

```text
va_research/
  models/
    baseline_flow/
    token_v2/
    token_v3/
    v4/
    v4_1/
    bcz_hand/
    oracle/                 # diagnostic only
  datasets/                 # contract-driven readers
  training/
  rollout/                  # policy-to-runtime adapters
  evaluation/               # offline and runtime reports
  registry.py
  cli.py
```

## Existing mapping

* `va_training/models`, `v3`, `v4`, `v4_1`, `v4_0_bcz_hand`, `v4_oracle` become registry entries, not one merged model. Existing imports remain compatibility shims.
* `v4/model.py:24-58` remains the V4 contract: RGB `[B,3,128,128]`, body 68D, hand 28D; outputs 10×64 body tokens, 50×14 hand references, phase and progress.
* `v4/dataset.py:43-125` remains the authoritative V4 reader until the shared contract has parity. It uses explicit canonical IDs, 50 Hz and wxyz.
* `v4/training.py`, `v4/train_merged.py` and `grasp_pipeline/training.py` become training entry-point wrappers. MUSA/container requirements remain unchanged.
* `v4/evaluation.py` and `grasp_pipeline/evaluation.py` remain separate offline reports. `v4/rollout.py`, `one_shot_rollout.py` and BCZ rollout become `RolloutBackend` implementations; native SONIC remains outside training.

## Model registry record

Each family records model ID/version, input schema, output schema, dataset schema, normalization, checkpoint hash, training command, supported rollout backend, offline gate and known limitations. V4.1/BCZ additions must declare changed output heads and checkpoint compatibility. Oracle is diagnostic and cannot certify learned behavior.

## Evaluation gates

Reports must contain independent `schema_pass`, `offline_pass`, `runtime_pass`, `task_pass` and `overall_pass`. Runtime completion does not imply grasp success; the existing BCZ report demonstrates this. Closed-loop code keeps the absolute-tick `PlanBuffer`, protocol-v4 adapter, renderer gate, native token-consumption comparison and hand PD unchanged.

## Migration acceptance

For each family: checkpoint load parity, output shape parity, offline metric parity, provenance parity and (where applicable) rollout trace parity. Roll back by selecting the existing family-specific entry point. No model architecture, loss, normalization, target horizon or checkpoint key changes are part of organization work.
