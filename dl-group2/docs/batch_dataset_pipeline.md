# XY-only expert dataset production

`produce_experts.py` samples a finite, seeded list of positions, saves the list
before execution, and processes each position once with bounded replay/correction
budgets. It never resamples a failed position until success. Source failure,
physical failure, incomplete synchronization and packaging failure remain visible.
A corrected candidate always needs another actual SONIC + independent-hand replay.
Safety/synchronization failures stop calibration of that sample.

Use the simulation environment on an allocated GPU node:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl
python scripts/produce_experts.py \
  --output outputs/expert_datasets/xy_random_v1 \
  --num-episodes 30 \
  --x-min 0.38 --x-max 0.40 --y-min -0.17 --y-max -0.13 \
  --seed 20261002 --max-replays 3 --max-corrections 2
```

`--num-episodes` counts requested candidate positions, not guaranteed successes.
Re-running the identical command resumes its saved plan/results. A changed
configuration is rejected; choose a new output directory for another finite batch.
Only cube XY varies. Neutral pose, object size/mass/friction, lighting, cameras and
table remain fixed. Schema v3 is retained; generation provenance adds
`generation_method` (`direct_expert`, `corrected_expert`, `failed_sample`),
`correction_iterations`, `replay_attempts`, `source_candidate`, and
`selected_candidate`. An environment fingerprint records fixed scene parameters.

Outputs:

- `sampling_plan.json`: every requested position and seed/configuration.
- `results.json`, `summary.json`: successes and failures, attempts, correction
  iterations, direct/corrected counts, averages and a 3×3 spatial coverage histogram.
- `episodes/<case>/`: unchanged v3 reference, actual, source, candidates, SONIC,
  contacts, validation records and RGB layout.
- `training_manifest.json`: only complete physically certified episodes with
  checked RGB/reference/state alignment. Paths are relative to the manifest.
- `failure_manifest.json`: unsuccessful samples and packaging failures.
- `logs/<case>/`: source, replay, correction, rendering and export process logs.

The training manifest gives episode paths, schema/validity, initial XY, selected
validation rollout, state/reference length, RGB archive path and explicit RGB
reference-frame indexes. Cube ground truth remains in the episode but is excluded
from default model inputs; overview RGB is evaluation-only. Failed replay RGB is
retained alongside its failure report.

## Dataset loader

```python
from scripts.vision_action_dataset import VisionActionDataset

train = VisionActionDataset(
    'outputs/expert_datasets/xy_random_v1/training_manifest.json',
    horizon=50, split='train', split_seed=0, tail='pad')
validation = VisionActionDataset(
    'outputs/expert_datasets/xy_random_v1/training_manifest.json',
    horizon=50, split='validation', split_seed=0, tail='pad')
sample = train[0]
# sample['input']: head_rgb uint8 [240,320,3], body_q float32 [29],
#                  hand_q float32 [14], instruction string
# sample['target']: body_ref_q float32 [50,29], hand_ref_q float32 [50,14],
#                   action_mask bool [50]
```

The class is a map-style NumPy Dataset compatible with PyTorch DataLoader; it does
not require importing PyTorch. RGB normalization/channel permutation is left to
the model transform. Each sample uses the image's **stored reference_frame**, not
an estimated timestamp conversion: current states and the first action target
refer to precisely that frame. Future targets are contiguous 50 Hz reference
frames, while sample anchors occur at 10 Hz camera frames.

At the end, `tail='pad'` repeats the final action and marks padded entries false in
`action_mask`; training loss must ignore those entries. `tail='drop'` omits anchors
without H future frames. Horizon may exceed episode length with padding. The
bounded cache holds at most two decoded head-RGB episodes by default.

Splits use complete episodes and a deterministic seed; the loader rejects overlap
or missing episode IDs in an explicit `split_file`. No adjacent frames of an
episode are randomized between training and validation.

```bash
python scripts/audit_expert_dataset.py \
  --manifest outputs/expert_datasets/xy_random_v1/training_manifest.json --horizon 50
python scripts/vision_action_dataset.py \
  --manifest outputs/expert_datasets/xy_random_v1/training_manifest.json --horizon 50
python tests/test_vision_action_dataset.py
python tests/test_production_limits.py
```

The dataset audit writes `dataset_audit.json` and reproducible `splits.json`, checks
all episode physical/data certificates, common neutral pose and fixed task
parameters, validates real first/last action chunks, masks and episode split
isolation. The synthetic test also rejects timestamp mismatch and leaking splits.
No model training is started.
