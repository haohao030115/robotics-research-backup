# Vision-Action training

Independent project at `/home/group2/va_training`, visible inside the existing
`va-train` container as `/workspace/group2/va_training`. ARDY, Kimodo and SONIC are
not dependencies and are not modified. There is no real dataset in this project
yet. All first-milestone measurements use labeled synthetic fixtures.

## Environment and entry points

Use the validated d0930 container, PyTorch 2.9.0, torch_musa 2.9.0+8432052,
MUSA toolkit 4.3.6, host driver 3.3.5-server, container toolkit 2.2.0 and enabled
Intel IOMMU. No package installation is required for these scripts. The requirements
file records the runtime dependencies; the torch_musa wheel is vendor-specific.
Do not install public PyTorch wheels over this environment.

```bash
docker exec -it va-train bash
cd /workspace/group2/va_training
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python3 -B -m unittest discover -s tests -t . -v
```

Standard-library `unittest` requires no pytest installation. Root entry points:
`train_baseline.py`, `train_flow.py`, `eval.py`. Configuration is JSON, with paths
relative to the project root unless absolute. CLI overrides are recorded in the
resolved config. `batch_size_per_rank` multiplied by world size is the effective
batch size; change learning rate deliberately when changing it.

## Data contract and integrity (Stage 0)

Schema v3 has not been uploaded. The NPZ adapter supports explicitly configured
keys; default names in `configs/*.json` are proposed adapter names, not a claim
about the actual export. Adapt these keys and verify clocks/mapping semantics
against the actual exporter before using real data. `allow_pickle=False` is used.

A manifest:

```json
{
  "episodes": [
    {"id": "episode_001", "episode_path": "episode_001/states.npz",
     "rgb_path": "episode_001/images.npz", "split": "train",
     "instruction": "Pick up the red cube and lift it from the table."},
    {"id": "episode_002", "episode_path": "episode_002/states.npz",
     "rgb_path": "episode_002/images.npz", "split": "val"}
  ]
}
```

NPZ paths are relative to the manifest. Every episode must have a distinct ID and
distinct source files. Provide explicit splits for all episodes, or omit all
splits to use seeded episode hashing. Individual frames are never split randomly
between train and validation. The saved `split.json` freezes the actual selection.
For one-episode overfitting validation is absent and explicitly reported as such.

The preferred manifest is training_manifest.json. Legacy states/images aliases
are supported. rgb_reference_frame_indices may be in a manifest entry or
images.npz; simultaneous sources must agree. See datasets/SCHEMA_V3.md for
directory paths and source-episode grouping.

State NPZ has timestamps in seconds, actual `body_q [T,29]`, `hand_q [T,14]`,
reference `body_ref_q [T,29]`, `hand_ref_q [T,14]`. Optional dq arrays are checked
if keys are configured but are not model inputs. Images NPZ has true RGB arrays
`[N,height,width,3]` (uint8 or float [0,1]), capture timestamps and an optional
integer state-frame mapping.

With `alignment=reference_frame_mapping`, configure
`mapping_semantics=rgb_capture_state_index` ONLY if the mapping means the state
index at image capture, not an index of a desired future waypoint. The adapter
cross-checks indices against timestamps and rejects ambiguous or inconsistent
mapping. If that exporter mapping has different semantics, use
`alignment=nearest_timestamp` after confirming clocks are synchronized. An
explicit `rgb_time_offset_seconds` handles a known clock offset. The adapter
rejects alignment errors beyond `alignment_tolerance_seconds` and state periods
inconsistent with 50Hz. Do not increase tolerances to conceal unsynchronized data.

For each RGB frame aligned to state index `a`:

* observation is actual `concat(body_q[a], hand_q[a])`;
* target is `concat(body_ref_q, hand_ref_q)[a:a+H]`;
* incomplete tail chunks are excluded; H=20 at 50Hz spans 0.00 to +0.38 seconds.

Initial samples use RGB frames as anchors (typically 10Hz); action chunks contain
50Hz reference samples. No repeated/interpolated RGB is silently synthesized.
States are resident in memory; RGB NPZ files are validated one episode at a time
and loaded lazily with a bounded one-episode cache per dataset/worker. Compressed
NPZ is not random-access storage; convert to indexed storage later for very large
datasets. Start with 1–2 episodes rather than loading a huge archive.

Statistics come exclusively from the training samples: resized RGB channel
statistics, actual state mean/std and future action mean/std. Action chunks weight
overlapping reference frames by how often they occur in the training set. A
minimum std prevents division by zero. Body and hand statistics are also recorded
separately. Statistics are saved alongside config and embedded in checkpoints and
model buffers. Validation does not update statistics.

```bash
# First set data.manifest and confirm keys in a copied experiment config.
python3 scripts/check_dataset.py --config configs/my_experiment.json \
  --output outputs/my_experiment/integrity.json
```

## Policies

Both policies share a lightweight CNN returning spatial tokens with learned
positions and a LayerNorm/MLP encoding current 43-D actual state. Baseline pools
visual tokens, fuses them with the state and regresses `[B,H,43]` using L1.
The Flow policy has one token per action timestep, `Linear(43,d_model)`, trajectory
positions and sinusoidal flow-time embeddings, then pre-norm self-attention and
cross-attention to observation tokens. Default DiT is d=256, depth=6, heads=8;
all dimensions/depth are configurable. Attention uses ordinary matmul/softmax,
not fused CUDA kernels or FlashAttention.

Conditional flow matching: x0 ~ Normal(0,I), t ~ Uniform(0,1),
xt=(1-t)x0+t*x1, target velocity=x1-x0, MSE loss. Training x1 and ODE sampling
are in normalized action space. The separate FlowScheduler performs explicit
Euler integration from t=0 to t=1 (default 16 steps). Sampling returns the full
trajectory, not a single action. Optional language uses a small UTF-8 byte
embedding as a task-label interface; it is disabled by default and is not a
pretrained language-understanding model. No VLM or weights are downloaded.

Inference after loading a checkpoint:

```python
from utils.checkpoint import load_policy
policy, checkpoint = load_policy("outputs/run/checkpoints/latest.pt", device)
action_chunk = policy.predict(image, body_q, hand_q, instruction=None)
body_reference = action_chunk[..., :29]  # abstract body reference
hand_reference = action_chunk[..., 29:]  # abstract hand reference
```

`image` must already be resized `[B,3,image_size,image_size]`, float [0,1].
`body_q [B,29]` and `hand_q [B,14]` are physical current state. predict handles
normalization internally and returns inverse-normalized reference joints
`[B,H,43]`. Input tensors are moved to the policy device. This project provides
no actuator dispatch or closed-loop robot integration; body and hand controllers
remain separate.

## Staged training

Stage 1: baseline overfit 1–2 train episodes. Stage 2: baseline small dataset with
held-out episodes. Stage 3: Flow tiny-subset overfit. Stage 4: Flow full training
only after earlier checks pass. Stage 5: separately designed closed-loop evaluation.
No script automatically starts real-data or full training.

```bash
python3 train_baseline.py --config configs/my_baseline.json
python3 train_flow.py --config configs/my_flow.json
python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  train_baseline.py --config configs/my_baseline.json
# Same launch form with train_flow.py for Flow.
python3 eval.py --checkpoint outputs/my_run/checkpoints/latest.pt --device musa
```

LOCAL_RANK chooses the GPU; installed torch_musa registers MCCL. CPU uses gloo
for optional development tests. Training uses DistributedSampler, set_epoch and
rank-reduced losses. Validation partitions without padding, uses the unwrapped
model to avoid mismatched collectives on uneven ranks, then reduces totals/counts.

The installed d0930 PyTorch 2.9 build raises `reducer_->timer_` in its optional
DDP runtime timing logger from the second forward. `utils/distributed.py` wraps
that logger locally to skip timing collection on MUSA. Native DDP initialization,
reducer/buckets, backward gradient communication and optimizer behavior are
unchanged; training metrics are still logged. No installed package is modified.
`training.disable_musa_ddp_runtime_timing=true` enables the workaround. The smoke
test runs multiple steps and checks parameter consistency across ranks to cover
this issue; a single-step all_reduce test does not detect it. Retest after runtime
upgrades before disabling the workaround.

Each run directory contains resolved config, train-only normalization, episode
split, JSONL logs and atomic `latest.pt`/epoch checkpoints. Logged values include
train/val loss, physical body/hand/action L1, grad norm, lr, epoch and global step.
Flow also logs normalized FM MSE, t mean/min/max and predicted/target velocity
norms. Flow training body/hand L1 and `action_l1_proxy` are one-step endpoint
diagnostics; validation action L1 comes from actual Euler sampling on the first
configured number of validation batches per rank. Offline L1 is not closed-loop
success. No data means real performance is unknown.

Checkpoint fields: model, optimizer, cosine scheduler, epoch, global_step, config,
normalization_stats, resume_progress and per-rank CPU/MUSA/Python/NumPy RNG states.
New format-v2 checkpoints restore the next batch inside an epoch and the original
scheduler plan. Sample order is seeded by epoch; validation RNG is isolated from
training. Exact resume requires unchanged world size, model, source/split, batch
size and training schedule. Use `--stop-after-steps N` to interrupt a short run
without changing planned max_steps. Legacy checkpoints without progress keep
epoch-level resume behavior. A new schedule needs a separate experiment.
Existing checkpoints are never
overwritten by a new run unless explicitly resumed. Pick separate output paths
for new experiments.

## Short synthetic verification only

```bash
python3 scripts/make_synthetic.py
python3 scripts/check_dataset.py --config tests/fixtures/synthetic/baseline_smoke.json \
  --output outputs/synthetic_integrity.json
python3 train_baseline.py --config tests/fixtures/synthetic/baseline_smoke.json
python3 train_flow.py --config tests/fixtures/synthetic/flow_smoke.json
python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/run_ddp_smoke.py --config tests/fixtures/synthetic/baseline_smoke.json --steps 2
```

Synthetic loss decrease verifies that learning can start; it is not evidence of
real expert-data accuracy. Real schema validation, real episode overfitting,
long-duration DDP stability and robot closed-loop evaluation remain pending.

First-milestone evidence is stored in `outputs/milestone.json` with links to
per-rank logs, synthetic training checkpoints, default-size model checks and
dataset integrity results. The DDP regression runs at least two iterations.

## Model-only inference and latency

```python
from models.inference import ModelOnlyPolicy, split_action_chunk
interface, checkpoint = ModelOnlyPolicy.from_checkpoint(path, device)
action_chunk = interface.predict(head_rgb, body_q, hand_q)  # [20,43] for one observation
body_ref, hand_ref = split_action_chunk(action_chunk)       # [20,29], [20,14]
```

NumPy images use HWC/BHWC; tensors use CHW/BCHW. uint8 is scaled by 255;
float images must be [0,1]. The interface resizes and uses saved statistics.
No joint mapping, dispatch, simulator or controller implementation is imported.

```bash
python3 scripts/offline_predict.py --checkpoint outputs/run/checkpoints/latest.pt \
  --output outputs/run/prediction
python3 scripts/verify_readiness.py --output outputs/new_short_verification
python3 scripts/benchmark_flow.py --output outputs/flow_latency.json
```

Offline inference loads one sample and validates [20,43], finite values and inverse
normalization against checkpoint statistics, then saves npz/json. Readiness uses
synthetic data, three steps per model/device mode, stop/resume, validation and
prediction exports. The benchmark uses default 7.146M-parameter Flow, batch=1,
128px RGB, FP32, Euler steps 4/8/16, warmups and synchronized wall-clock timings.
Without --checkpoint its weights are randomly initialized; it measures execution
cost, not task quality. End-to-end model timing starts with resident device tensors
and excludes disk decoding, external camera preprocessing, transfer and controls.
