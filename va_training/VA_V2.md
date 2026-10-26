# VA V2: native release-token baseline

V2 is available as `model_type: token_v2`; V1 reference policies and checkpoint meanings are preserved. The only change to an existing V1 module is the additional factory branch in `models/policies.py`. SONIC weights, deployment source, expert episodes and upstream repositories are unchanged.

## Modules and interfaces

| Files | Purpose |
|---|---|
| `configs/token_v2.json`, `utils/v2_config.py` | Separate configurable V2 path, relative paths resolved against this project |
| `datasets/tokens.py`, `scripts/validate_token_dataset.py` | Native-label extraction, causal alignment, all-episode audit |
| `models/encoders/structured_state.py`, `models/token_v2.py` | Named field encoders, masked GRU history, shared fusion and independent output heads |
| `models/token_inference.py`, `models/observation_history.py` | Checkpoint inference and measured live history |
| `losses/token.py`, `trainers/token_training.py`, `train_token_v2.py` | Weighted Huber/MSE token loss, hand MSE, staged training |
| `sonic/release.py`, `sonic/runtime.py`, `sonic/token_handoff.py` | Frozen bundle validation, existing v4 packer/native process and acknowledged body/hand pairing |
| `scripts/oracle_token_replay.py`, `scripts/evaluate_token_v2.py`, `scripts/run_token_closed_loop.py` | Oracle, held-out evaluation and gated live closed loop |
| `tests/test_token_v2.py` | Alignment, boundaries, model/checkpoint, projection, gates, live synchronization and real-data tests |

Default batched model inputs are RGB `[B,3,128,128]`, named history fields `body_q [B,3,29]`, `body_dq [B,3,29]`, `hand_q [B,3,14]`, `hand_dq [B,3,14]`, `base_quat [B,3,4]` (wxyz), `base_ang_vel [B,3,3]` (local), `previous_action [B,3,29]` (raw SONIC action), derived `projected_gravity [B,3,3]`, and boolean validity mask `[B,3]`. Each history row has 125 scalars. History is consecutive 50 Hz state; startup repeats the first row with invalid padding. Image preprocessing and VisionEncoder reuse V1; normalization is fitted on training episodes only. No augmentation is introduced.

Default outputs are raw regressed `sonic_token [B,64]` and independent absolute `hand_ref [B,14]` in radians. Configurable K gives `[B,K,64]` and `[B,K,14]`; current live scheduler requires K=1 and target offset=1. Optional projection rounds each token coordinate to `integer/16`, clamped to integers -16 through 15. Raw regression is the default; projection is an explicit inference ablation. Reports include aggregate and per-dimension token MAE/MSE, snapped FSQ accuracy and hand MAE/MSE.

## Exact label alignment

Read the selected validation replay's `policy_logs/token_state.csv` by **named** `token_0` through `token_63`. The first five columns are `index,time_ms,time_realtime_ms,time_monotonic_ms,ros_timestamp`; they never become token coordinates. Join token, action, encoder-mode and playing logs using identical metadata. Keep the contiguous active rows only: current episodes have 801 rows, 800 active and one inactive. Active row r maps to reference frame r, verified against complete selected replay `reference_frame` and clocks. Never regenerate labels through an encoder.

The published replay state/RGB is post-step, whereas native token logging is pre-action. Therefore an RGB anchor at reference t (0,5,...795) and measured state history ending at t supervise **native token[t+1] and hand_ref[t+1]**, with one common execution timestamp. K>1 extends consecutive future targets; incomplete tails are dropped. Previous action uses the native pre-action logger's previous action at anchor t, deliberately retaining that observable lag in both training and live inference. Recorded world-frame angular velocity is rotated into the body frame; gravity is derived deterministically from the unit quaternion. Samples and histories never cross episode boundaries.

All 32 selected episodes have 800 state/reference/token frames and 160 RGB frames, producing 5,120 K=1 samples. Split reuses V1's source-episode split function with seed 38: 26 train / 6 validation episodes. Existing historical V1 results may use other manifests or a single episode; those are not a matched controlled comparison.

## Native control and timing

`ExternalTokenAdapter` reuses SONIC's `pack_pose_message` with protocol v4, `token_state float32[1,64]` and `frame_index`. The native executable is launched with the release decoder and observation configuration, **without `--encoder-file`**. It maintains its own 10-frame decoder history and performs the existing ordering, offsets, scaling, PD and DDS commands. The hand stays on the existing independent controller.

Important repository discrepancy: native v4 logs `frame_index` but does **not** schedule execution by it. Rollout verifies the full token actually consumed in native control logs, then applies its paired hand reference on that observed tick. Late predictions are explicitly logged as late. Visual decisions are requested at 10 Hz and held at 50 Hz; intended next-tick deadlines are stricter than the 100 ms visual period. Unmatched tokens or missed control snapshots abort rather than fabricate history. Startup uses a measured initial snapshot at anchor -1 for execution 0. Native bootstrap contains only the initial pose, not an expert future trajectory.

The current MUSA checkout differs from dl-group2's pinned submodule. Runtime uses the explicit baseline robot XML in configuration; it does not edit upstream scene files. Concurrent SONIC simulations share DDS domain 0 on loopback: run oracle/closed-loop in an isolated network namespace as below.

Historical V1's bridge differentiates future joint references numerically and approximates future pelvis orientation using a constant/reference value. Until its complete reference interface is implemented, V1/V2 differ in interface fidelity and conditioning as well as representation.

## Reproduce

Run these host-shell commands from `/home/group2`. They require the existing privileged `va-train` container and installed Torch/MUSA, ONNX, MuJoCo, DDS and release binary. Use fresh output directories. This helper preserves the actual container environment used for tests/training:

```bash
run_v2() {
  docker exec va-train env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$@"
}
root=/workspace/group2
project=$root/va_training
out=$root/outputs/va_v2_repro

run_v2 python3 -B "$project/scripts/validate_token_dataset.py" \
  --config "$project/configs/token_v2.json" --output "$out/dataset_validation.json"

run_v2 env VA_V2_REAL_CONFIG="$project/configs/token_v2.json" bash -c \
  'cd /workspace/group2/va_training && python3 -B -m unittest discover -s tests -v'

run_v2 unshare --net bash -c 'ip link set lo up; exec "$@"' bash \
  python3 -B "$project/scripts/oracle_token_replay.py" \
  --config "$project/configs/token_v2.json" --episode-id xy_random_v1_random_0000 \
  --output "$out/oracle_0000"

run_v2 python3 -B "$project/train_token_v2.py" \
  --config "$project/configs/token_v2.json" --oracle-report "$out/oracle_0000/report.json" \
  --stage overfit --episode-id xy_random_v1_random_0000 --output "$out/overfit_0000"

run_v2 python3 -B "$project/train_token_v2.py" \
  --config "$project/configs/token_v2.json" --oracle-report "$out/oracle_0000/report.json" \
  --overfit-report "$out/overfit_0000/report.json" --stage heldout --output "$out/heldout"

run_v2 python3 -B "$project/scripts/evaluate_token_v2.py" \
  --checkpoint "$out/heldout/checkpoints/latest.pt" --split val --output "$out/heldout/offline_val.json"

run_v2 unshare --net bash -c \
  'ip link set lo up; source /workspace/group2/outputs/mujoco_gpu_rendering/enable_mthreads_egl.sh; exec "$@"' bash \
  python3 -B "$project/scripts/run_token_closed_loop.py" \
  --checkpoint "$out/heldout/checkpoints/latest.pt" --episode-id xy_random_v1_random_0001 \
  --oracle-report "$out/oracle_0000/report.json" --overfit-report "$out/overfit_0000/report.json" \
  --offline-report "$out/heldout/report.json" --output "$out/closed_loop_0001"
```

The existing EGL environment script is specific to this machine. Other machines should supply their own working MuJoCo renderer configuration. Add `--project-token` and use a new output directory for the projected-token ablation. Training supports CPU/MUSA single device; distributed V2/resume and chunk rollout are not implemented in this baseline. Each training and rollout output stores resolved configuration, SONIC paths/hashes, manifest and token-label provenance, and joint orders. Checkpoints retain training-only statistics and provenance.

## Recorded results

Artifacts are under `/home/group2/outputs/va_v2`, with JSON reports retaining complete per-dimension metrics and provenance.

| Test | Result |
|---|---|
| Oracle, random_0000 | PASS: 800/800 exact tokens, 50 Hz, body/hand skew 0, fall steps 0, physical grasp success; maximum lift 0.15012 m, final lift 0.14903 m, continuous hold 4.355 s |
| Stage A, 2,000 MUSA steps | PASS: token MAE 0.17307 → 0.003360, FSQ accuracy 99.73%, hand MAE 0.000237 rad |
| Stage B, 2,000 MUSA steps, 26/6 split | Offline gate PASS: held-out token MAE 0.007663, MSE 0.00013707, FSQ accuracy 96.65%, hand MAE 0.000732 rad; independent evaluation reproduced these metrics |
| Full regression suite | All 37 V1/V2 tests passed, including eight V2 tests, all 32 real episodes and protocol v4 payload |
| Stage C, raw tokens, held-out random_0001 | Task FAIL: 800/800 frames, actual 50 Hz, body/hand skew 0, fall steps 0, lift 0 m, hold 0 s. Median/max decision latency 24.51/30.43 ms; 0/160 visual-period misses, 160/160 next-control execution misses |
| Stage C, FSQ projection ablation, same episode | Task FAIL: 800/800 frames, 50 Hz, synchronized, no fall, lift/hold 0. Median/max latency 38.76/100.02 ms, 1/160 visual-period misses and 160/160 next-control misses; no demonstrated control improvement |

Passing offline thresholds establishes a baseline, not broad generalization or closed-loop success. See the closed-loop result report for actual task success, latency and deadline misses.

The remaining blocker is learned closed-loop task performance, not a missing implementation stage: oracle, overfit, held-out evaluation and live integration have executed. The measured 20 ms next-tick budget is exceeded by the live render/inference path. K=1 also holds a single token over multiple native ticks; offline labels do not validate that temporal behavior. State distribution drift and differences between hardware-rendered live images and saved expert images are additional hypotheses, not established causes. Short token chunks, timing alignment and additional closed-loop evaluation should precede any success claim.

Result files: `../outputs/va_v2/dataset_validation.json`, `oracle_0000/report.json`, `overfit_0000/report.json`, `heldout/report.json`, `heldout/offline_val.json`, `closed_loop_0001_isolated/report.json`, and `closed_loop_0001_projected/report.json` (all latter paths relative to `../outputs/va_v2`). Training checkpoint: `../outputs/va_v2/heldout/checkpoints/latest.pt`.
