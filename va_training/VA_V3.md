# VA V3 current-observation baseline

V3 lives alongside V2. No V2, SONIC decoder, SONIC observation packing,
SONIC history/reset, hand controller or expert dataset is modified.

## Implementation plan and reuse

The pre-edit inspection covered `datasets/tokens.py`,
`models/observation_history.py`, `diagnostics/chunk_rollout.py`,
`sonic/runtime.py`, `sonic/token_handoff.py`, `diagnostics/control.py`,
`run_sonic_grasp.py`, and the previous physical video protocol.

Reuse native-label validation and provenance, source-episode split, RGB resize,
release-token transport, native process/reset, physical scene and hand PD,
and rendering of recorded physical qpos/qvel. New files: `v3/{model,data,config,
training,inference,runtime,rollout}.py`, `configs/token_v3.json`,
`train_token_v3.py`, `scripts/{evaluate_token_v3,benchmark_token_v3,
run_token_v3_closed_loop}.py`, tests and this report. No existing files require
modification. The model factory intentionally remains independent of V2.

## Architecture

| Stage | Tensor |
|---|---|
| Current RGB, bilinear resize, round uint8 then /255 | B,3,128,128 |
| Conv 3x3 stride 2 + GroupNorm + GELU, channels 32,64,128,256 | B,256,8,8 |
| Flatten + learned spatial position | B,64,256 |
| Current proprio | B,96 |
| Linear 96→256, GELU, Linear 256→256, LayerNorm, GELU | B,256 |
| Query projection | B,1,256 |
| One residual 4-head cross-attention block + feed-forward | B,1,256 |
| LayerNorm, Linear 256→512, GELU | B,512 |
| Body: Linear 512→256, GELU, Linear 256→128 | B,2,64 |
| Hand: Linear 512→256, GELU, Linear 256→28 | B,2,14 |

1,525,148 trainable parameters. Raw native post-FSQ labels, deterministic
regression, no token snapping. Body Huber delta 0.0625 plus absolute hand MSE,
equal weights. Only training observations fit proprio mean/std (floor 0.001).
No history, previous actions, previous VA output, recurrent state, temporal
transformer, diffusion/flow/DiT, deformable attention or auxiliary task heads.

The 96 coordinates concatenate body q(29), body dq(29), hand q(14), hand dq(14),
base quaternion wxyz(4), local base angular velocity(3), projected gravity(3).
Joint lists are copied from validated episodes into checkpoint provenance.
Recorded world angular velocity is rotated into the local frame by the existing
validated loader. Live MuJoCo free-joint angular velocity already uses that
frame. `previous_action` appears only in physical diagnostic logs needed by the
existing video tool; it is never part of V3 model input or normalization.

## Timing and alignment

Let r be the recorded post-step snapshot index, and t=r+1 the next executable
native control index. Both body and hand targets are native rows [r+1,r+2].
The snapshot is the first physics step after observing native row r, matching
the original recorder; it is not a claim that the entire preceding 20 ms has
elapsed. Nominal reference timestamps and actual capture/execution clocks must
not be conflated.

| Observation index | RGB/proprio reference label | Slot 0 label/execution index | Slot 1 label/execution index |
|---|---|---|---|
| reset -1 | measured reset state | 0 | 1 |
| 1 | 0.020 s | 2 / 0.040 s | 3 / 0.060 s |
| 3 | 0.060 s | 4 / 0.080 s | 5 / 0.100 s |
| training RGB 0 | 0.000 s | 1 / 0.020 s | 2 / 0.040 s |
| training RGB 5 | 0.100 s | 6 / 0.120 s | 7 / 0.140 s |

Training RGB exists at 10 Hz, anchors 0,5,...795. All 5,120 complete current
observation samples can be used as independent supervised decisions. No images
are repeated/interpolated to pretend this is 25-Hz training capture. The live
policy captures a fresh scene every two ticks (25 Hz). A future denser image
collection would improve coverage, but no missing observations are invented.
The label offset is **1**, derived from recorder semantics; it is not the prior
K=5 offset=2. Windows never cross episodes and incomplete tails are dropped.

`outputs/va_v3/train/alignment.json` records every training/validation decision:
observation index, RGB and proprio label time, actual capture simulation time,
both target indices/label times, native execution log index, native time_ms and
native monotonic time. Native/reference/simulation clocks have distinct origins.
Live `timing.json`, `handoffs.json`, `predictions.npz`, `live_observations.npz`
record actual capture and consumption. Latency uses the shared monotonic clock.

Reset has no history padding. Graphics shaders and inference operator kernels
are initialized before capturing the reset observation. Native initialization
and reset remain the existing SONIC implementation. A short native transport
preload is distinguished from steady-state latency.

The native protocol v4 accepts one 64-D token, but does not schedule by its
frame_index. V3 therefore tracks slot indices and verifies the full token in
native consumption logs, then pairs its hand reference on the same observed
tick. Strict mode aborts on a stale slot or a missing control snapshot. This
is detection of a failed deadline, not a hard real-time guarantee. Diagnostic
`--diagnostic-continue-on-miss` permits collecting a physical failure video;
such a run must be reported as failing the timing contract.

## Training and validation

Seed 38, unchanged episode-level split: 26 train (4,160 samples), 6 validation
(960 samples). AdamW, learning rate 0.0003, weight decay 0.0001, batch 32,
2,000 updates on MUSA. Checkpoints include model, optimizer, configuration,
normalization buffers, split, metrics, joint orders and native bundle hashes.
Best validation checkpoint is selected by body+hand validation loss.

Independent held-out evaluation reproduces token MAE 0.00934346 and hand MAE
0.00340520 rad. Slot token MAE: [0.00939441,0.00929252]. Slot hand MAE:
[0.00355877,0.00325164]. Per-coordinate and slot MSE/MAE are in `offline_val.json`.
Offline performance does not establish grasp success.

## Reproduction

From the host, use the existing container (paths below are container paths):

```bash
docker exec va-train env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -B /workspace/group2/va_training/train_token_v3.py \
  --config /workspace/group2/va_training/configs/token_v3.json \
  --output /workspace/group2/outputs/va_v3_repro/train

docker exec va-train env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -B /workspace/group2/va_training/scripts/evaluate_token_v3.py \
  --checkpoint /workspace/group2/outputs/va_v3/train/best.pt \
  --output /workspace/group2/outputs/va_v3_repro/offline_val.json

docker exec va-train env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  unshare --net bash -c 'ip link set lo up;
  source /workspace/group2/outputs/mujoco_gpu_rendering/enable_mthreads_egl.sh;
  exec python3 -B /workspace/group2/va_training/scripts/run_token_v3_closed_loop.py \
    --checkpoint /workspace/group2/outputs/va_v3/train/best.pt \
    --episode-id xy_random_v1_random_0000 \
    --output /workspace/group2/outputs/va_v3_repro/learner --device musa'
```

Use fresh output directories. CPU and MUSA benchmarks are separate from live
render/transport latency. `benchmark_token_v3.py` accepts checkpoint, output,
device and iteration count. The closed-loop runner writes stage mean, median,
p95 and max, and also separates startup from steady-state inference timings.
The observation-to-command measurement ends at the later of value-matched DDS body-command receipt and paired hand application. Native pre-action token consumption is measured separately. Full 29-D motor targets are matched against the following native previous-action log row; unmatched rows are counted, never guessed.


Runtime rendering/inference now uses an independent spawned process to avoid
Python GIL contention with 200-Hz physics. It receives only the current qpos,
qvel and 96-D state per request; weights and graphics context are persistent,
but no observation/action history or recurrent memory exists. Transport is
polled at up to 1 kHz while physics remains configured at 200 Hz. Process queues
carry exactly one new request every two observed control ticks.

The passive DDS timestamp wrapper forwards the original callback unchanged;
no SONIC source, executable, model or observation-history behavior is changed.
`execution_alignment.json` value-matches native decoder outputs to actual DDS
commands, and `control_timeline.json` records every observed tick including stale
slot consumption in diagnostic mode. Its labels are not decoder inputs.

Run `bash /home/group2/va_training/v3/reproduce.sh` from the host to reproduce
training, offline evaluation, CPU/MUSA benchmarks, strict timing test, a clearly
marked full diagnostic trajectory, fresh native GT/GT expert and 25-FPS physical
video. The video uses overview/head/wrist cameras, the previous 800-tick episode,
identical bootstrap file hashes and grasp criterion, and saved actual qpos/qvel
only (no trajectory playback or physics re-integration). Episode 0000 is a
training episode; this comparison does not establish held-out grasp success.


## Final runtime measurements and limits

The default is the independent process, MUSA inference, 120x160 live render
preserving the original 4:3 camera aspect, then the shared 128x128 preprocessing.
Saved training cameras are 240x320; the lower live raster resolution is an
explicit latency tradeoff, not a change in camera pose/FOV. CLI render dimensions
can retain 240x320, and `--vision-worker thread` reproduces the earlier transport
variant. Neither option changes the learned architecture or inputs.

The latest strict test (`strict_lowlatency_0000`) executes 34 correct slots
before detecting a stale token at tick 34 and aborting. Its measured VA forward
median is 2.22 ms; render median 6.06 ms; capture-to-prediction median 9.66 ms.
These averages do not guarantee the next native deadline. The native consumer
has no frame-index scheduling and the observation is captured partway through
the preceding tick. A fully compliant 800-tick 25/50-Hz rollout has **not** been
achieved. Changing labels to offset=2 or silently holding/relabeling a slot would
hide this failure and is not done in strict mode.

The 16-second video under `outputs/va_v3/diagnostic_videos` uses the actual full
thread-worker 240x320 diagnostic run (`diagnostic_0000`), not the aborted strict
run. It has 800 observed ticks at 50 Hz and 400 predictions, but 399 stale ticks;
its overlay explicitly says TIMING FAILED. No fall, max lift 0.02997 m, no
sustained hold, grasp failure. The unchanged recorded native GT/GT expert from
`outputs/va_v2_video/expert` passes (0.15050 m lift, 4.375 s hold). Decoder/runtime
hashes, bootstrap file hashes, episode and physical success criterion match.
All three MP4 files contain 400 decodable frames at 25 FPS. The independent
process diagnostic (`learner`) aborts after 189 observed ticks due to a missed
control snapshot; its separate `process_videos` clip contains only the actual
prefix, without padding.

`outputs/va_v3/RESULTS.md` contains the full latency table and deliverable index.
Treat this as an implemented and trained baseline with a demonstrated timing
integration failure, not a deployment-ready or successful grasp policy.
