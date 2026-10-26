# Native execution-path audit (2026-10-08)

## Source of truth

The successful source is `kimodo-t6/g1-integration/task8_expert_reference_batch_v1_20261007/episodes/kimodo_episode_0001`. Its validation records 0.12304299381683059 m maximum lift, 4.32999999999993 s continuous hold, physical success and expert validity. Its provenance points to `trials/seed_100/qualification/replay`.

The original executable hash is `a4c1a00c6794184cb062d150a294d36047c748dc9590474146a68c2b7368102e`. The current shared executable matches it. The successful producer's frozen `run_sonic_grasp.py`, `generate_freebase_expert.py`, `scene.py`, `hand_trajectory_controller.py`, `expert_trajectory.py` and tabletop XML are byte-identical to the corresponding current `dl-group2` files. No original controller, scene, trajectory, or model file is edited by this repair.

## Execution chain and code locations

Paths below are relative to `/workspace/group2`. Native line numbers refer to the unchanged original source, not the instrumented copy.

| Stage | Existing implementation reused | Semantics |
|---|---|---|
| Simulator reset | `dl-group2/scripts/run_sonic_grasp.py:154` (`TabletopDDS.__init__`), `cases.py:39` (`reset_case`) | Free base, case cube XY/yaw; reference frame-0 robot pose; zero initial velocity |
| Startup body | `dl-group2/scripts/generate_freebase_expert.py:21` (`WholeBody`), called from `TabletopDDS.step` | Original startup balance WBC while native controller initializes; no trajectory tuning or active replay IK |
| Startup transition | Native `InitControl`, `Control` state machine; original `PlaybackCursor` | Wait for `Init Done`, send `]`, observe five standby ticks, send `t`; wall-paced 200 Hz physics continues during startup |
| Observation gathering | `GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:2856` (`GatherRobotStateToLogger`) | Read actual DDS LowState selected by native DataBuffer; native float32 sensor representation and joint mapping |
| History | Same C++ file `GatherHis*` around lines 1460–1650; `src/state_logger.cpp:199` (`GetLatest`) | Original ring buffer, stride/time history and zero-padding; no canonical_va history implementation in control path |
| Reference heading/update | Same C++ file `UpdateHeadingState:569`, `ComputeApplyDeltaHeading:603`, `CurrentFrameAdvancement` | Capture real initial heading; advance original reference cursor at native control cadence |
| Encoder | Same C++ file `GatherEncoderObservations`, `GatherTokenState` around 1668 | Original mode-0 filtering and input gathering, then original EncoderEngine.Encode |
| Actual inference backend | Original `include/musa_inference.hpp`, original `scripts/musa_onnx.py` | Existing native SONIC build uses Torch/MUSA FP32 execution of the frozen release ONNX models, not the new CPU control loop |
| Decoder and current action | Same C++ file `CreatePolicyCommand:3151` | Original exact decoder buffer; current raw action obtained immediately after Infer; previous_action copied before overwrite |
| Scaling/mapping | Same C++ function and `include/policy_parameters.hpp` | Original IsaacLab-to-motor permutation, action scale, defaults, kp/kd and motor gain scale |
| DDS command publication | Same C++ file `LowCommandWriter:2708` | Original independent 500 Hz writer; repeatedly publishes the latest command, no forced synchronization with simulator |
| Body application | `dl-group2/scripts/run_sonic_grasp.py:219` (`TabletopDDS.step`) | Original selected DDS command under original mutex, original PD and joint torque clipping |
| Hand application | Same step, plus `hand_trajectory_controller.py:44` (`target`) and `:50` (`apply`) | Original independent hand PD at each physical step; 50 Hz playback cursor drives interpolation time |
| Physics | Same original step | 200 Hz MuJoCo, original mj_step then mj_forward; no forces, welds, attachment, or cube repositioning during replay |

## Reference consumption

All arrays are copied unchanged into `source_reference`. Native CSV serialization is lossless `%.17g`, checked with exact equality to input arrays. There is no producer-specific execution branch.

| Canonical field | Actual consumer | Limits / provenance |
|---|---|---|
| `body_ref_q` | `joint_pos.csv` → native MotionSequence.JointPositions → `GatherMotionJointPositionsMultiFrame:750` → mode-0 encoder input | Original 10 future frames at step 5; standby holds current frame; frame 0 also initializes robot body |
| `body_ref_dq` | `joint_vel.csv` → native MotionSequence.JointVelocities → `GatherMotionJointVelocitiesMultiFrame:829` → mode-0 encoder input | Supplied values, no derivative regeneration |
| `root_ref_pos` | `TabletopDDS.__init__` uses frame 0; all frames serialized as `body_pos.csv` | **Later root position frames are not consumed by mode-0 encoder/decoder**: root-z observation fields are disabled by mode-0 filtering. Later values are retained as reference/provenance, not claimed as body control input |
| `root_ref_quat` | Frame 0 initializes root; `body_quat.csv` → `GatherMotionAnchorOrientationMutiFrame:626`, original heading correction and measured base quaternion | Original mode-0 relative orientation features; no canonical_va orientation formula in production |
| `hand_ref_q` | Original `HandTrajectoryController.target/apply`, called from original `TabletopDDS.step` using the original PlaybackCursor reference time | Named canonical 14D absolute radians; np.interp, kp=12, kd=0.3, original clipping. No extra scale/filter. Native Dex3 DDS hand commands are not the simulator's independent hand reference source in this existing replay path |

Unused root velocity serialization is compatibility data for the native reader; it is not mode-0 supervision or a new controller input. Zero hand_ref_dq does not alter the canonical hand PD because that PD does not consume desired velocity.

## Logging changes and isolation

`native_build.py` copies the original deployment source into `canonical_va/native_build`. Only trace writes and identity metadata are added. `instrumentation.patch` gives the exact diff. The original functions, math, state transitions, update order, thread periods, and engine remain intact. A regression test checks that no existing source line is removed without retaining the same line.

- LowState.reserve[0] carries a unique sequence assigned immediately before the original publisher writes the physical snapshot. The original LowState.tick is unchanged.
- Native trace captures the actual selected LowState sequence, same-decision encoder input/output, decoder input/output, prior action, current reference frame/play flag, and all mapped motor command fields.
- A bookkeeping member travels with the immutable MotorCommand buffer. LowCmd.reserve[0] carries its decision identity; reserve[1] carries the publisher sequence. Motor fields are unchanged and CRC is recomputed by the original code.
- The original Python DDS callback and locking still receive messages. A property hook records the actual selected message when original step reads it under its mutex. No second subscriber or timestamp matching guesses which command was applied.
- All physics steps retain exact integration state, applied controls, consumed publication identity and the **independent actual hand reference index**. Command durations are variable, and commands with zero consumption are recorded; an unconsumed active command fails dataset validation.
- RGB is rendered offline from exact native-consumed observation snapshots, avoiding live rendering delays. A second reconstruction compares pixels exactly.

Instrumentation incurs trace-copy/I/O overhead. No deterministic equality of wall-clock scheduling is claimed. Network namespace isolation keeps the original domain 0 and loopback transport while avoiding unrelated concurrent controllers; it is recorded in runtime metadata. The first requested output (`native_retry1`) stopped before physics because a concurrent native replay occupied shared DDS. Its report is retained unchanged. The actual replay uses a fresh `native_retry2` directory in an isolated network namespace.

## Previous synchronous implementation

The old replay and forensic loops are archived as non-executable `.py.txt` files in `diagnostics/`. `replay.py` dispatches only to native replay, and `forensic.py` dispatches only to native trace validation. CLI accepts only `--backend native` and has no fallback.

`runtime.py` snapshot/camera methods and `policy.py` frozen model/parameter readers remain available for diagnostics. The old Runtime control methods and Policy observation builder are **not called by production**. Native control histories, orientation features, scaling, PD, hand timing, startup and inference are all original. Independent verification uses saved inputs and frozen CPU ONNX inference; physics reconstruction calls original TabletopDDS.step with the actual saved DDS motor fields.

## Reproduction

Build once (the isolated build directory must be fresh):

```bash
cd /workspace/group2
python3 -B -m canonical_va.native_build
```

In the existing privileged `va-train` container, use an isolated network namespace and a fresh output:

```bash
docker exec va-train unshare --net bash -lc '
  ip link set lo up && cd /workspace/group2 &&
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  python3 -B build_canonical_va_episode.py \
    --reference /workspace/group2/kimodo-t6/g1-integration/task8_expert_reference_batch_v1_20261007/episodes/kimodo_episode_0001 \
    --output /workspace/group2/outputs/canonical_va/kimodo_episode_0001_native_retry2 \
    --backend native --gpu 0
'
```

The output is never overwritten. Use a new output name to reproduce after this run exists. Reference content is unchanged. See that episode's REPORT.md and validation JSON for actual test outcomes; source reuse alone does not establish physical or dataset validity.

## Standby verifier correction and immutable revalidation

The initial native replay completed successfully. Its initial verifier rejected a standby reference-velocity comparison. Native `GatherMotionJointVelocitiesMultiFrame` (lines 855–889) explicitly sets all reference joint velocities to zero when not playing. The incorrect check expected the nonzero source frame-0 velocity (maximum mismatch 0.003568207). The first such mismatch was decision 1, standby, frame 0; it was a verifier expectation error, not a token/action shift or controller divergence.

The check now follows that explicit native behavior: standby velocity is zero, active velocity is the original supplied future reference. A regression test covers nonzero source velocity in both states. No controller code, trace, source reference, physical criterion or replay result was changed. Recording-time implementation files were archived and matched against the originally recorded SHA256 before editing the verifier; new runs archive them before launch.

The same trace is revalidated and compiled into a separate output using the actual production entry, without any new physical steps:

```bash
cd /workspace/group2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
python3 -B build_canonical_va_episode.py \
  --reference /workspace/group2/kimodo-t6/g1-integration/task8_expert_reference_batch_v1_20261007/episodes/kimodo_episode_0001 \
  --output /workspace/group2/outputs/canonical_va/kimodo_episode_0001_native_verified1 \
  --recorded-episode /workspace/group2/outputs/canonical_va/kimodo_episode_0001_native_retry2
```

`validation/recorded_replay_reuse.json` hashes all reused evidence and records `new_physics_steps: 0`; `verification_provenance.json` identifies the updated verifier. Original retry1/retry2 reports are retained. The final acceptance result is in `native_verified1/REPORT.md`.

## Effective body control during startup

The original simulator reads a DDS message during standby but then calls `startup_wbc.apply()`, which replaces body torque. Final canonical physics now distinguishes `selected_dds_command_id` from the effective `command_id` and labels every step's `body_control_source` as `startup_WBC` or `native_DDS`. Only effective native body consumption enters each command's consuming-step list. Raw received/selected message evidence remains unchanged. Original hand PD still runs after WBC.

Final materialization and full verification use the same raw trace in the new directory:

```bash
cd /workspace/group2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
python3 -B build_canonical_va_episode.py \
  --reference /workspace/group2/kimodo-t6/g1-integration/task8_expert_reference_batch_v1_20261007/episodes/kimodo_episode_0001 \
  --output /workspace/group2/outputs/canonical_va/kimodo_episode_0001_native_final \
  --recorded-episode /workspace/group2/outputs/canonical_va/kimodo_episode_0001_native_retry2
```

Use `native_final/REPORT.md` and its validation JSON as the final acceptance artifacts. `native_verified1` and all earlier outputs remain preserved. Across these commands only `native_retry2` performed a physical replay; all subsequent verification/materialization uses immutable recorded evidence and zero new physics steps.
