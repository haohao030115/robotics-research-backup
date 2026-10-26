# Canonical VA production performance audit

2026-10-09. This audit uses the saved accepted `reference_episode_008` from the
50-reference production batch. No native run was started during this audit and
no active batch was interrupted. The batch summary was already terminal
(44 PASS, 6 FAIL).

## Measured baseline

Source: `outputs/grasp_pipeline_phase_f_even50/production_training_attempt_0002/reference_episode_008/validation_summary.json`.
The episode passed production basic QA, retained 159 V4 samples, and used the
MTT S4000 EGL renderer. The measured stages are:

| Stage | Seconds |
|---|---:|
| Startup | 43.0248 |
| Native rollout | 18.7125 |
| Trace materialization | 9.9309 |
| RGB rendering | 1.1813 |
| Sample compilation | 0.6890 |
| Validation / QA | 2.4891 |
| Disk writing | 1.4279 |
| **Total wall clock** | **77.7173** |

The listed components sum to 77.4553 seconds; 0.2620 seconds is orchestration
and final bookkeeping overhead. `startup_simulation_seconds=40.785` is simulated
time, not a wall-clock measurement.

## Execution path and startup root cause

`canonical_va.pipeline.run()` validates and probes the renderer, calls
`native_replay.replay()`, materializes the trace, runs `basic_validate()`,
compiles and packs samples, then publishes the retention profile. The replay
process starts the unchanged native executable and the unchanged
`TabletopDDS.step` path.

In `canonical_va/native_replay.py:142-167`, the Python loop advances one
MuJoCo step every 5 ms even while native playback has not started. It waits for
`Init Done`, a received LowState, five standby observations, and the keyboard
play request. Those waiting iterations are recorded as `playing=false`, which
explains the 8,157 startup physics steps and the 40.785 seconds of simulated
time.

The native controller's `InitControl()` is a 50 Hz linear ramp with
`duration_=3.0` seconds (`g1_deploy_onnx_ref.cpp:191,2194-2199,2776-2804`).
The remaining startup wall time includes native process/model/DDS initialization
and the time until the Python side observes the readiness and playback
conditions. It is not evidence that 40.785 seconds of physical stabilization is
required.

`native_truth.materialize()` then restores MuJoCo integration states and builds
ID-linked observation, reference, encoder, decoder, post-state and physics
tables (`native_truth.py:31-93`). This is required for the current canonical
retention contract. It also performs exact RGB snapshot rendering and writes
the derived tables. The production retention step subsequently stages and
copies the selected training files before pruning forensic files.

## Optimization ranking

| Proposal | Stage | Expected saving | Risk | Required validation |
|---|---|---:|---|---|
| Avoid or accelerate pre-playback MuJoCo stepping | Startup | Potentially large (unmeasured) | **High**: changes DDS readiness, initial state and timestamp semantics | Native scheduler, startup-state, physical and exact timing equivalence |
| Reuse a persistent native worker | Startup | Potentially large | **High**: reset/process isolation and DDS lifecycle | Independent-episode reset proof and native race tests |
| Reduce `materialize()` MuJoCo restores / derive tables directly | Trace materialization | Potentially several seconds | **Medium/High**: changes snapshot/contact/hash construction | Byte/structure invariants and full QA comparison |
| Avoid the second training-retention staging copy | Disk writing | Small to moderate | **Medium**: atomic publication and failure recovery | Fresh-output, interruption and V4 builder tests |
| Change native cadence or remove the 5 ms pacing | Rollout | Potentially large | **High**: token handoff race and physics semantics | Strict token-consumption and physical equivalence tests |

No low-risk implementation-only change was identified. In particular, removing
the startup stepping or changing pacing would violate the established timing
and reset contract without a separately approved native-runtime experiment.

## Implemented changes

None. Compiler, native controller, DDS scheduling, MuJoCo configuration,
retention contracts and source hashes were left unchanged. This is deliberate:
the available opportunities all affect physical timing or canonical snapshot
construction and therefore require a controlled benchmark before implementation.

## Before / after benchmark

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Startup | 43.0248 s | Not run | — |
| Native rollout | 18.7125 s | Not run | — |
| Trace materialization | 9.9309 s | Not run | — |
| RGB rendering | 1.1813 s | Not run | — |
| Sample compilation | 0.6890 s | Not run | — |
| Validation | 2.4891 s | Not run | — |
| Total | 77.7173 s | Not run | — |

The baseline artifact remains accepted (`expert_valid=true`), has 159 samples,
hardware EGL identity `Moore Threads / MTT S4000`, and is V4-builder compatible.
No unverified speedup is claimed.

## Further work requiring approval

The first useful experiment is startup substage instrumentation around native
process launch, `Init Done`, LowState readiness, standby completion and playback
request. A second experiment can profile the restore/snapshot loop in
`native_truth.py`. Any resulting optimization must use a fresh output and prove
physical success, complete tick coverage, renderer identity, source hashes and
V4 compatibility against the saved baseline.
