# Canonical VA native startup optimization

## Fast-track result (2026-10-09)

The simulator-only startup optimization is implemented and qualified in
[`startup_benchmark_0001`](../../outputs/canonical_va_startup_optimized_20261009/startup_benchmark_0001/).
Canonical replay now sets `CANONICAL_SIMULATOR_RUNTIME=1`. In that explicit
path, `AudioThread` keeps its DDS client and warning thread but skips the
startup-only `AudioClient::SetVolume(100)` request. Direct hardware launches
without the flag retain the original audio initialization. MotionSwitcher,
DDS command/state transport, SONIC inference, MuJoCo physics, the 50-Hz
controller, and the 3-second ramp are unchanged.

The previously unexplained interval is now accounted for. The constructor
does four Dex3 DDS channel creations, constructs the audio client, and then
calls `SetVolume(100)` after setting a 10-second client timeout. No audio
service exists in the isolated simulator namespace, so that request blocks
before the MotionSwitcher marker. The old hidden interval was 12.011861 s;
the optimized run measures 1.006491 s for the remaining Dex3/audio/client
setup, a reduction of 11.005370 s. The remaining interval is split as:

| Operation | Wall seconds |
|---|---:|
| Dex3 channel initialization | 0.201777 |
| Audio client/thread (volume request skipped) | 0.402065 |
| MotionSwitcher client construction | 0.401982 |
| Other constructor work to `SetTimeout` | 0.000667 |
| **ChannelFactory done → `SetTimeout` start** | **1.006491** |

The MotionSwitcher service check remains intact and still times out after
6.005479 s with `rc=3104` (`UT_ROBOT_ERR_CLIENT_API_TIMEOUT`), an empty mode
name, no `ReleaseMode` call, and no retry sleep. This optimization does not
bypass that service or suppress its failure; the timeout remains required by
the existing hardware lifecycle until a separate service-equivalence review
approves a change.

The retained per-call intervals are `SetTimeout(5.0)` 4.138 µs,
`Init()` 7.509 µs, and `CheckMode()` 6.005440069 s from the `Init` completion
marker to its detail record (6.005479093 s including the surrounding marker
overhead). `ReleaseMode()` was not called.

The measured MUSA stages in this run were Python initialization 0.023490 s,
policy module import 4.191302 s, policy session construction 11.041547 s,
policy I/O metadata 0.000020 s, and complete policy initialization 15.256471 s.
Encoder import was 0.000015 s, session construction 0.450163 s, metadata
0.000066 s, and complete initialization 0.450288 s. These stages were not
optimized; run-to-run variation explains their differences from the prior
diagnostic run.

Before/after performance is:

| Stage | Prior diagnostic | Optimized run | Difference |
|---|---:|---:|---:|
| Startup | 38.916452 s | **28.214649 s** | **-10.701804 s** |
| Total Full-QA wall time | 87.969013 s | **74.958713 s** | **-13.010300 s** |
| Native rollout | 18.504906 s | 18.050284 s | -0.454622 s |

The optimized run passed every Full-QA flag, including encoder/decoder
reconstruction, command trace, state transition, future-target alignment,
schema, dataset, and expert checks. It completed 800 control ticks and 3,200
physics steps with zero missing ticks and zero fall steps. Maximum lift was
0.123048 m with a 4.365 s qualifying hold, and 159 training V4 samples were
retained. Rendering was hardware MTT S4000 EGL (`software=false`). The
qualified runtime hash is
`875b39f0bd75a37609024c89decbe1ce1bee4e59855f9013fbb4f61902e71563`; the
source-policy gate passed with the existing non-critical warnings.

No accepted episode or 50-reference output was modified. The remaining safe
startup opportunities are the MUSA session/cache path and the MotionSwitcher
service timeout; both require independent equivalence/design review before
any further runtime change.

The sections below retain the earlier diagnostic baseline and detailed source
evidence for comparison.

## What the source does

The native constructor performs `ChannelFactory::Init`, creates the
`MotionSwitcherClient`, sets a 5-second timeout, calls `Init`, and then loops
over `CheckMode`/`ReleaseMode` with a literal `sleep(5)` until the mode name is
empty (`g1_deploy_onnx_ref.cpp:2299-2307`). Only after that loop are the
LowState channels created. The same constructor reads the motion CSV and calls
`PolicyEngine::Initialize` (`...cpp:2350-2356`). That routine converts/loads the
policy ONNX model into the TensorRT/MUSA inference engine and creates its input
and output buffers (`control_policy.hpp:59-198`). Encoder graph setup then
takes place before the 50-Hz control thread is started (`...cpp:2419-2422,
2645-2653`).

`InitControl()` itself increments `time_` by `control_dt_ = 0.02` and compares
it with `duration_ = 3.0` (`...cpp:2801-2835`). The control thread is created
with a 20-ms period (`...cpp:2645`). Therefore the ramp is 150 valid calls,
about 3.0 seconds; it is not an 18.5-second ramp.

## Measured native timeline

The sidecar is
[`startup_event_timeline.json`](../../outputs/canonical_va_startup_instrumented_20261009/startup_event_timeline.json),
and the raw native records are
[`native_startup_events.jsonl`](../../outputs/canonical_va_startup_instrumented_20261009/native_startup_events.jsonl).
Native event stamps use `steady_clock`; physics counts and MuJoCo time are the
nearest Python observation when the event was discovered.

| Interval / event | Wall seconds | Observed physics steps | Simulated seconds |
|---|---:|---:|---:|
| Process launch → constructor entry | 0.004075 | 1 | 0.005 |
| Constructor entry → ChannelFactory done | 0.001771 | 1 | 0.005 |
| **ChannelFactory done → MotionSwitcher release done** | **18.017514** | **3,603** | **18.015** |
| MotionSwitcher release → LowState channels ready | 0.101112 | 20 | 0.100 |
| LowState channels → first native LowState callback | 0.001627 | 1 | 0.005 |
| First LowState → motion data loaded | 0.008246 | 1 | 0.005 |
| Motion data → policy initialization complete | **14.974469** | **2,995** | **14.975** |
| Policy graph → encoder graph complete | 0.437984 | 88 | 0.440 |
| Encoder graph → first valid `InitControl` | 0.004151 | 1 | 0.005 |
| First valid call → call 50 | 0.979949 | 196 | 0.980 |
| Call 50 → call 100 | 1.000001 | 200 | 1.000 |
| Call 100 → call 150 | 1.000000 | 200 | 1.000 |
| Call 150 → `Init Done` | 0.000010 | 0 | 0.000 |
| `Init Done` → five standby observations | 0.129259 | 25 | 0.125 |
| Standby → playback request | 0.000010 | 0 | 0.000 |
| Playback request → first playing tick | 0.015446 | 3 | 0.015 |

The latest native event marker to first playing tick is 36.676267 seconds. The
pipeline's `timings_seconds.startup` is 38.916452 seconds because it also
includes Python/MuJoCo replay setup before the native-process marker. The
measured native bottlenecks are unambiguous within this run:

1. **18.017514 s in the post-ChannelFactory constructor phase, dominated by
   MotionSwitcher client/check timeout.** The source contains the 5-second retry sleep and
   the client has a 5-second timeout. The event is after the loop, so this
   measurement covers `msc_->Init`, mode checks, release calls, retry sleeps,
   and the small constructor work immediately before that block; it is not a
   DDS LowState wait. A further split would require another run with events
   immediately around `msc_->Init`, each `CheckMode`, and each `ReleaseMode`
   call.
2. **14.974469 s in `PolicyEngine::Initialize`.** This is the ONNX-to-TRT
   conversion/cache load, TensorRT engine initialization, input/output setup,
   and MUSA device/session construction. Encoder graph setup adds 0.437984 s.
3. **2.979949 s for the actual 150-call initialization ramp.** The three
   call-count intervals are approximately 0.98, 1.00, and 1.00 seconds at the
   configured 50 Hz. Thus the observed first-LowState → `Init Done` interval of
   about 18.42 s is approximately 14.97 s policy construction + 0.44 s encoder
   setup + 0.01 s handoff + 2.98 s ramp. It does not show an 18.5-second
   controller ramp.

`ChannelFactory::Init` completed in 1.771 ms after constructor entry. The
LowState subscriber was registered by `InitChannel` at the
`lowstate_channel_init_done` marker, and the first callback arrived 1.203 ms
later. This rules out a long Python-side callback-registration wait; the long
first interval occurs before channel registration, in the MotionSwitcher/
constructor phase.

## Completed substage profiling

The latest raw records are
[`native_startup_events.jsonl`](../../outputs/canonical_va_startup_instrumented_20261009/native_startup_events.jsonl)
and the materialized timeline is
[`startup_event_timeline.json`](../../outputs/canonical_va_startup_instrumented_20261009/startup_event_timeline.json).
The detail parser retained the tab-delimited MotionSwitcher records; all
steady-clock intervals below are calculated from the raw native timestamps.

### MotionSwitcher

| Operation | Measured wall time | Result |
|---|---:|---|
| ChannelFactory done → timeout setup start | 12.011861 s | Enclosing constructor/planner interval; the client constructor itself has no internal marker |
| `SetTimeout(5.0)` | 23.748 µs | Completed |
| `Init()` | 6.695 µs | Completed |
| `CheckMode` call → detail marker | 6.005593 s | `rc=3104` (`UT_ROBOT_ERR_CLIENT_API_TIMEOUT`), empty form/name |
| `ReleaseMode` calls | 0 | No mode name was returned, so no release was attempted |
| Retry sleeps | 0 | One check attempt; no 5-second sleep |
| ChannelFactory done → release block done | **18.017514 s** | One attempt, no active MotionSwitcher mode |

The isolated DDS namespace had no responding MotionSwitcher service. The
single `CheckMode` timed out after the client timeout path and returned the
SDK's documented API-timeout code. This explains the service behavior, but it
does not authorize bypassing the client or changing its lifecycle.
The marker is emitted immediately after the call, so the interval includes
the small return-detail formatting cost; no pre-call marker was added.

### MUSA PolicyEngine and encoder

| Substage | Policy | Encoder |
|---|---:|---:|
| Python interpreter initialization (`Py_Initialize`, once) | 23.099021 ms | Reused; wait interval 3.558 µs |
| `musa_onnx` module import | **4.115117013 s** | 18.522 µs |
| `Session` construction | **10.836113416 s** | **0.437661832 s** |
| I/O metadata and buffers | 22.148 µs | 62.381 µs |
| Complete `Initialize` interval | **14.974461215 s** | **0.437789639 s** |
| Native graph-capture milestone | 14.974474758 s policy stage overall | 0.437984312 s to encoder milestone |

`musa_onnx.Session` source shows that session construction includes ONNX load
and validation, MUSA tensor/device allocation, graph construction, Torch JIT
tracing, validation passes and warm-up runs, followed by synchronization. The
run did not have separate `torch`/`torch_musa` import markers, so those costs
are included in the module-import interval; the ONNX/JIT/MUSA breakdown is
source-based inference rather than separately measured timing.

The largest remaining measured costs are the 6.006-second MotionSwitcher API
timeout and the roughly 11-second MUSA policy session/import path. The former
checks a required service and the latter builds and validates the inference
graph. The simulator-only audio volume guard described at the top of this
report is the only implemented optimization; a supported MUSA cache or
service-aware MotionSwitcher timeout would need a separate equivalence review.

## Historical diagnostic verification

The latest benchmark used the unchanged physical/control semantics, native SONIC +
DDS + MuJoCo, MTT S4000 EGL, training retention, and the existing Full QA
decision path. Its validation summary is
[`validation_summary.json`](../../outputs/canonical_va_startup_instrumented_20261009/startup_benchmark_0002/validation_summary.json).

| Check | Result |
|---|---|
| Full QA | pass; reconstruction, command trace, state transition, future-target and schema checks all true |
| Runtime | complete; 800 control ticks, 3,200 physics steps, 0 missing ticks |
| Physical grasp | pass; max lift 0.124985 m, hold 4.370 s, 0 fall steps |
| Renderer | `Moore Threads` / `MTT S4000`, EGL, `software=false` |
| Training retention | pass; 159 retained canonical V4 samples |
| V4 retention | 159-sample training export passed; standalone builder was not rerun while the separate training job was active |
| Regression suite | native unittest subset: 5 passed |

The qualified run records canonical runtime hash
`40e1b8a5e603f9993b122fc209697a4f268fb6fa038ae44713706c93de6c6cf5`; the
source-policy check passed with only the pre-existing non-critical warnings.

The latest benchmark timings were startup 38.916452 s, native rollout
18.504906 s, trace materialization 9.698289 s, RGB rendering 1.169245 s,
sample compilation 0.668619 s, Full QA validation 17.276243 s, disk writing
1.408296 s, and total wall time 87.969013 s. This is a diagnostic replication,
not a speedup claim; the prior accepted run remains the comparison baseline.
The separate V4 training job observed after this run was left untouched; no
training code, dataset, or checkpoint was modified.

For comparison, the fresh qualified run versus the preceding accepted
diagnostic run is:

| Stage | Prior accepted diagnostic | Fresh profiling run | Difference |
|---|---:|---:|---:|
| Startup | 39.313140 s | 38.916452 s | -0.396688 s |
| Total wall clock | 88.909183 s | 87.969013 s | -0.940170 s |
| MotionSwitcher block | 18.017849 s | 18.017514 s | -0.000335 s |
| Policy initialization | 15.082753 s | 14.974469 s | -0.108284 s |

These differences are run-to-run variation; no optimization was applied.

For the older comparable Full-QA reference, the preceding instrumented run in
`outputs/canonical_va_startup_optimization_20261009/startup_benchmark_0001`
had these timings:

| Stage | Previous Full-QA run | This diagnostic run | Difference |
|---|---:|---:|---:|
| Startup | 39.300618 s | 39.313140 s | +0.012522 s |
| Native rollout | 18.438108 s | 18.538947 s | +0.100839 s |
| Trace materialization | 9.798896 s | 9.692725 s | -0.106171 s |
| RGB rendering | 1.180805 s | 1.187696 s | +0.006891 s |
| Sample compilation | 0.691285 s | 0.679530 s | -0.011755 s |
| Full-QA validation | 17.548930 s | 17.690528 s | +0.141598 s |
| Disk writing | 1.438746 s | 1.455198 s | +0.016452 s |
| **Total** | **88.745541 s** | **88.909183 s** | **+0.163642 s** |

This is run-to-run variation; the new run is a diagnostic replication, not a
speedup claim.

## Changes and optimization decision

The optimized tree includes the earlier instrumentation/detail retention plus
the simulator-only audio guard:

- `canonical_va/native_build/src/g1/g1_deploy_onnx_ref/include/canonical_trace.hpp`
  — steady-clock event writer;
- `canonical_va/native_build/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`
  — constructor, model/graph, thread, LowState and InitControl milestone
  events;
- `canonical_va/native_replay.py` — accepts legacy and tab-delimited native
  records, preserves detail records in the persistent timeline, and sets the
  explicit simulator-runtime flag;
- `canonical_va/native_build/src/audio_thread/audio_thread.hpp` and
  `audio_thread.cpp` — make the default-volume request optional;
- `canonical_va/source_policy.json`,
  `canonical_va/native_build/build_validation.json`, and the rebuilt native
  executable — updated through the normal fingerprint process.

No sleep, controller period, 3-second ramp, DDS frequency, MuJoCo timestep,
initial condition, SONIC computation, indexing, or physical gate was changed.
The next safe step is a design review for a validated MUSA graph cache. The
MotionSwitcher timeout should remain unchanged until service behavior is
established on hardware.
