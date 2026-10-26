# Canonical VA Startup Optimization Handoff

Date: 2026-10-09

## 1. Objective and current status

The objective was to reduce canonical episode startup time while preserving
SONIC, DDS, MuJoCo, controller timing, canonical indexing, physical gates, and
training retention. A simulator-only audio initialization guard is now
implemented, rebuilt, source-policy qualified, and exercised in one fresh
Full-QA run. Hardware launches retain the original audio path; MotionSwitcher
and all controller/physics semantics are unchanged.

## 2. Confirmed performance measurements

The latest qualified Full-QA benchmark is
`outputs/canonical_va_startup_optimized_20261009/startup_benchmark_0001/`.

| Stage | Seconds |
|---|---:|
| ChannelFactory done → `SetTimeout` start | 1.006491 |
| MotionSwitcher check timeout | 6.005479 |
| PolicyEngine initialization | 15.256471 |
| Encoder graph initialization | 0.450288 |
| InitControl ramp (150 valid calls at 50 Hz) | 2.980000 |
| Pipeline startup total | 28.214649 |
| Total wall clock (Full QA) | 74.958713 |

The optimized run completed 800/800 control ticks and 3,200 rollout physics
steps, with 0 missing ticks and 0 fall steps. It produced 159 training V4
samples. Maximum lift was 0.123048 m and qualifying hold was 4.365 s.

## 3. Root causes and source locations

* `canonical_va/native_build/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`
  constructs Dex3 DDS channels, creates `AudioThread`, then constructs
  `MotionSwitcherClient` (around lines 2239–2313). The old trace's 12.011861 s
  pre-timeout interval was caused by the audio client calling
  `SetVolume(100)` after `SetTimeout(10.0f)` when no audio service exists. The
  optimized simulator path skips only that startup-only request; the remaining
  Dex3/audio/MotionSwitcher setup is 1.006491 s. The MotionSwitcher check still
  returns API timeout 3104 after 6.005479 s, with an empty mode name, one
  attempt, no `ReleaseMode`, and no sleep. The simulator bridge publishes
  LowState/LowCmd but no MotionSwitcher server responds in the isolated DDS
  namespace, so bypassing this client remains a service-semantics change.
* The same C++ constructor calls `PolicyEngine::Initialize` around lines
  2365–2371. With the current MUSA build, `musa_inference.hpp` creates a
  Python `musa_onnx.Session`; the optimized trace measured 23.490357 ms for
  `Py_Initialize`, 4.191302246 s for module import, 11.041546584 s for policy
  session construction, and 20.063 us for I/O metadata, totaling 15.256471 s.
  Encoder session construction took 0.450162906 s.

The retained MotionSwitcher call intervals are `SetTimeout(5.0)` 4.138 µs,
`Init()` 7.509 µs, and `CheckMode()` 6.005440069 s from the `Init` completion
marker to its detail record (6.005479093 s including marker overhead).
`ReleaseMode()` was not called.
* `InitControl()` (around lines 2816–2848) increments `time_` by 0.02 s until
  the configured 3.0 s duration. The measured three one-second intervals are
  the expected 50 Hz ramp; do not shorten it.

The measurements above are verified in the retained optimized raw JSONL and
timeline. The simulator-only audio guard is the only implemented optimization;
the MotionSwitcher timeout and MUSA session lifecycle remain unchanged.

## 4. Changes already made

The earlier source-policy-qualified instrumentation added steady-clock startup
markers, Python-side timeline materialization, and milestone markers for
constructor/channel setup, policy/encoder setup, LowState, threads, and
InitControl. That version produced the accepted Full-QA benchmark.

The current working tree contains qualified detail markers for
MotionSwitcher calls/sleeps and MUSA Python/import/session/metadata stages in:

* `canonical_va/native_build/src/g1/g1_deploy_onnx_ref/include/canonical_trace.hpp`
* `canonical_va/native_build/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`
* `canonical_va/native_build/src/g1/g1_deploy_onnx_ref/include/musa_inference.hpp`

`canonical_va/native_replay.py` now accepts both two-field events and
tab-delimited detail records, preserving detail records as
`native_detail_records` in the startup timeline. The optimized executable was rebuilt in `va-train` on 2026-10-09 and is recorded in
`build_validation.json` with SHA-256
`229772cd6eea5db0654db32bce36da754674b5166ada8cdf699c2fc1ef2adf90`.
The parser, native markers, and simulator-only audio guard were exercised by
the latest benchmark.

## 5. Current Git/runtime/source-policy state

The repository metadata at `/home/group2/.git` is an empty directory in this
session. `git status`, branch, commit, and diff therefore cannot be obtained
reliably; nothing was committed, reset, discarded, or overwritten by this
handoff.

The native executable is
`/home/group2/canonical_va/native_build/target/release/g1_deploy_onnx_ref`
(container path `/workspace/group2/canonical_va/native_build/target/release/g1_deploy_onnx_ref`).
`canonical_va/native_build/build_validation.json` records the qualified binary
(sha256 prefix `229772cd`).

The baseline policy is `canonical_va/source_policy.json`; its review text now
records the explicitly reviewed 2026-10-09 simulator-only audio guard and
instrumentation qualification.
The qualified runtime hash is
`875b39f0bd75a37609024c89decbe1ce1bee4e59855f9013fbb4f61902e71563`.

## 6. Completed experiments and QA evidence

* `outputs/canonical_va_startup_optimization_20261009/startup_benchmark_0001/`
  is the preceding comparable Full-QA run (total 88.745541 s).
* `outputs/canonical_va_startup_bottleneck_20261009/startup_benchmark_0001/`
  is the latest accepted diagnostic run (total 88.909183 s).
* `outputs/canonical_va_startup_instrumented_20261009/startup_benchmark_0002/`
  is the fresh qualified profiling run (total 87.969013 s).
* `outputs/canonical_va_startup_optimized_20261009/startup_benchmark_0001/`
  is the optimized Full-QA run (startup 28.214649 s; total 74.958713 s).
* Both runs passed reconstruction, command trace, state transition,
  future-target alignment, schema, physical validation, and renderer checks.
  The renderer was hardware EGL `Moore Threads / MTT S4000`, software=false.
* The optimized run retained 159 training V4 samples and passed the existing
  export and validation gates.

No experiment modified the 44 accepted episodes or the separate training
dataset/checkpoint.

## 7. Active processes and concurrency restrictions

At the optimized benchmark launch there was no competing Native controller,
SONIC/DDS/MuJoCo, compiler, CMake, Ninja, Cargo, or training process. The run
completed without interfering with dataset production or training. Do not
launch another Native run without a fresh contention check.

## 8. Outstanding issues

1. The MotionSwitcher service is absent in the isolated DDS namespace; its
   single `CheckMode` call returns API timeout 3104 after 6.005479 s. No
   `ReleaseMode` call or retry sleep occurs.
2. MUSA policy session construction is the largest measured remaining
   substage at 11.041546584 s; no supported cache path has been established.
3. The simulator-only audio guard is qualified. Direct hardware launches do
   not set `CANONICAL_SIMULATOR_RUNTIME` and retain the original volume request.
4. The va-train Python regression subset passes (9 tests); the fixed CMake
   test directory has no registered tests, and the optimized Full-QA run
   passes all physical and temporal gates.
5. Git branch/commit/diff inspection remains blocked by the empty `.git`
   metadata; it was not required for build or qualification.

## 9. Recommended next action

Use the retained raw JSONL and timeline to review a validated MUSA graph-cache
design. Do not change controller lifecycle, bypass the MotionSwitcher service,
alter the 3 s ramp, change cadence or initial conditions, or launch another
Native benchmark automatically. Any such candidate requires explicit
equivalence review and a new qualification plan.

## 10. Exact paths and continuation commands

```text
cd /home/group2
cat docs/grasp_pipeline/STARTUP_OPTIMIZATION_REPORT.md
cat outputs/canonical_va_startup_optimized_20261009/startup_event_timeline.json
cat outputs/canonical_va_startup_optimized_20261009/native_startup_events.jsonl
cat outputs/canonical_va_startup_optimized_20261009/startup_benchmark_0001/validation_summary.json
cat canonical_va/source_policy.json
cat canonical_va/native_build/build_validation.json
```

After source review, the previously used build/qualification entry point was:

```text
docker exec --user 0 va-train bash -lc \
  'cd /workspace/group2/canonical_va/native_build && cmake --build build -j4'
```

The one optimized benchmark is complete. Do not launch another Native run
automatically, launch a batch, reuse a training output, or alter the original
accepted artifacts.
