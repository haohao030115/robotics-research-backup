# Startup event timing — preflight blocked

No native benchmark was launched in this task. The six-event capture preflight
failed, so launching another episode would violate the requested execution gate.
The previous 39.2329-second startup does not resolve the event timeline.

## Verified findings

- `canonical_va/native_replay.py` still matches the pinned source-policy SHA-256
  `8ca529d8cbd81f9c67cf2e0ec67ad29deda590c43e2bdb5b3eb1e9dc585b0866`.
- The shared `/workspace/group2/startup_trace/` path is visible in `va-train`.
  A subprocess in `unshare --net` wrote `persistence_check.json`; the host read
  it after process exit. Network namespace isolation does not isolate this mount.
- The old `/tmp/startup_trace/sitecustomize.py` is absent in this container.
- Synthetic capture produced only five named events. Monotonic ordering of
  those five records passed; complete six-event capture failed. These synthetic
  values are not measurements of native execution.
- Inspection also found stale hard-coded probe line numbers and a semantic
  error: `runtime.bridge.low_cmd_received` denotes Python receiving LowCmd,
  **not** native receiving LowState. The rejected probe is disabled (renamed
  `startup_trace/rejected_probe.py`); it must not be used for measurements.

## Required observation locations

| Event | Actual source boundary | Available measurement |
|---|---|---|
| Native launch | `native_replay.py:131`, `subprocess.Popen` | Not measured |
| Native Init Done | native `G1Deploy::InitControl`, C++ line 2814 | Not measured |
| First accepted LowState | native `G1Deploy::LowStateHandler`, C++ line 2690 | Not measured |
| Fifth standby observation | `PlaybackCursor.update`, `run_sonic_grasp.py:102` | Not measured |
| Playback request sent | `native_replay.py:148`, successful `os.write(master,b't')` | Not measured |
| First playing tick observed | `native_replay.py:149-150`, cursor transition | Not measured |

The C++ locations refer to
`canonical_va/native_build/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`.
First accepted LowState necessarily precedes Init Done: `InitControl` returns
false without a buffered LowState. Event numbers in the request are therefore
not chronological order. Intervals must be sorted by measured monotonic time.
Python receipt of a PTY message is also later than the native event that emitted
it; it must be labelled as an observation, not the exact native timestamp.

No event-level wall duration, physics-step delta or simulated-time delta is
available from this attempt. No largest-delay cause is established.

## Fixed waits found by inspection

- Python: nonblocking PTY `select(...,0)`; paced 5 ms physics loop; 180 s
  replay deadline (`native_replay.py:136,142,167`).
- Native initialization: MotionSwitcher `SetTimeout(5.0f)`, a `CheckMode`
  call and a conditional `sleep(5)` during release (C++ 2295-2301).
- Native INIT: 100 ms sleep when LowState is unavailable (C++ 3883-3886).
- InitControl: the existing 3-second control ramp; subsequent five-observation
  standby gate is preserved.

These are possible delay sources, not evidence that any particular wait
dominates. No wait was removed or shortened.

## Concrete instrumentation-only proposal for review

A Python-only trace hook cannot observe an internal callback in the separate
native process. The native binary retains LowStateHandler/InitControl symbols,
but no `perf` or `bpftrace` executable was found on the inspected host/container;
the container's `perf_event_paranoid` is 4. Native uprobes are consequently not
a verified usable route in this environment.

Proposed minimal reviewed update:

1. Add one-shot native `CLOCK_MONOTONIC` timestamps at first accepted LowState
   and at Init Done, retaining the received observation sequence ID. Store in
   a small diagnostics buffer and flush through the existing trace sink;
   preserve callback ordering, arithmetic and thread periods.
2. Add Python timestamps immediately around Popen and after successful playback
   write, and record standby/playing transitions with current physics-step ID
   and MuJoCo simulation time. Capture publication timestamps keyed by the
   existing observation sequence; do not claim an exact simulator step at an
   asynchronous native event unless it is actually observed.
3. Retain this diagnostic sidecar outside the training-retention pruning tree.
   Test all six events, exit/error flushing, shared-path persistence, and actual
   chronological ordering before a real episode.
4. Review the instrumentation diff, then update only the affected pinned source
   and rebuilt-binary identities, preserving the old policy and binary. Do not
   relabel critical sources as non-critical or reuse old QA qualification.

This proposal has **not** been applied. A new native binary/source fingerprint
may require Full QA qualification before Production Basic QA can legitimately
be reused. That conflicts with an immediate Basic-only benchmark under the
unchanged policy and must be resolved explicitly in review.

No performance optimization is justified yet. Further native-stage profiling
is necessary. A candidate to investigate is MotionSwitcher service wait time;
skipping or shortening it would affect startup safety/lifecycle and is not
approved or recommended from the current evidence alone.

Evidence: `startup_trace/preflight_result.json`, `synthetic_events.json`, and
`persistence_check.json`. No historical episode or source-policy file was changed.
