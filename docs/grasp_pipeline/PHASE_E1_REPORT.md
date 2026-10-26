# Phase E.1 — Native token consumption mismatch investigation (2026-10-09)

This investigation was forensic only. No retraining, checkpoint modification, native rerun, or DDS/runtime change was made.

## Exact invariant and source

The check is in [`va_training/v4/rollout.py`](/home/group2/va_training/v4/rollout.py:167):

```python
if pending is None or pending['intended_tick'] != actual \\
   or not np.allclose(tr[5:], pending['token'], rtol=0, atol=1e-6):
    scheduler.counts['deadline_misses'] += 1
    ...
    raise RuntimeError(f'Native token consumption mismatch at tick {actual}; strict abort')
```

`actual` comes from the native `policy_logs/token_state.csv` row; `pending` is the token that the Python loop published for that absolute tick. The invariant requires the native post-control token to equal the exact token published for the same tick within `1e-6`, with contiguous tick IDs.

## Tick-289 evidence

At tick 289 the scheduler recorded generation **57**, prediction slot **4**, intended tick **289**. The expected token begins `0.0063246898`. The native `token_state.csv` row 289 begins `0.001934925`; its maximum absolute difference from the expected slot-4 token is **0.0084344145**. The same actual row differs from the preceding tick-288 token by only **4.85e-10**. Therefore the native controller consumed the previous token again.

The scheduler event for tick 289 was sent at monotonic `515828.962805779`–`515828.963900075`; the native row-289 control sample timestamp is `515828962.905 ms`, giving only **0.099 ms** from send start to the control sample. The native log nevertheless shows receipt of frame 289 (`sonic.log` lines 1437–1438), after the control sample that produced the stale row. This is consistent with a late asynchronous receipt: the frame-289 message arrived in the native input buffer, but the 50 Hz control cycle had already copied the frame-288 token.

No plan handoff occurred at the failing tick: generation 57 progressed through slots 0, 1, 2, 3, then slot 4. There is no missing or duplicated scheduler entry, and the scheduler reports zero uncovered, repeated, skipped, overlap-fallback, and plan-switch events. The preceding worker result at anchor 278 was one tick late (`available_tick=280` rather than 279), and the observed control loop was already drifting. At tick 288 the Python observation loop was about 9.25 ms after the native sample; the next token was consequently published only at the next control boundary.

In the successful Phase D.1 run, the same interval had stable approximately 14 ms publish-to-consumption margin, and the native rows matched every intended token through 810/810 ticks. The failed run had the same hardware EGL identity, frozen native binary, SONIC bundle and scenario semantics.

## Root-cause classification

**Observed evidence:** the external token transport is asynchronous ZMQ. Native logs confirm receipt of frame 289, while the native post-state CSV proves that the control cycle at tick 289 still used frame-288 values. The Python publisher sends one token just in time after processing the previous control snapshot. A short host-side scheduling/log-following delay removed the normal margin.

**Inference:** the immediate cause is a just-in-time transport race: frame 289 was published too late for the native 50 Hz control sample, so native consumed the latest available frame (288). The strict checker correctly detected a real stale-token consumption; it is not an indexing or numerical-tolerance error. The broader trigger is timing variability in the Python control loop/worker handoff, with the late anchor-278 worker result and the 9 ms observation delay as direct precursors.

**Confidence:** high for stale previous-token consumption and high for the immediate late-publication cause; medium for the exact OS scheduling event that produced the extra delay. `dds_python.log` contains DDS initialization/runtime transport records, but this token path is ZMQ and has no native monotonic receipt timestamp. The native source logs frame receipt order, not the precise receive-to-control ordering, so that timestamp would be the remaining useful confirmation.

This is not a CLI graphics/environment failure, not an offline-gate issue, not a plan switch, and not evidence of a canonical off-by-one error. It is a native synchronization/timing boundary in the existing asynchronous external-token handoff.

## Fix decision

No code was modified in Phase E.1. The obvious remedies—publishing multiple future tokens ahead of time, changing the native handoff to queue by frame index, adding a wait/retry, or accepting the previous token—would change native scheduling/control semantics or weaken the strict invariant. They are therefore not justified as a silent minimal fix.

The next safe instrumentation step is to record a monotonic timestamp and frame index at native ZMQ receipt and at the control-thread token copy, then compare those records with the Python send timestamp. That instrumentation must be logging-only and separately reviewed before any scheduling change.

## Verification and artifacts

No unchanged rerun was performed, as required when the available evidence does not justify a semantics-preserving fix. Existing focused source checks remain available; the prior Phase E run’s 21 grasp-pipeline tests passed. The immutable failure evidence remains under:

- `outputs/grasp_pipeline_phase_e_20261009_v1/closed_eval/sonic.log`
- `outputs/grasp_pipeline_phase_e_20261009_v1/closed_eval/dds_python.log`
- `outputs/grasp_pipeline_phase_e_20261009_v1/closed_eval/failure_traceback.txt`
- `outputs/grasp_pipeline_phase_e_20261009_v1/closed_eval/policy_logs/token_state.csv`
- `outputs/grasp_pipeline_phase_e_20261009_v1/closed_eval/scheduler_events.json`
- `outputs/grasp_pipeline_phase_e_20261009_v1/closed_eval/control_timeline.json`

The Phase D.1 successful 810-tick reference remains unchanged at `outputs/grasp_pipeline_phase_d1/closed_run3/`.

Runtime completion for Phase E remains **not achieved** (`289/810`, strict abort). Grasp success is separately **false** (`0 m` lift before abort). No claim of a completed rollout or successful grasp is made.
