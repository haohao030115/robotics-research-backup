# Canonical VA startup timing experiment

2026-10-09. One fresh native production run was completed for
`reference_episode_008` at
`outputs/grasp_pipeline_startup_timing/production_reference_episode_008_attempt_0002/`.
The first directory attempt was rejected before native launch because a
root-owned stale output directory was not fresh; it is preserved as evidence
and was not used for measurements.

## Result

The native command was run once in `va-train` with the existing MTT EGL
profile, isolated network namespace, `--mode production --backend native`,
`--retention training`, and the existing QA registry. No training or evaluation
was started.

| Check | Result |
|---|---|
| Runtime | complete; 800 control ticks / 3,200 physics steps |
| Physical grasp | PASS; 0.123886 m maximum lift, 4.365 s hold |
| Fall / missing ticks | 0 / 0 |
| Renderer | `Moore Threads`, `MTT S4000`, EGL; software=false |
| QA | production Basic (`full_reconstruction_performed=false`) |
| Retention | training; 159 V4 samples |
| V4 Dataset Builder | PASS, `dataset_exp_4265b29102a3_0001` |

The authoritative validation is in
[`validation_summary.json`](../../outputs/grasp_pipeline_startup_timing/production_reference_episode_008_attempt_0002/validation_summary.json).
The builder check is under
[`dataset_builder_check_0001`](../../outputs/grasp_pipeline_startup_timing/dataset_builder_check_0001/).

## Wall-clock comparison

| Stage | Previous accepted run | Controlled benchmark | Change |
|---|---:|---:|---:|
| Startup | 43.0248 s | 39.2329 s | -3.7919 s |
| Native rollout | 18.7125 s | 18.4647 s | -0.2478 s |
| Trace materialization | 9.9309 s | 9.7925 s | -0.1384 s |
| RGB rendering | 1.1813 s | 1.1823 s | +0.0010 s |
| Sample compilation | 0.6890 s | 0.6691 s | -0.0198 s |
| Validation / QA | 2.4891 s | 2.4995 s | +0.0105 s |
| Disk writing | 1.4279 s | 1.4612 s | +0.0334 s |
| **Total** | **77.7173 s** | **73.6025 s** | **-4.1148 s** |

The controlled benchmark recorded 7,378 waiting physics steps and 36.890 s of simulated time,
versus 8,157 and 40.785 s in the previous run. This is run-to-run readiness
variation; no stepping, polling, sleep, controller, DDS, MuJoCo, or reference
behavior was changed.

## Instrumentation limitation

The requested event timestamps were first attempted by adding fields to
`canonical_va/native_replay.py`. The compiler's source policy correctly rejects
that module as runtime-critical, so the change was reverted before the real
run; its pinned hash remains unchanged. An external `sys.settrace` probe was
then injected for the benchmark, but the container has an isolated `/tmp`, so
the probe did not persist its event file. Consequently the accepted artifact
contains stage timings but does **not** contain trustworthy per-event timestamps
for process launch, `Init Done`, LowState receipt, standby completion, playback
request, and first playing tick. Those intervals cannot be reconstructed from
the retained training profile without inventing evidence.

The largest measured interval is therefore the aggregate startup phase
(39.2329 s), which includes native launch/readiness, standby polling, and the
required pre-playback MuJoCo stepping. The exact readiness sub-interval
responsible for the delay remains unresolved. No optimization is proposed or
implemented. A follow-up should place the probe in a container-visible path or
make an explicitly reviewed, policy-aware instrumentation-only change, then
repeat the single-run benchmark with event records retained.

## Files and verification

No runtime source file was modified. `canonical_va/native_replay.py` matches
the pinned critical source hash, and Python compilation passed. The benchmark
used the original native binary/controller and produced `expert_valid=true`,
complete reference-frame coverage, valid token/control counts, hardware EGL
identity, production Basic QA, and a builder-readable training retention.
