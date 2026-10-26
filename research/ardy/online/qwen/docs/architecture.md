# Architecture and control boundary

The current SONIC reference controller loads complete motion folders at process
startup. The new target architecture uses five event-driven segments:

```text
START → RAISE_HAND → LOWER_TO_BLOCK → GRASP → LIFT_WITH_BLOCK
```

Finger motion is not stored in a time-indexed CSV. The GRASP gate is opened only
when live geometry reports a pregrasp distance below 3.5 cm and vertical error
below 2.5 cm. Closure then advances from contact feedback and stops tightening
once thumb plus an opposing finger group touch the object. Lifting is forbidden
until those contacts remain stable.

Until reference hot reload is added, the runnable implementation closes the
body-motion loop between attempts:

```text
run physical attempt → collect live telemetry → Qwen/geometry decision
→ create new ARDY constraints → generate reference → restart SONIC → retry
```

`telemetry.jsonl` is written during physics execution and can be tailed by a UI
or supervisor. Qwen is intentionally called after an attempt because the loaded
motion cannot be replaced safely while the existing SONIC process is playing.

For true within-episode replanning, add a versioned reference-stream interface
to `g1_deploy_onnx_ref` with an atomic swap at a 50 Hz frame boundary. The new
reference must start from measured body/root state, pass joint/velocity/workspace
validation, and preserve hand/body clock synchronization. Qwen should remain a
slow event-driven planner; a deterministic state machine and geometry layer must
own high-rate safety and coordinate conversion.
