# Actual closed-loop diagnostic videos

Run from the host with a fresh container output directory:

```bash
bash /home/group2/va_training/diagnostics/reproduce_video.sh /workspace/group2/outputs/va_v2_video_repro
```

The learner uses the retained K=5, offset=2 checkpoint at
`outputs/va_v2_diagnosis/chunk5_training/checkpoints/latest.pt`, with the existing
asynchronous paired buffer and frozen SONIC release decoder. Corrective variants
are not substituted. Episode `xy_random_v1_random_0000` supplies only the existing
initialization to the learner. The oracle uses GT native tokens and GT hand
references through the existing physical GT/GT runner. Each process has an
isolated DDS network namespace.

Both live runs finish before rendering starts. The renderer reads their actual
post-step `physics.npz` qpos/qvel and calls `mj_forward` to display those states;
it never calls `mj_step` or drives the robot with expert joint trajectories.
This preserves the inference/controller timing while showing recorded physical
motion, including the actual cube and hand states. Initialization bootstrap file
hashes, frozen SONIC/runtime hashes, episode and success criteria must match.

The `videos/` directory contains `learner.mp4`, `expert.mp4` and
`expert_vs_learner.mp4` (expert left). Existing overview/head/wrist cameras are
used. Videos run at 25 FPS in real control time; the original synchronized logs
retain all 800 states at 50 Hz. All encoded frames are decoded for verification.

Overlays display control time, absolute simulation time, step, cube height/lift,
right-hand measured mean, applied predicted/GT hand reference mean, nearest
expert index, signed phase lag and wrist separation. Plan phase windows are
nominal labels from the prior diagnosis, not policy phase predictions. Nearest
states use body q/dq, wrist position and hand q with physical scales
0.1 rad / 1 rad/s / 0.05 m / 0.2 rad; matching is diagnostic only.

`learner_synchronized.npz` contains measured body/hand q/dq, native-consumed body
tokens, applied hand references, previous raw actions, cube/wrist/root poses,
nearest indices, clocks and full qpos/qvel/ctrl. Unconsumed raw predictions,
capture anchors and intended execution indices remain in
`learner/predictions.npz`; native logs, handoffs and timing remain alongside it.

`videos/report.json` records checkpoint/source hashes, decoded video counts,
task outcomes and event times. The reported divergence is the first wrist
separation above 5 cm for five consecutive 50-Hz states. Hand closure uses a
right-hand mean increase above its initial value by 0.10 rad for five states,
reported separately for measured joints and applied references. These are
explicit diagnostic thresholds, rather than subjective event annotations.
