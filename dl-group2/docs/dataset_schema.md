# Expert episode schema v3

Every new episode is generated from the task-independent mirrored pose in
[`configs/neutral_standing.json`](../configs/neutral_standing.json). Both arms
start beside the body with relaxed elbows and open hands. The same pose supports
future left, right and bimanual planners. The present planner moves the right arm;
the left arm remains at its neutral posture reference. All 29 body and 14 hand
joint positions/velocities and references remain in the dataset.

The source balance controller initializes its COM target from this pose's actual
model COM. A one-second standing segment precedes raising the hand outside the
table, crossing above the edge, lowering, closing, lifting and holding. The default
source is 16 seconds: 800 reference/state frames at 50 Hz and 160 RGB frames at
10 Hz per camera. The SONIC body controller executes the entire reference; the
independent hand controller supplies all fourteen hand commands.

## Additive metadata

Existing source, candidate, actual, contact, validation and vision files keep their
layout. Every episode metadata contains:

```json
{
  "schema_version": 3,
  "instruction": "Pick up the red cube and lift it from the table.",
  "camera_role": {
    "head_rgb": "policy_observation",
    "wrist_rgb": "policy_observation_optional",
    "task_overview": "evaluation_only"
  },
  "task_config": {
    "object_name": "task_red_cube",
    "cube_initial_position": [0.4, -0.15, 0.785],
    "cube_initial_quaternion_wxyz": [1, 0, 0, 0],
    "cube_size": [0.07, 0.07, 0.07],
    "cube_size_convention": "full side lengths in metres",
    "cube_mass": 0.1,
    "table_height": 0.75,
    "random_seed": 0,
    "case_id": "center",
    "episode_id": "pilot_xy_center"
  }
}
```

The numerical example above describes the current scene. Actual records read the
initialized MuJoCo body pose, box half-sizes (multiplied by two), body mass and table
geometry. Positions/size are metres, mass is kilograms, and quaternions use wxyz.
The pilot uses explicit XY coordinates without stochastic variation; seed 0 is a
recorded source-generation seed; real-time DDS replay timing is not bitwise deterministic. Historical episodes without a recorded seed use null.

`initial_pose` identifies the pose definition and initialized named joint angles.
`freebase_wbc_002` keeps its original starting pose and is explicitly marked
`legacy_tablefront_preparation`, not neutral compliant. Its additive metadata
upgrade does not regenerate motion, physics or RGB.

## Reference, state and RGB clocks

- `body_ref_q/dq`, `hand_ref_q/dq`: action targets for future learning. Body order
  remains the official 29-joint reference order; hand order remains both seven-joint
  hands, left then right. No array slicing substitutes for name mapping.
- `body_q/dq`, `hand_q/dq`, root state, cube pose and full simulation state: measured
  physical observations from the selected SONIC replay.
- `timestamps.npy`: 50 Hz reference clock. `actual_timestamps.npy` and trial
  `timestamps.npy` identify the reference frames that were actually recorded.
  `sim_timestamps.npy` retains the physical simulator clock, including startup.
- `validation/<selected_run>/vision/images.npz`: uint8 RGB arrays plus
  `reference_frame`, `timestamps`, and `sim_timestamps`. Image timestamps equal
  `episode.timestamps[reference_frame]` exactly. For a complete 800-frame rollout,
  10 Hz image indexes are 0, 5, ..., 795. Offline rendering reads saved actual
  qpos/qvel and never steps physics.
- `validation/<run>/contacts.json` and `report.json`: contacts and full-rate
  physical success/safety metrics. Source contacts stay in `source/contacts.json`.
- `sonic_reference/`: the validated candidate's original reference CSVs; reference
  and actual arrays coexist, including after export.

Only head RGB is the default policy observation. Wrist RGB is optional. Overview
RGB is for evaluation; do not include it as a policy input accidentally.

## Certification and pilot accounting

The original physical criteria remain mandatory: free base, gravity, no weld or
band, actuator-only control, no object attachment or teleport after reset,
SONIC-driven body, independent hands, opposing real contacts, at least 10 cm lift,
at least two seconds of continuous final hold, no fall, finite states, complete
frame coverage and verified SONIC/hand synchronization.

New neutral episodes additionally reject any robot self-contact or left-arm contact
with scene objects during the physical replay. These are evaluated on every physics
step, not inferred from the 50 Hz images. `neutral_audit.json` adds geometry
clearances and sampled pose diagnostics; its sampled counts are supplementary to
the runtime's full-rate safety counts.

`pilot_xy_summary.json` reports each position, source attempts, SONIC replay
attempts, correction history, selected candidate/run, lift/hold metrics and failures.
`attempts` counts SONIC replays, including independent confirmations; source attempts
are separate. The success rate is successful SONIC replays / all recorded SONIC
replays of the final standard. Pre-standard pose experiments remain in
`pilot_development/` and are excluded, with their reason recorded.

`pilot_xy_training_manifest.json` contains only physically certified episodes whose
RGB alignment and exported dataset were checked. It is an eligible episode pool,
not a frame-randomized train/test split. Failed episodes, candidates and RGB remain
on disk but never enter this manifest.

## Commands

Use the project's simulation Python environment and a GPU node for SONIC/EGL:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl
python scripts/run_xy_pilot.py --max-replays 4 --corners-if-stable
```

The runner is resumable. It starts with center and four 2 cm axis offsets. It adds
four corners only when all five positions are valid with at most two replays each.
Each failed complete replay may produce a corrected candidate; every correction
requires another physical replay. Replay attempts are bounded. The pilot never
changes size, mass, friction or illumination.

To add only metadata to an older episode:

```bash
python scripts/standardize_episode.py --episode outputs/expert_episodes/freebase_wbc_002
```

To check an exported episode, including images and reference/actual alignment:

```bash
python scripts/verify_expert_episode.py --episode outputs/expert_episodes/pilot_xy_center
```

Complete failed replays can also be exported with `export_expert_episode.py
--include-failed` for a consistent inspection dataset. Their `expert_valid=false`
and failure reasons are preserved; verification returns exit code 2 and they remain
outside the training manifest.
