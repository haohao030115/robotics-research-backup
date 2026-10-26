# Free-base physical expert — freebase_wbc_002

`expert_valid=true`. The successful source is a free-base, gravity-on, contact-only MuJoCo rollout. The final dataset exposes the independently successful SONIC + hand replay `sonic_safe_confirm_001`; the same candidate also succeeded in `sonic_safe_003`.

Method: contact-constrained whole-body inverse dynamics maintains feet, COM and pelvis orientation; arm-only wrist IK supplies a grasp geometry reference, and independent 14-joint torque PD closes the hand. This uses existing MuJoCo, NumPy and SciPy, without installing anything. The inverse-dynamics contact wrenches are calculations used to derive joint torques, never applied forces. Only the 43 robot actuators enter the physics. No weld, elastic band, object attachment, post-reset object state writes or external force is used.

SONIC initially introduces body tracking and root drift, so arm references were corrected using measured Cartesian errors at the actual free-base pose. The final corrected reference is `candidates/cartesian_safe_003`. This is a calibrated command reference, not an assertion that SONIC precisely reproduces the original source joint states. Source, every candidate and every failed replay remain available.

## Verified results

| Metric | Free-base source | Final SONIC + hand |
|---|---:|---:|
| Simulation duration | 13 s | 13 s playback |
| Max cube lift | 0.150670 m | 0.146917 m |
| Final cube lift | 0.140661 m | 0.133590 m |
| Final continuous contact hold | 4.520 s | 4.385 s |
| Falling steps | 0 | 0 |
| Frames | 650 | 650 |
| Robot-table contact physics steps | 335 at 500 Hz | 167 at 200 Hz |

Final opposing contacts: right thumb and index with positive normal forces, approximately 6.4–6.7 N per contact point. No cube-table contact during hold. Robot-table contacts are explicitly recorded: left wrist/middle finger during the early transition and right thumb during grasp; sampled table contacts end at frame 278, before lift, and there are none during valid hold. The final robot pelvis remains above 0.7536 m; maximum sampled tilt is 5.158 degrees.

Frame coverage is complete. Applied hand-reference max error is 0; SONIC target CSV agrees within 4.99e-6 rad. Body-reference RMSE is 0.11518 rad: this is a successful physical playback, not perfect tracking. The independent repeat also lifted 0.133590 m and held 4.385 s. Final metadata has an empty failure_reason list.

## Dataset and clock

Body reference/actual q and dq: `[650,29]`. Hand reference/actual q and dq: `[650,14]`. Timestamps: `[650]`, 50 Hz, 0 through 12.98 s. Root pos/lin_vel/ang_vel: `[650,3]`; root/cube quaternions: `[650,4]`, wxyz. Cube pos: `[650,3]`. Actual root velocities use world-frame pelvis-origin Jacobians. SONIC root-reference angular velocities are kinematic derivatives of source quaternions. Saved `sim_timestamps` retain the actual physical capture times.

The root CSV/NPY actual arrays correspond to the selected successful SONIC replay. Original inverse-dynamics actual arrays remain in `source/`. Reference arrays are synchronized through the SONIC 50 Hz active playback tick index, checked against the policy target stream and explicit current_frame log. Hands use that exact same reference index, never a separate wall clock. The hand controller uses Kp=12, Kd=0.3, actuator torque limits, and the position target; hand_ref_dq is saved but not used by this first-version PD.

RGB observations are rendered from saved actual qpos/qvel, without advancing physics, at 10 Hz: `[130,240,320,3]` per camera (`head_rgb`, `wrist_rgb`, `task_overview`). Each images.npz has frame indices and reference timestamps; sim_timestamps are obtained by indexing the corresponding saved rollout. Videos and the image archive are in `validation/sonic_safe_confirm_001/vision/`; original source images are in `source/vision/`.

## Exact joint order

The body columns are deployment **IsaacLab reference order**. They were read from `~/GR00T-WholeBodyControl/gear_sonic/envs/env_utils/joint_utils.py:G1_ISAACLab_ORDER` and independently cross-checked with both C++ permutations and named motor comments in `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp`. `motion_data_reader.hpp` consumes numeric CSV columns directly. Hand order comes from `G1_HAND_JOINTS` in the same Python source. All model qpos/qvel/actuator addresses are resolved by joint name. The episode's joint_names.json and metadata.json retain full names and address mappings.

Body, zero-based:

```text
00 left_hip_pitch_joint
01 right_hip_pitch_joint
02 waist_yaw_joint
03 left_hip_roll_joint
04 right_hip_roll_joint
05 waist_roll_joint
06 left_hip_yaw_joint
07 right_hip_yaw_joint
08 waist_pitch_joint
09 left_knee_joint
10 right_knee_joint
11 left_shoulder_pitch_joint
12 right_shoulder_pitch_joint
13 left_ankle_pitch_joint
14 right_ankle_pitch_joint
15 left_shoulder_roll_joint
16 right_shoulder_roll_joint
17 left_ankle_roll_joint
18 right_ankle_roll_joint
19 left_shoulder_yaw_joint
20 right_shoulder_yaw_joint
21 left_elbow_joint
22 right_elbow_joint
23 left_wrist_roll_joint
24 right_wrist_roll_joint
25 left_wrist_pitch_joint
26 right_wrist_pitch_joint
27 left_wrist_yaw_joint
28 right_wrist_yaw_joint
```

Hand, zero-based:

```text
00 left_hand_index_0_joint
01 left_hand_index_1_joint
02 left_hand_middle_0_joint
03 left_hand_middle_1_joint
04 left_hand_thumb_0_joint
05 left_hand_thumb_1_joint
06 left_hand_thumb_2_joint
07 right_hand_index_0_joint
08 right_hand_index_1_joint
09 right_hand_middle_0_joint
10 right_hand_middle_1_joint
11 right_hand_thumb_0_joint
12 right_hand_thumb_1_joint
13 right_hand_thumb_2_joint
```

## Reproduce the validated SONIC replay

On the existing GPU host; no other domain-0 controller should be running:

```bash
ssh gpu-4080-413 'cd ~/dl-group2; export PYTHONPATH=$HOME/GR00T-WholeBodyControl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl; ~/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/run_sonic_grasp.py --episode outputs/expert_episodes/freebase_wbc_002 --candidate cartesian_safe_003 --band off --startup-support none --startup-body freebase-wbc --warmup-frames 0 --run-name replay_$(date +%Y%m%d_%H%M%S) --no-images'
```

`--band off --startup-support none` explicitly disables elastic band throughout initialization and playback, independently of the official simulator YAML. A no-force whole-body standing controller runs only before the first playing tick; during every playing physics step all 29 body torques come from SONIC motor commands. The hand controller remains separate. Legs/body/hand never receive cube state edits. Playback runs the existing decoder/encoder ONNX through the existing C++ deployment binary. The runner stages a single candidate motion directory and passes its parent as the motion-data argument, with encoder mode 0. Runtime command and binary hash are saved in each validation metadata.json.

Do not use the bare official simulator with its default band-on YAML for certification. The project runner composes the same G1 + ground + physical table + red cube + activated fingers and bypasses the band unless explicitly requested for debugging. Any band-on, startup-band, welded replay or supported source is rejected by expert certification.

Generate a fresh free-base source (choose a new directory):

```bash
cd ~/dl-group2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/generate_freebase_expert.py --episode outputs/expert_episodes/freebase_new
```

This saves an uncertified candidate until SONIC replay succeeds. The measured original actual source is not itself the final calibrated SONIC command. The iterative correction history is preserved as: `sonic_freebase_001` → `cartesian_ilt_001` → `sonic_ilt_001` → `cartesian_ilt_002` → `sonic_safe_001` → `cartesian_safe_002` → `sonic_safe_002` → `cartesian_safe_003`. The final step uses correction gain 0.6; earlier steps use 0.8. `refine_freebase_reference.py` performs the named correction; candidate metadata records each parent, gain and input hash.

After a new successful replay, refresh exports and audit:

```bash
~/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/export_expert_episode.py --episode outputs/expert_episodes/freebase_wbc_002
~/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/verify_expert_episode.py --episode outputs/expert_episodes/freebase_wbc_002
```

SONIC-compatible files live in `sonic_reference/`: joint_pos.csv, joint_vel.csv, body_pos.csv, body_quat.csv, body_lin_vel.csv, body_ang_vel.csv, metadata.txt and info.txt. The six CSV files are byte-for-byte copies of the physically validated candidate. Standard generic SONIC column headers are retained; actual names are explicit in metadata.json and joint_names.json. The root also contains named CSV reference/actual state arrays, root/cube arrays, actual timestamps, complete qpos/qvel/control snapshots and csv_manifest.json.

## Code changes

Maintenance cleanup: current shared IK/planning lives in `scripts/grasp_reference.py`, and CSV/reference export in `scripts/reference_io.py`. The supported-pelvis entrypoints and unrelated experiments have been removed. See [current code layout](code_layout.md) for the 13 retained modules and the clean official checkout. The existing episode RESULT.md records the original generation history.

New project scripts: generate_freebase_expert.py, refine_freebase_reference.py, render_expert_rollout.py, export_expert_episode.py. Existing project scripts updated: run_sonic_grasp.py (unassisted startup and certification guards), expert_trajectory.py (actual world-frame root velocities), verify_expert_episode.py (free-base source support and binary/CSV consistency audit). README.md points to this result. During expert generation, neither third_party nor the NVIDIA official checkout was modified, and no environment was reinstalled. Later maintenance restored the obsolete official DDS patch and removed the old reference copy; see code_layout.md.

Validation is specific to this center-cube episode and model. It demonstrates one repeatable expert and does not establish robustness across cube placements.
