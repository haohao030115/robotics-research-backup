# Neutral standing / XY pilot results

Five final-standard positions passed physical SONIC + independent hand validation.
The common task-independent pose is `neutral_standing_v1`; both arms use the same
mirrored configuration, both hands start open, and only the right arm performs the
present grasp. The initialized arm/table geometry clearance is approximately
16.7 cm on each side. All five source initial body poses are exactly identical.

| Position relative to (0.40, -0.15) m | SONIC attempts | Corrections | Final lift | Final continuous hold |
|---|---:|---:|---:|---:|
| center | 2 (includes confirmation) | 0 | 14.32 cm | 4.400 s |
| x + 2 cm | 5 | 4 | 15.61 cm | 4.470 s |
| x - 2 cm | 1 | 0 | 13.21 cm | 4.390 s |
| y + 2 cm | 1 | 0 | 12.04 cm | 4.290 s |
| y - 2 cm | 1 | 0 | 13.96 cm | 4.435 s |

The final-standard replay success rate is 6/10, including one independent center
confirmation. Four of five positions passed directly. The x+2 cm position needed
four reference corrections (gains 0.6, 0.6, 0.6, 0.8); its first four failures pushed
the cube during closing and failed lift/hold criteria. Those failed references,
actual states, contacts, images and reasons remain in its validation history.
These counts describe this small pilot, not a statistical estimate of deployment
reliability. No corner positions or larger batch were generated.

All selected replays have zero falls, zero robot self-contact and zero left-arm
contact with the environment. Robot/table contact is **not zero**: the right thumb
contacts the tabletop during grasp approach/closure. It does not occur during the
initial raising/crossing segment or valid hold. These contacts are explicitly
reported, not filtered out. Every selected replay is free-base, gravity-on, without
weld/band, object attachment, object teleport after reset or scripted body control
during SONIC playback.

The final training pool has five episodes, each with 800 synchronized 50 Hz
reference/state frames and 160 RGB frames per camera at 10 Hz. This is 4,000 state/
action frames and 800 frames per camera (2,400 RGB images across three cameras).
Head RGB is a policy observation, wrist RGB optional, overview evaluation-only.
Both 29-DoF body and 14-DoF hand reference/actual arrays remain available.

Final checks passed for all five episodes and all ten final-standard replay image
archives: RGB frame indexes/timestamps, joint mapping, NPY/CSV agreement, selected
SONIC reference bytes, complete physical frame coverage and hand synchronization.
The original `freebase_wbc_002` was only upgraded additively in metadata; all 38
CSV/reference checksums still match its existing manifest. The upstream
`GR00T-WholeBodyControl` checkout remains clean.

Pose development is accounted separately: five source rollouts and ten SONIC
replays remain under `pilot_development/`. Earlier neutral drafts had startup
instability or hand/hip self-contact and are excluded from training. Together with
the final pilot, twenty SONIC replays were run.

Local artifacts (ignored by Git):

- `outputs/expert_episodes/pilot_xy_summary.json`: per-position attempts, failures,
  corrections, chosen reference/run and physical metrics.
- `outputs/expert_episodes/pilot_xy_training_manifest.json`: only five eligible
  selected episodes; not a random frame-wise train/test split.
- `outputs/expert_episodes/pilot_xy_failure_manifest.json`: four failed
  final-standard replays plus the excluded development archive.
- `outputs/expert_episodes/pilot_xy_dataset_audit.json`: final dataset checks.
- `outputs/expert_episodes/PILOT_XY_RESULTS.md`: compact generated summary.
- `outputs/expert_episodes/pilot_xy_center/validation/sonic_confirm_001/vision/`:
  `task_overview.mp4`, `head_rgb.mp4`, `wrist_rgb.mp4`, `images.npz`.

The neutral starting pose works in this pilot. Nearby XY generalization is uneven:
three axis offsets work directly, while x+2 cm needs substantial calibration.
Further work should first measure repeatability of that corrected candidate before
expanding the sampling region.
