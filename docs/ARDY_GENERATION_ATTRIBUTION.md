# ARDY generation attribution and adaptation audit

This audit traces fresh ARDY run `ardy_fresh_20261016_01`, executes a fresh five-seed fixed-center study, a three-position spatial pilot, and a no-calibration ablation. It preserves all failures and does not treat historical cached retargets as fresh generation evidence.

## Verified accepted lineage

The accepted raw motion is `/home/group2/outputs/canonical_production/ardy/ardy_fresh_20261016_01/artifact/source/reference/motion/ardy_motion.npz`. Its `generation.json` records `model=ARDY-G1-RP-25FPS-Horizon52`, `source_fps=25`, `actual_frames=406`, `native_ardy_api_calls=5`, seed 2, five segmented prompts, 10 denoising steps, and checkpoint hashes for `config.yaml`, `denoiser.safetensors`, and `tokenizer.safetensors`. The file is produced by the maintained `research/ardy/overlap/generate_segmented.py` path and is not copied from a prior episode.

The accepted adapted input is `/home/group2/outputs/canonical_production/ardy/ardy_fresh_20261016_01/artifact/replay/input_reference.npz`. It is generated from the fresh candidate by `research/ardy/adaptation/adapt_ardy.py`; the native replay is under `/home/group2/outputs/canonical_production/ardy/ardy_fresh_20261016_01/artifact/replay/`. The replay passed the ARDY verification gate with 0.137346 m lift, 4.840 s geometric opposing-contact hold, `physical_success=true`, and `expert_valid=true`. The canonical factory then produced the accepted episode and 161 exported rows (152 complete V4 horizons) under `/home/group2/outputs/canonical_production/ardy/ardy_fresh_20261016_01/canonical_episode/`.

The fresh generation, adaptation, replay, canonical compilation, and sample hashes are recorded in `/home/group2/outputs/canonical_production/production_summary.json`.

## Actual executed transformation chain

`research/ardy/generation/prepare_conditions.py` constructs the fixed-center scene and structured constraints. The cube position is taken from the shared case (`center` is `[0.4, -0.15]`), not parsed from the prompt. It builds a 406-frame, 25 Hz conditioning timeline, matrix-preserving pose constraints, root constraints, planned wrist targets, and an independent initial hand reference. The model’s source output is full-body/root motion; `generation.json` explicitly records `model_generated_hands=false`.

`research/ardy/overlap/generate_segmented.py` loads the local ARDY checkpoint and encodes each of the five prompts once. It uses the native ARDY diffusion model with `diffusion_steps=10`, `cfg_weight=[2,4]`, segment lengths `[100,100,50,50,106]`, five-frame overlap, and `native_history_frames=4`. For segment 0 it samples from the task constraints. For each later segment it takes the last five generated frames, reconstructs full-body and end-effector constraints, prepends one native history token, and samples the new prompt. The five-frame overlap is linearly blended in model feature space. `segments/00_native.npz` through `segments/04_native.npz` retain each native sample before stitching. This is learned generation plus autoregressive history conditioning; the overlap rule is deterministic once the model samples are fixed.

`research/ardy/adaptation/adapt_ardy.py` then performs these actual steps:

| Stage | Implementation | Successful center behavior |
|---|---|---|
| Resampling | `adapt_ardy.py` source interpolation | 25 Hz, 406 frames to 50 Hz, 810 frames; linear joint/root interpolation and quaternion SLERP. No padding or extrapolation. |
| Wrist target | `GraspReference` and `setting.npz` | Uses structured planned world wrist position/quaternion from the task, then blends entry over 1.5 s. |
| Wrist projection | per-frame `residual`, `projectedp`, `projectedr` | Position half-width 0.005 m and rotation radius 0.04 rad; refinement blend 1–2 s. |
| Tracking calibration | calibration branch | Enabled in the accepted run with fixed center calibration `center_segmented_tracking_seed0.json`; position/rotation biases blend from 6–8 s and close-to-hold interpolation from 10–12 s. |
| Clearance | target construction | +0.03 m, blended 4–7 s and decayed 6–8 s. |
| Body IK | `GraspReference.ik()` | Runs for every 50 Hz frame after frame zero. Maximum accepted IK error is recorded in `candidate/adaptation.json`. |
| Hand schedule | final adapter block | Model hands are not used. Left hand is zeroed; right hand follows an independent cubic preclose from 7–8 s to 40%, then closes to full at 10 s. |
| SONIC export | `convert_single_motion` | Produces canonical 29-joint body motion, body FK fields, velocities, hand CSVs, and the adapted qpos stream. |

The adapter is task-conditioned through `setting.npz` wrist targets and case-specific initial scene state. The center tracking calibration is manually measured evidence and is fixed across the five center seeds; it is not learned from each new rollout. The adapter can materially overwrite generated wrist and arm motion because it changes the target pose before per-frame IK.

Native replay uses `research/ardy/runtime_support/replay_ardy_isolated.py` through the isolated DDS shell. The original SONIC controller, MuJoCo scene, hand gains, physics timing, and acceptance thresholds are unchanged. ARDY replay stores complete state and contact traces in `simulation.npz`, `physics_steps.json`, and `result.json`; the canonical factory separately reruns the accepted reference and compiles RGB, state, token, and hand-reference joins.

## Measured fixed-center diversity

The fresh five-seed study manifest is `/home/group2/outputs/ardy_attribution_20261016_v2/center_study_manifest.json`. Seeds 1–5 all completed fresh generation, adaptation, native replay, and verification with one fixed center calibration policy and no per-seed tuning.

All five runs passed `physical_success` and `expert_valid`. Lift ranged from 0.133908 to 0.146667 m; qualifying hold ranged from 4.835 to 4.875 s. The diagnostic table-contact counts were 23, 27, 20, 34, and 21 steps. The strict no-table flag remains false because this ARDY verifier treats robot-table contact as diagnostic while preserving the geometric grasp gate.

The comparable metrics are in `/home/group2/outputs/ardy_attribution_20261016_v2/study_metrics.json`; aligned stage arrays are in `/home/group2/outputs/ardy_attribution_20261016_v2/stage_comparisons/seed_1.npz` through `seed_5.npz`.

Across the five seeds, raw-to-final body RMS differences are 0.555–0.562 rad. Raw wrist-to-final-target RMS position differences are 0.0167–0.0176 m, with maximum adapter position corrections of 0.0397–0.0450 m. Pairwise body RMS diversity averages 0.03865 rad for raw ARDY output and 0.00650 rad after adaptation. The final motion therefore retains some variation but shows an approximately sixfold reduction in this metric. Executed body tracking against the final target is approximately 0.146 rad RMS across the center study.

For seed 2, the phase-resolved raw-to-final body RMS values are stored in the metrics file. The hand closure onset is approximately 7.02 s and is adapter-authored, not model-authored. Wrist target errors and executed wrist tracking are reported separately from body target errors.

## Seed dependence and calibration

`prepare_conditions.py` writes the requested seed into the source setting. `generate_segmented.py` calls ARDY’s `seed_everything(cfg['seed'])` immediately before segmented inference. The five segments share that seeded process; there is no candidate ranking or retry selection in the maintained pipeline. The accepted seed 2 is therefore one fixed-seed draw, not a selected winner among hidden alternatives.

The accepted center run uses a fixed calibration file whose evidence is identified as seed-0 development tracking evidence. `adapt_ardy.py` validates the calibration source hashes and the case identity, but it does not fit new values during the run. The successful five-seed study reuses those same center values, so its success does not depend on per-seed manual tuning. It does depend on a center-specific calibration policy.

## Controlled ablation

The no-calibration ablation is `/home/group2/outputs/ardy_attribution_20261016_v2/ablations/seed2_no_calibration/`. It reuses the same fresh seed-2 source and disables only the tracking calibration. Replay integrity remains valid, but the cube reaches only 0.004971 m, has zero qualifying hold, and fails `physical_success`. This demonstrates that the measured compensation materially contributes to the accepted grasp. It does not prove that calibration alone explains every success because wrist projection, clearance, IK, and hand scheduling remain enabled.

The model-generated-hand ablation is not meaningful as a separate replay: ARDY declares `model_generated_hands=false` and supplies no final dexterous hand trajectory. The hand schedule is therefore an explicit downstream dependency. Removing it would produce an incomplete reference and should fail the schema rather than be forced into physics.

## Spatial pilot

The fresh spatial manifest is `/home/group2/outputs/ardy_attribution_20261016_v2/spatial_study_manifest.json`. It uses seed 2 at `center`, `near`, `left`, and `right`, with the same default adapter and no CLI calibration for every case. The fresh center case passed with 0.1371 m lift and 4.840 s hold. The fresh near case passed with 0.125960 m lift and 3.175 s hold. The fresh left case physically lifted 0.193903 m but failed the 2 s hold gate with 1.385 s qualifying hold. The fresh right case did not grasp the cube (0.007966 m lift and zero hold). All failures are retained under `/home/group2/outputs/ardy_attribution_20261016_v2/spatial/`.

The pilot shows that ARDY and the task-conditioned adapter respond to case changes, but acceptance is not uniform across the reachable cases tested. It does not establish full workspace coverage or independent seed-by-position robustness. Historical five-position artifacts remain context only and are not counted as fresh trials here.

## Comparison with Kimodo

Under aligned center-study definitions, Kimodo’s raw pairwise body RMS diversity was 0.05183 rad and its adapted value was 0.00443 rad; ARDY measured 0.03865 rad raw and 0.00650 rad adapted. Kimodo therefore showed more raw variation and stronger collapse in this comparison, while ARDY retained more post-adaptation variation. The configurations are not identical: Kimodo used 30 Hz/487-frame source motion, a 0.05 m accepted clearance, contact refinement, approach refinement, and seed-1 measured calibration; ARDY used 25 Hz/406-frame source motion, 0.03 m clearance, five-frame native history/overlap, bounded wrist projection, independent hands, and center tracking calibration. These numbers support an attribution comparison, not a head-to-head success ranking.

## Conclusions

ARDY generates segmented full-body/root motion with native autoregressive history and structured task constraints. It does not generate the final dexterous hand trajectory. The adapter constructs the final hand motion, projects and calibrates the wrist target, adds clearance, and solves arm IK at 50 Hz. In the center study, raw diversity survives partially but is reduced by about sixfold. The fixed-center calibration is reusable across seeds but is tied to the center case. The spatial pilot demonstrates fresh near-position success and side-position failures, so automated spatial data production is feasible only as a bounded, case-qualified process today.

Scaling to hundreds or thousands of episodes requires a calibration policy valid across positions, a position-conditioned correction model with bounded errors, automated rejection and retry accounting, and a larger spatial-by-seed study. The current evidence supports fresh center and near episodes under the tested policy; it does not support broad workspace coverage or closed-loop robustness claims.
