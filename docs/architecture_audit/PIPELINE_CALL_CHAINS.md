# Pipeline call chains

All tensor shapes and timings below are **VERIFIED** where cited. A source named without an observed execution report is **INFERRED** as a callable entry point.

## 1. Expert motion execution

**Kimodo:** `kimodo/scripts/generate.py:286` → Kimodo `load_model` → generated motion → `MujocoQposConverter` export (`generate.py:384`) → `kimodo_to_t7_reference.py:163` → 50 Hz body/root intermediate (29 body joints, root position/quaternion, velocities) → local case adaptation/IK (`dl-group2/scripts/adapt_model_reference.py`) → candidate reference (`reference.npz`, metadata, provenance).

**ARDY:** `run_pipeline.py:58-74` → `prepare_conditions.py` → `generate_segmented.py` (25 FPS, 406 frames, native overlap/history) → `adapt_ardy.py:40-113` (resample to 810 frames/50 Hz, SLERP, wrist target and IK, optional calibration) → isolated replay script → validation.

**SONIC execution:** candidate reference → `canonical_va.native_replay.export_native_reference` → native `g1_deploy_onnx_ref` → DDS LowState/LowCmd → `TabletopDDS.step` → body torque and independent hand PD → MuJoCo physics → `PhysicalGraspEvaluator` (`expert_trajectory.py:148-215`). Native binary creates encoder input 1762D, 64D post-FSQ token, decoder input 994D and 29D action (`native_replay.py:24-25`; `musa_onnx.py:71-117`).

**Cross-process boundaries:** Python canonical recorder ↔ native C++ process through PTY/DDS; native C++ ↔ Python/MuJoCo bridge through Unitree SDK/DDS; isolated network namespace in batch mode. Timing is asynchronous at native transport, with 200 Hz MuJoCo, 50 Hz policy/control and 500 Hz command writer.

## 2. Canonical dataset production

`canonical_va.batch` → `unshare --net`, loopback, `python -m canonical_va.pipeline` → `schema.validate` (50 Hz, shapes `[T,29]`, `[T,14]`, root `[T,3]`, quat `[T,4]`, wxyz) → `native_replay.replay` or immutable recorded trace → `native_truth.materialize` (joins decision, observation sequence, publication and receipt IDs) → `forensic.verify` when full audit is required → `compiler.compile_samples` → `va_dataset/samples.npz`, RGB copies, `sample_mapping.jsonl`, metadata → `validate_export` → `training_manifest.json`.

The canonical sample contract is three observations with a validity mask, active phase only, future execution offsets 2–6, body post-FSQ tokens `[N,5,64]`, absolute hand references `[N,5,14]`, RGB 10 Hz and control 50 Hz (`compiler.py:43-70`). `CanonicalV4` extends those authored IDs to `[N,10,64]` and `[N,50,14]`; it drops incomplete tails rather than padding (`v4/dataset.py:87-112`).

## 3. VA training

`grasp_pipeline.training.run` → `v4.train_merged.main` or V4 trainer → `CanonicalV4` loaders → train-only normalization (mean/std clamp `.05`) → V4 policy → losses → periodic train/validation metrics → checkpoint with model state, manifest/provenance, configuration and offline gate. Direct single-episode V4 training is `v4.training.main`; BCZ hand tuning loads a prior V4 checkpoint, adds closure-intent head and uses hand/trigger/advance losses (`v4_0_bcz_hand/train_full.py:18-37`).

Inputs: RGB 128×128, current body 68D = 29 q +29 dq +4 root quat +3 angular velocity +3 gravity, current hand 28D =14 q +14 dq. Outputs: body tokens 10×64, hand reference 50×14, 3 phase logits, scalar progress; BCZ additionally emits a closure-intent logit. V4 has no temporal observation history in the model; temporal targets are carried by canonical target horizons and the absolute scheduler.

## 4. VA closed loop

`run_token_v4_closed_loop.py` → `v4.rollout.run` (or `v4_0_bcz_hand` rollout) → frozen canonical manifest/checkpoint/provenance and offline gate → `TabletopDDS` reset → spawned vision worker → validated MTT EGL renderer → `Inference.predict` (RGB resize, MUSA model) → plan returned → `PlanBuffer.add` at authored lead → `ExternalTokenAdapter.message` packs protocol-v4 64D token → ZMQ → native `NativeTokenSonic`/SONIC decoder → native CSV token/action logs → strict comparison of consumed token and intended tick → `TimestampedHand.set` → `TabletopDDS.step` → snapshot/contact/grasp evaluation → new RGB/state request.

The open BCZ report is a useful boundary case: `recorded_ticks=810`, `expected_ticks=810`, `uncovered_ticks=0`, `strict_runtime_pass=true`, but `physical_grasp_success=false`, `max_lift_m=0.02276`, hold `0`. That is a complete runtime execution with a failed task criterion, not a successful grasp.

## Validation and tests

Canonical contract/native/production tests live in `canonical_va/tests/`; V4 model, inference, scheduler, dataset and diagnostics tests live in `va_training/tests/`; `grasp_pipeline/tests/` validates adapters and artifact contracts. Native physical tests require the container, MUSA, EGL and isolated DDS; offline tests do not.
