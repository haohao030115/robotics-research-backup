# Group 2 Visuomotor Learning System: Final Technical Report

**Evidence snapshot:** 2026-10-18 (Asia/Shanghai)  
**Authoritative VA artifact:** [`outputs/va_v4_1_kimodo_rule_20261009/checkpoint.pt`](../../outputs/va_v4_1_kimodo_rule_20261009/checkpoint.pt)  
**Authoritative training data:** [`expert_datasets/train_dataset_v0`](../../expert_datasets/train_dataset_v0)  
**Scope:** source-verified technical reference for motion generation, canonical expert-data construction, V4.1 training, offline evaluation, and native learned-policy evaluation.

## Evidence status and model designation

This report uses three evidence labels. **Verified** means that the value is present in a source file, checkpoint, hash, or execution trace in this workspace. **Reported** means that a preserved historical report states the value but the complete primary bundle is not available for an independent replay. **Not verified** means that the current evidence does not establish the claim.

The final trained/validated VA artifact is **V4.1**, not the historical V4 prior. Its checkpoint metadata contains `v41=true`, a `grasp_ready_head`, epoch 50, and 3,502,977 trainable parameters. The file SHA-256 is `92de206b703d66cea50dfd8b92296778d2400d9969e3e964527a8bbd2328143f`; the runtime-compatible V4-only copy used by the native rollout is `rollout_checkpoint.pt` (SHA-256 `1beb45426b9b072873cc8e48f6ab2f2676cc355f916065482eb824660ed90e67`).

The V4.1 training run finished and passed its recorded offline gate. The native learned-policy rollout completed all 810 expected control ticks and passed scheduler integrity, but **did not physically grasp the cube**. It reached 0.0198275 m maximum lift, 0 m continuous hold, had 107 robot-table contact steps and 640 self-contact steps, and left the cube on the table. Therefore “successful VA model” is used below only for the final completed training/offline-validation artifact; no learned checkpoint has verified physical task success in the current evidence. Successful expert replays are deterministic demonstrations and are kept separate from learned-policy results.

The later 21-episode Unified Adapter V2 dataset is explicitly excluded from the V4.1 provenance. The V4.1 checkpoint points to the older combined V4 dataset, and the model was trained from the byte-verified `expert_datasets/train_dataset_v0` tensor export documented here.

## 1. Motivation and system boundary

The research problem is to convert physically executable motions from text-conditioned motion generators into demonstrations for a visuomotor action policy while preserving the semantics of an existing SONIC whole-body controller. Kimodo and ARDY generate reference motions; deterministic adaptation maps them to the G1 coordinate, timing, and hand contracts; native SONIC and MuJoCo execute and record those references; the Canonical Data Factory rejects traces that cannot be joined without ambiguity; and VA learns current-observation-to-future-command mappings.

The learned component replaces the **reference-selection layer** at deployment. It predicts future 64-D SONIC body tokens and absolute 14-D hand references from the current RGB observation and current robot state. SONIC continues to own decoder history, ONNX/MUSA inference for the body controller, motor scaling and permutation, DDS publication, body PD, timing of its native state machine, and the independent hand PD controller. The learned policy never changes the protected SONIC arithmetic or joint order.

The end-to-end architecture is shown in [Figure 1](figures/system_architecture.svg), and the generator-to-execution boundary is shown in [Figure 2](figures/generation_to_execution.svg). Deterministic components are the generators' adapters, native controller, MuJoCo integration, hand controller, data validation, and scheduler. The only learned component in the final VA loop is V4.1 (plus the frozen upstream Kimodo/ARDY motion generators used to make expert references).

## 2. End-to-end pipeline

1. **Motion generation.** Kimodo produces a 30 Hz, 487-frame body/hand motion from a five-stage text task. ARDY produces a 25 Hz, five-segment motion with lengths `[100,100,50,50,106]` and transition overlap.
2. **Unified Adapter.** Producer output is converted to canonical 50 Hz, SONIC joint order and world-frame quaternion convention. Wrist approach, task-space constraints, bounded IK, table-clearance and hand scheduling are applied only through recorded, bounded policies. Raw, adapted and executed trajectories remain separate artifacts.
3. **Reference validation.** The canonical schema requires finite arrays, 29 body joints, 14 hand joints, `wxyz` unit quaternions, 0.02 s spacing, zero initial time, explicit case identity and zero initial hand reference.
4. **Native execution.** The original SONIC executable and original `TabletopDDS.step` run in an isolated DDS namespace. MuJoCo physics runs at 200 Hz; SONIC control decisions run at 50 Hz; command publication runs at 500 Hz; hand PD is applied independently at every physics step.
5. **Expert qualification.** The factory applies physical gates (lift, hold, contacts, no fall), native trace completeness, renderer identity, exact encoder/decoder reconstruction, command causality and state replay checks. Failed traces are retained and marked rejected.
6. **Canonical compilation.** Accepted traces are joined by explicit observation/reference/token IDs. RGB is rendered from exact integration states. Current observation history and future body/hand targets are exported with masks and provenance.
7. **VA training.** The V4.1 policy is initialized from a V4 checkpoint and trained on 7,530 training samples for 50 epochs using AdamW and the auxiliary objective in Section 7.
8. **Inference and evaluation.** A current RGB/state observation triggers a 10-body-token/50-hand-reference plan. `PlanBuffer` indexes each target by absolute tick, SONIC consumes the body token, hand PD consumes the paired hand reference, and MuJoCo advances physics. Offline metrics, scheduler integrity and physical success are reported separately.

## 3. Kimodo and ARDY interfaces

### 3.1 Five-stage task interface

The accepted default task is a standing G1 reaches over a tabletop, grasps a colored cube with the designated hand, lifts it, and holds it. The maintained task representation contains five semantically ordered stages: raise/stand preparation, reach and lower, grasp/closure, lift, and hold. Exact text differs by producer and is stored in each generation manifest; the default Kimodo task records the standing/reach/grasp/lift/hold prompt and three stage prompt strings for grasp, lift and hold.

Case geometry is explicit rather than inferred. The canonical legacy case uses cube center `(x,y)=(0.40,-0.15)` m, yaw 0°, and initial cube height 0.785 m. Seeds are recorded in generation metadata and are part of the raw-motion provenance. Additional near, left, far and right cases are preserved in the Unified Adapter study; right and far do not have accepted samples in the current production evidence.

Kimodo natively samples at 30 Hz with frame counts `[120,120,60,60,127]` (487 frames, about 16.2 s). ARDY natively samples at 25 Hz with `[100,100,50,50,106]` (406 frames, about 16.2 s) and uses transition history/overlap between segments. Both are resampled to an authored 50 Hz reference before native replay. The native frame rate and stage boundaries are never silently treated as the canonical control clock.

### 3.2 Unified Adapter responsibilities

The shared adapter performs the following deterministic operations, with per-frame corrections recorded in `adaptation_perframe.json` or equivalent metadata:

- map producer body and hand coordinates into the protected 29-body/14-hand SONIC order;
- resample positions at 50 Hz using interpolation and rotations using quaternion SLERP;
- preserve the declared world-frame root pose and `wxyz` convention;
- plan wrist approach to a task-space grasp corridor and apply bounded position/orientation corrections;
- enforce end-effector pose constraints and perform bounded offline IK;
- account for table clearance and collision-aware hand projection when the selected policy enables it;
- apply optional tracking calibration/compensation only when its evidence and hashes are present;
- keep body and hand references on the same authored timeline while allowing the native hand cursor to be logged independently;
- generate hand preshape and closure references for the independent hand PD path; and
- retain producer-specific parameters (native frame rates, stage lengths, diffusion/CFG controls, prompt text, seeds and spatial-conditioning flags) in provenance instead of hiding them behind a single generic interface.

The adapter does not claim that raw generated motion equals executed motion. Raw motion is the generator output; adapted reference is the canonical 50 Hz target after mapping/IK/corrections; executed motion is the native MuJoCo state after SONIC controller dynamics, asynchronous DDS delivery, body PD and hand PD. The Unified Adapter study measures substantial compression and change between these layers, which is why generator diversity cannot be inferred from the final dataset alone.

### 3.3 Evidence and limitations

The Unified Adapter comparison reports 24 strict accepted canonical episodes in one later v8 union (Kimodo 13, ARDY 11), with 3,648 complete V4 samples; a matched 5+5 spatial audit accepted 2/5 native-complete non-center trials. These artifacts are context for adapter behavior and are not the V4.1 training provenance. They show that the adapter can compress body diversity and that hand profiles and right-side geometry can create self/table contacts. They do not establish that one generator is statistically superior or that spatial generalization is solved.

## 4. Native execution and physical qualification

### 4.1 Native clocks and state semantics

Let `k` denote a 50 Hz control tick, `p` a 200 Hz physics step, and `r` a 500 Hz command-publication event. `LowState.reserve[0]` tags the exact published physical snapshot selected by the original native controller. The snapshot used as an observation is **pre-action**: it is the state gathered before the current decision's command is applied. The simulator may consume a published command after zero through several physics steps because DDS publication, receipt and the MuJoCo mutex are asynchronous.

SONIC's frozen encoder consumes the native history and current state to produce a 64-D post-FSQ token. The decoder consumes that token and its native history to produce 29 body actions. The current action and previous action are logged distinctly. The independent hand cursor selects the hand reference at the native execution time and applies a PD command at every 200 Hz physics step. RGB is sampled at 10 Hz in the canonical runtime and is rendered offline from the exact integration state associated with the selected observation. The camera is `head_rgb`, 320x240, with the recorded pose and FOV in `closed_loop/provenance.json`.

### 4.2 Physical and dataset gates

The physical evaluator requires minimum lift `0.10 m`, continuous hold `2.0 s`, positive normal-force contact from the designated thumb and index/middle fingers, no cube/table contact during the valid hold, no fall, and a complete finite synchronized native rollout; robot-table and self-contact counts are retained as diagnostics, and accepted expert episodes had zero forbidden contacts. Dataset validity additionally requires canonical reference/schema validity, complete frame coverage, finite arrays, explicit IDs, RGB/state/token/reference joins, source hashes, renderer identity, and successful forensic checks. `physical_success` and `expert_valid` are retained as separate fields: a trace can have a physical result while still failing provenance or completeness.

The Canonical Data Factory is fail-closed. It validates the source before native execution, refuses software rendering in production, runs or reuses only immutable native traces, checks RGB identity and pixel-perfect replay, reconstructs encoder/decoder outputs from frozen ONNX, verifies command publication/receipt/consumption causality, replays original `TabletopDDS.step`, and only exports training data when all required gates pass. Failed traces and reasons are preserved.

## 5. Canonical Data Factory and supervision construction

The architecture is shown in [Figure 3](figures/data_factory.svg), and the tick join is shown in [Figure 4](figures/temporal_alignment.svg).

### 5.1 Canonical episode

A canonical episode retains source reference arrays, native observations, RGB frames, SONIC encoder inputs and post-FSQ tokens, decoder inputs/outputs, commands and receipts, hand cursor/reference rows, physics integration states, contact records, provenance hashes and validation reports. Typical accepted traces contain 810 source/reference frames, 816 retained observation rows including startup context, 816 encoder/token rows, and 161 packed sample rows plus 162 rendered RGB frames in the retained trace (10 Hz over the active interval). The packed canonical sample arrays use RGB `[N,240,320,3]` uint8, history mask `[N,3]`, body target `[N,5,64]`, hand target `[N,5,14]`, and explicit mapping JSON.

### 5.2 ID-based alignment

For an anchor execution index `t`, the compiler selects history observations `[t-2,t-1,t]` at 50 Hz, with startup mask `[false,false,true]` when earlier rows do not exist. Future targets are the same decision/reference identity at offsets `t+2,...,t+6` (40–120 ms at 50 Hz). Body token IDs and hand reference IDs must resolve to the same future reference IDs and control ticks. The V4 loader then extends the verified prefixes to body horizon 10 and hand horizon 50; it drops only incomplete tails.

The join deliberately distinguishes:

- generated reference timestamp: authored producer time before adaptation;
- canonical reference timestamp: 50 Hz target time after adaptation;
- native control tick: the control decision index selected by SONIC;
- pre-action observation: actual state used by the native encoder before command application;
- post-step MuJoCo observation: state after integration, retained for replay and diagnostics;
- SONIC token execution time: the physics/control tick at which a token is actually consumed;
- RGB sampling time: the selected observation state's simulation time, rendered from exact integration state; and
- hand reference execution time: the hand cursor time used by the independent PD controller.

There is no nearest-timestamp guess, row-number join, silent interpolation of RGB, or fabricated future label. Missing IDs, noncontiguous target IDs, mismatched observation/reference identities, nonfinite values, and inconsistent clocks reject the sample or episode.

### 5.3 V4 tensor construction

For current body input, the loader concatenates body position `q∈R^{29}`, body velocity `dq∈R^{29}`, root quaternion `q_root∈R^4` in `wxyz`, local angular velocity `ω∈R^3`, and projected gravity `g∈R^3`, yielding `x_body∈R^{68}`. Current hand input concatenates 14 joint positions and 14 velocities, yielding `x_hand∈R^{28}`. RGB is converted to float `[3,128,128]` from the recorded 320x240 RGB image. The targets are `Y_body∈R^{10×64}`, `Y_hand∈R^{50×14}`.

Closure progress is derived from the current canonical hand reference, not a human annotation. The loader computes the initial-to-peak displacement of moving right-hand joints (absolute displacement >0.05 rad), projects the current reference onto that direction, and clips the scalar to `[0,1]`. Phase labels are `OPEN` for progress ≤0.05, `CLOSING` for intermediate values, and `HOLD` for progress ≥0.95. These labels describe the reference schedule; they are not independent physical contact labels.

## 6. Authoritative V4.1 training dataset

The dataset architecture and composition are shown in [Figure 5](figures/dataset_structure.svg), with reusable values in [`tables/dataset_statistics.csv`](tables/dataset_statistics.csv).

`expert_datasets/train_dataset_v0/manifest.json` records a V4 tensor export with 9,488 samples and 63 episodes: 19 Kimodo episodes (each 152 samples, 810 frames, 16.18 s) and 44 rule-based episodes (each 150 samples, 800 frames, 15.98 s). The explicit seed-4 split contains 50 training episodes / 7,530 samples and 13 validation episodes / 1,958 samples. The dataset is not a 20-episode or 21-episode dataset, and no frame-level random split is used.

The exact tensor contract is:

| tensor | shape | dtype | meaning |
|---|---:|---|---|
| `rgb` | `[9488,3,128,128]` | float32 | current RGB observation |
| `body` | `[9488,68]` | float32 | current body state |
| `hand` | `[9488,28]` | float32 | current hand state |
| `body_tokens` | `[9488,10,64]` | float32 | future SONIC post-FSQ body targets |
| `hand_ref` | `[9488,50,14]` | float32 | future absolute hand references |
| `phase` | `[9488]` | int64 | OPEN/CLOSING/HOLD reference label |
| `progress` | `[9488]` | float32 | clipped closure progress |

The manifest is exact-serialization (`baseline_tensor_serialization_exact=true`) and declares no normalization was applied at export. The trainer computes body/hand means and standard deviations from the training samples only and clamps standard deviations to a minimum of 0.05. Dataset hashes are: manifest `cc507f87571703906c3c6e8eb38a6e9dc8daee51109a64c014d51645786ffa2c`, samples `b7ffb8437b913369d0202f3f024376182c764ebb0b445fb1e98e1db32a623fd9`, split `263761da8e7afb0b13a0cf2bf1e1f2d9f2bce33d9cf0060d582cb15ae350f4a9`, index `3c8bad5e4bef8133112c062abaec0dd8f1894b127995ec6f3429e07702d9f13c`.

The combined provenance names the Kimodo and rule-based source manifests and records exact episode-ID deduplication. The training set is compositionally imbalanced toward rule-based references (44/63 episodes); the split is episode-disjoint but the offline evaluation file also reports `validation_overlap=true` because its aggregate evaluation list contains both train and validation episodes. Consequently, the aggregate offline metrics are not an independent generalization test. The per-epoch validation values in `history.json` are the held-out validation measurements.

## 7. Final V4.1 model architecture and objective

The exact network is documented in [VA_FINAL_MODEL_ARCHITECTURE.md](VA_FINAL_MODEL_ARCHITECTURE.md) and Figures 6–9. The forward pass is:

1. `rgb [B,3,128,128]` passes through four `Conv2d(k=3,s=2,p=1) → GroupNorm(8,C) → GELU` blocks with channels `3→32→64→128→256`, producing `[B,256,8,8]`.
2. Flattening yields 64 visual tokens `[B,64,256]`; learned position `[1,64,256]` is added.
3. Body and grasp query MLPs map normalized body input `[B,68]` to two independent `[B,1,256]` queries.
4. Each query attends with a four-head `MultiheadAttention(embed_dim=256)` block over visual tokens. Query/value LayerNorm, residual attention and residual `256→512→256` feed-forward blocks yield body context `c_b` and grasp context `c_g`, each `[B,256]`.
5. The body pathway maps `c_b` through `256→512→512`, then `512→512→640`, and reshapes to `[B,10,64]`.
6. The grasp pathway maps `c_g` through `256→64` and produces three phase logits and one sigmoid progress scalar.
7. The hand state MLP maps `[B,28]→[B,128]`. Concatenating `c_g [256]`, phase feature `[64]` and hand feature `[128]` produces `[B,448]`; the hand head maps `448→512→512→700` and reshapes to `[B,50,14]`.
8. V4.1 computes a second grasp context from body query and visual tokens and applies `Linear(256,1)` to produce `grasp_ready_logit [B,1]`. Hand q/dq are excluded from this readiness head.

There is no temporal transformer over observations, recurrent state, diffusion sampler, DiT, feedback pathway or additional VA learned decoder; the protected SONIC decoder remains in the native loop. Temporal structure is represented by the output horizons and the absolute-tick scheduler.

The base objective for sample `i` is
\[
L_\mathrm{base}=L_\mathrm{smoothL1}(\hat Y_b,Y_b)+L_\mathrm{MSE}(\hat Y_h,Y_h)+0.1L_\mathrm{CE}(\hat p,p)+0.1L_\mathrm{MSE}(\hat u,u).
\]
V4.1 adds readiness BCE, hand-velocity smooth-L1, and a reopening penalty:
\[
L=L_\mathrm{base}+0.1L_\mathrm{BCE}(\hat r,\mathbb{1}[p\ge1])+0.1L_\mathrm{smoothL1}(\Delta\hat Y_h,\Delta Y_h)+0.02\,\mathrm{mean}\,\max(0,-\Delta\hat Y_h\,\mathbb{1}[p\ge1]).
\]
The readiness, velocity and reopening terms are training regularizers. The native rollout consumes body tokens and hand references; auxiliary logits are not used to alter deployment outputs.

## 8. Training protocol and development history

V4.1 uses seed 4 for PyTorch and MUSA, AdamW with learning rate `3×10^{-4}`, batch size 32, weight decay 0, 50 epochs, gradient-norm clipping at 10, no learning-rate schedule, and initialization from matching tensors in the prior V4 checkpoint. Body/hand normalization is fitted from the 7,530 training samples only. The checkpoint is selected at the completed 50-epoch run; its status records 3,502,977 trainable parameters.

The recorded validation history is sparse (epochs 1, 10, 20, 30, 40 and 50). Final epoch 50 values are validation loss 0.00290098, hand MAE 0.00264212, phase accuracy 0.99591422, and trigger rate 0.46892756. The lowest recorded validation loss is also at epoch 50 among the retained checkpoints. No fabricated per-step curve is supplied; Figure 12 plots only these recorded points.

The development sequence is:

- V1/V2/V3 established current-observation interfaces, native token labels and timing diagnostics. V2 used named history fields and one-token hand/body outputs; V3 used 96-D current state with two-slot horizons. Their documented live trials failed task success or timing contracts.
- V4 introduced current RGB + 68-D body + 28-D hand, visual cross-attention, 10×64 body output, 50×14 hand output, phase and progress heads, and a 50-train-episode/13-validation-episode split.
- V4.0 BCZ-hand explored close-intent and future-progress consistency. Those heads and losses are historical experiments, not the final checkpoint.
- V4.1 added `grasp_ready_head`, hand-velocity smooth-l1 and reopening regularization. The final checkpoint's `v41=true` and state-dict key `grasp_ready_head.*` verify that this is the deployed architecture for the reported rollout.
- V4 ablations (`AUX-1`, `AUX-2`, `AUX-3`, pooled MLP fusion and proprio-only) share the same dataset but are 20-epoch research variants. Their offline differences are not controlled causal evidence for deployment success; no physical learned-policy ablation passed the grasp criterion.

## 9. Inference and action scheduling

The learned rollout uses current RGB, current 68-D body and current 28-D hand. The renderer worker verifies the MTT S4000 hardware renderer before inference. A plan is requested every five absolute control ticks. The policy predicts 10 body-token slots and 50 hand-reference slots. `PlanBuffer.add` requires shapes `[10,64]` and `[50,14]`, finite values, and starts separated by exactly five ticks. It stores each slot under its intended absolute tick. `PlanBuffer.take` requires a contiguous tick sequence; it raises on uncovered, repeated or skipped ticks and records overlap fallbacks and deadline misses.

The closed-loop sequence is shown in [Figure 10](figures/closed_loop_inference.svg). An external-token message carries the 64-D body token and authored frame identity to the native SONIC process. Native SONIC owns decoder history and emits body motor commands through the original DDS path. The paired hand target is applied by the unchanged independent hand PD controller. MuJoCo integrates at 200 Hz, records contacts and states, and the next observation triggers receding-horizon replanning.

The reported rollout used authored lead ticks 2 and startup anchor -2. It made 162 plans, recorded 810/810 ticks, one overlap fallback and one deadline miss, with zero uncovered, repeated, skipped or stale control ticks. This is a scheduler-integrity result; it does not make the physical task successful.

## 10. Experimental results

### 10.1 Offline prediction

The aggregate `offline_metrics.json` reports token MAE 0.0041827890, token MSE 5.0398827×10⁻⁵, FSQ scalar accuracy 0.9891404, whole-token exact accuracy 0.5590746, hand MAE 0.0029544812 rad, phase accuracy 0.99304384, and progress MAE 0.0043467851. The confusion matrix is
\[
\begin{bmatrix}5051&6&1\\2&1008&56\\0&1&3363\end{bmatrix}.
\]
The ten body-horizon MAEs and fifty hand-horizon MAEs are retained in [`offline_metrics.json`](../../outputs/va_v4_1_kimodo_rule_20261009/offline_metrics.json); the scalar summary is exported to [`tables/offline_evaluation.csv`](tables/offline_evaluation.csv). Because `validation_overlap=true`, these aggregate values combine train and validation episodes and must not be interpreted as held-out generalization.

### 10.2 Native learned-policy rollout

The final learned-policy rollout is one complete 810-tick native run. Runtime integrity passed: `rollout_complete=true`, `strict_runtime_pass=true`, expected/recorded ticks 810/810, plans 162, uncovered/repeated/skipped ticks 0, and hardware renderer verified. Task success failed: physical grasp false, maximum lift 0.0198275 m, final lift −0.0001078 m, hold 0 s, cube-table contact remained true, 107 robot-table contact steps, 640 self-contact steps, and no fall. The first measured divergence from the expert comparison was body RMS 0.1240 rad at tick 5 (0.1 s), right-wrist position 0.0674 m at tick 5, hand RMS 0.1085 rad at tick 15, hand-reference MAE 0.1600 rad at tick 20, and cube position divergence 0.0545 m at tick 470 (9.4 s). These are diagnostics, not claims of causality.

### 10.3 Expert success versus learned success

Accepted expert Kimodo/ARDY/rule-based traces can pass the physical lift/hold gate and become training demonstrations. They are deterministic reference replays; the expert future token and hand reference are injected by the replay path. The final learned-policy rollout has `expert_future_injected=false`; its failure therefore cannot be hidden by a successful expert replay. Offline error, expert replay success, and learned closed-loop task success are three different measurements and are reported in separate tables.

## 11. Limitations and reproducibility

The dataset is concentrated in one tabletop task, one primary right-hand case for many episodes, and a mixed producer composition. Adapter corrections compress raw motion diversity; the 63-episode V4.1 dataset is older than the later Unified Adapter production set; no independent test split is present in the aggregate offline evaluator; and the phase/progress labels are derived from reference geometry rather than contact sensing. The one complete learned rollout failed physical grasp, so no claim of learned physical success or broad spatial generalization is justified.

Reproduction uses the existing `va-train` container and the prescribed MUSA/EGL/DDS setup. Training and offline evaluation need no DDS; native replay requires an isolated DDS namespace and the verified MTT S4000 rendering profile. Do not install or replace Torch/MUSA, rebuild SONIC, alter `TabletopDDS.step`, or substitute the newer dataset. The exact checkpoint, dataset hashes, source hashes, runtime binary hashes and controller provenance are retained in the output directory.

## 12. Artifact index

- Main report: `VA_FINAL_TECHNICAL_REPORT.md`.
- Model architecture/objective: `VA_FINAL_MODEL_ARCHITECTURE.md`.
- Data Factory and alignment: `VA_DATA_FACTORY_TECHNICAL_REFERENCE.md`.
- Training/evaluation protocol: `VA_TRAINING_AND_EVALUATION.md`.
- Auxiliary supervision: `VA_AUXILIARY_TASK_ANALYSIS.md`.
- Results and diagnostics: `VA_EXPERIMENT_RESULTS.md`.
- Development chronology: `VA_DEVELOPMENT_HISTORY.md`.
- Evidence and hashes: `REPORT_EVIDENCE_INDEX.md`.
- Figures and captions: `FIGURE_INDEX.md`.
- Completion audit: `REPORT_COMPLETION_CHECKLIST.md`.
