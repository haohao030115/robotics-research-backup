# VA Development History and Version Evolution

## V1–V3 native-token baselines

The V1/V2/V3 families established the native token interface and exposed synchronization risks. V2 used named state histories, one post-FSQ body token and one absolute hand reference, with explicit previous-action fields. Its documented held-out offline gate passed in a baseline trial, but live closed loop missed next-control deadlines and produced zero lift/hold. V3 used current RGB plus a 96-D state, a two-slot body/hand horizon, and strict absolute tick checks; its strict test aborted after stale-token detection, while the diagnostic video had 399 stale ticks and failed grasp. These results motivated chunked V4 predictions and a formal absolute-tick buffer.

## V4 architecture

V4 introduced the current RGB+proprioception architecture: 68-D body input, 28-D hand input, 64 visual tokens, two 4-head cross-attention queries, 10×64 body targets, 50×14 hand targets, and phase/progress heads. The base objective used smooth-L1 body, MSE hand, phase cross-entropy and progress MSE. The V4 source remains the structural base of V4.1.

## BCZ-hand and one-shot experiments

V4.0 BCZ-hand added a close-intent head and optional future-progress consistency. The one-shot runtime added OPEN/CLOSING/HOLD state-machine logic, two confirmations, velocity limits, and trigger thresholds. Those mechanisms are retained as historical experiments. They are not present as a deployment-time controller in the final V4.1 rollout, which accepts the model's body/hand chunks through `PlanBuffer`.

## V4.1 final training artifact

V4.1 adds `grasp_ready_head=Linear(256,1)` from grasp context, plus readiness BCE, hand-velocity smooth-L1 and reopening regularization. The final run uses the `expert_datasets/train_dataset_v0` tensor export, seed 4, AdamW 3e-4, batch 32, weight decay 0, gradient clipping 10 and 50 epochs. Checkpoint metadata and state-dict keys verify the architecture; `training_status.json` records the completion and parameter count.

## Ablation studies

The retained ablations use the same dataset and split: no phase, no progress, no phase+progress, pooled MLP visual fusion and proprio-only. They were 20-epoch probes with learning rate 3e-4 and seed 4. Each passed its recorded offline gate, but only AUX-1 has partial native trial evidence and it also failed physical grasp. No ablation establishes a controlled causal gain or a physical success improvement.

## Historical V4 versus V4.1 naming hazard

The historical V4 prior is referenced by the final checkpoint as an initialization source and by compatibility copies used for native rollout. It must not be mislabeled as the final successful model: the full checkpoint contains `v41=true` and `grasp_ready_head.*`. Conversely, the runtime-compatible `rollout_checkpoint.pt` is a stripped V4-compatible copy used to satisfy native loading; the complete training artifact remains `checkpoint.pt`. Both identities and hashes are recorded in `REPORT_EVIDENCE_INDEX.md`.
