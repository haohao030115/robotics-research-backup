# Auxiliary Task Analysis for V4.1

## Label construction

The canonical loader derives closure progress from reference hand trajectories. Let \(q_0\) be the initial 14-D hand reference and \(q_\star\) the frame with maximum squared displacement over the episode. Define a mask over right-hand joints with \(|q_{\star,j}-q_{0,j}|>0.05\) rad and direction \(d=(q_\star-q_0)\odot m\). For current reference \(q_t\),

\[
u_t=\operatorname{clip}\left(\frac{(q_t-q_0)^\top d}{d^\top d},0,1\right).
\]

The integer phase label is OPEN when \(u_t\le0.05\), CLOSING when \(0.05<u_t<0.95\), and HOLD when \(u_t\ge0.95\). This is a deterministic reference-derived label. It is not a contact sensor, force threshold, or manually annotated physical phase.

## Heads and activations

`phase_feature` maps the grasp cross-attention context `[B,256]` to `[B,64]` with Linear+GELU. `phase_head` is Linear(64,3) and returns unnormalized logits; cross-entropy applies the class loss. `progress_head` is Linear(64,1) followed by sigmoid; the regression target is the clipped scalar \(u_t\). V4.1 adds `grasp_ready_head`, Linear(256,1), over a second grasp context. Its target is `phase >= 1`, so OPEN is negative and CLOSING/HOLD are positive. The readiness output is a logit and BCE-with-logits is used.

The hand velocity loss uses consecutive future hand references. A reopening regularizer applies a ReLU penalty to negative predicted local increments when the source phase is CLOSING or HOLD. The exact implementation is `va_research/models/v4_1/losses.py`.

## Complete objective

For body token target \(Y_b\in\mathbb{R}^{10\times64}\), hand target \(Y_h\in\mathbb{R}^{50\times14}\), phase \(c\), progress \(u\), and readiness \(r=\mathbb{1}[c\ge1]\),

\[
\begin{aligned}
L_b &= \operatorname{SmoothL1}(\hat Y_b,Y_b),\\
L_h &= \operatorname{MSE}(\hat Y_h,Y_h),\\
L_c &= \operatorname{CE}(\hat z,c),\\
L_u &= \operatorname{MSE}(\hat u,u),\\
L_r &= \operatorname{BCEWithLogits}(\hat r,r),\\
L_v &= \operatorname{SmoothL1}(\Delta\hat Y_h,\Delta Y_h),\\
L_{open} &= \operatorname{mean}\max(0,-\Delta\hat Y_h\,\mathbb{1}[c\ge1]),\\
L &= L_b+L_h+0.1L_c+0.1L_u+0.1L_r+0.1L_v+0.02L_{open}.
\end{aligned}
\]

The reopening mask is broadcast over future hand slots and coordinates. It is a local regularizer; it does not force the expert target to be monotonic globally, and it does not overwrite predictions at inference.

## Feature connections and deployment use

The phase and progress heads receive the 64-D phase feature derived from grasp context. The hand head receives grasp context, the same 64-D phase feature, and the normalized 128-D hand-state feature. Thus phase/progress are auxiliary outputs sharing a representation with hand prediction, while readiness is a separate scalar over grasp context. The body head receives the independent body cross-attention context and is not directly conditioned on the auxiliary logits.

The native rollout scheduler accepts only `body_tokens` and `hand_ref`. `phase_logits`, `closure_progress`, and `grasp_ready_logit` are recorded model outputs but do not gate or rewrite action chunks. Auxiliary tasks therefore regularize the shared representation during training and provide diagnostic signals; they are not deployment-time feedback controllers in this artifact.

## What the evidence supports

The full model's aggregate offline phase accuracy is 0.99304384 and progress MAE is 0.00434679. The final recorded validation phase accuracy is 0.99591422. These establish that the heads fit the derived labels. They do not establish better physical grasping. The `AUX-1`/`AUX-2`/`AUX-3` ablations show different offline values, but they differ in schedule and are single runs; the repository has no controlled multi-seed or physical ablation proving a causal benefit. The possible rationale is that phase transitions and closure progress expose a low-dimensional temporal structure that may help hand prediction, but this remains a research hypothesis.
