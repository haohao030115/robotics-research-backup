# Final V4.1 VA Model Architecture

## Artifact identity

The complete checkpoint is [`../../outputs/va_v4_1_kimodo_rule_20261009/checkpoint.pt`](../../outputs/va_v4_1_kimodo_rule_20261009/checkpoint.pt), SHA-256 `92de206b703d66cea50dfd8b92296778d2400d9969e3e964527a8bbd2328143f`. Its metadata has `v41=true`, `epoch=50`, and matching initialization from the historical V4 prior. The trainable parameter count is 3,502,977; the state dictionary includes 192 normalization-buffer values in addition to those parameters. The source of truth is [`va_research/models/v4/model.py`](../../va_research/models/v4/model.py) plus [`va_research/models/v4_1/model.py`](../../va_research/models/v4_1/model.py).

## Inputs and normalization

For batch size (B), the model consumes:

- RGB \(I \in \mathbb{R}^{B\times3\times128\times128}\), float32 after conversion from the recorded 320×240 RGB image;
- body state \(x_b \in \mathbb{R}^{B\times68}\), concatenating 29 joint positions, 29 joint velocities, root quaternion `(w,x,y,z)`, root angular velocity and projected gravity; and
- hand state \(x_h \in \mathbb{R}^{B\times28}\), concatenating 14 joint positions and 14 joint velocities.

The trainer stores `body_mean`, `body_std`, `hand_mean`, and `hand_std` buffers fitted only on the training rows. Each standard deviation is clamped to at least 0.05. RGB is not normalized by the model source; the dataset supplies float RGB in [0,1].

## Layer-by-layer tensor flow

| module | exact operation | output |
|---|---|---|
| `visual.0–2` | Conv2d 3→32, k3/s2/p1; GroupNorm(8,32); GELU | (B×32×64×64) |
| `visual.3–5` | Conv2d 32→64, k3/s2/p1; GroupNorm(8,64); GELU | (B×64×32×32) |
| `visual.6–8` | Conv2d 64→128, k3/s2/p1; GroupNorm(8,128); GELU | (B×128×16×16) |
| `visual.9–11` | Conv2d 128→256, k3/s2/p1; GroupNorm(8,256); GELU | (B×256×8×8) |
| flatten/position | flatten spatial dimensions, transpose, add learned `[1,64,256]` position | (B×64×256) |
| `body_query` | Linear 68→256, GELU, Linear 256→256, LayerNorm, GELU | (B×256), then query (B×1×256) |
| `grasp_query` | same as body query | (B×256), then query (B×1×256) |
| `body_cross` | q LayerNorm; v LayerNorm; 4-head MHA 256-D; residual; LayerNorm; MLP 256→512→256; residual | (B×256) |
| `grasp_cross` | same independent block | (B×256) |
| `body_feature` | MLP 256→512→512, GELU | (B×512) |
| `body_head` | MLP 512→512→640 | \(B\times640\to B\times10\times64\) |
| `phase_feature` | Linear 256→64, GELU | (B×64) |
| `phase_head` | Linear 64→3 | (B×3) logits |
| `progress_head` | Linear 64→1, sigmoid | (B) scalar |
| `hand_state` | MLP 28→128→128, GELU | (B×128) |
| `hand_head` | Linear 448→512, GELU; MLP 512→512→700 | \(B\times700\to B\times50\times14\) |
| V4.1 readiness | second grasp context (B×256) → Linear 256→1 | (B×1) logit |

The source defines `mlp(a,b,c)` as `Linear(a,b) → GELU → Linear(b,c)`. In `body_feature` and `hand_head`, the outer `Sequential` contributes the explicitly listed GELU after the first linear layer; the nested MLP contributes its own GELU between the final linear stages.

## Forward pass

The visual tensor is formed once. The normalized body state produces two distinct queries. Body cross-attention yields the context used only for the body token head. Grasp cross-attention yields the context shared by phase/progress and hand prediction. The normalized hand state enters only the hand branch. The V4.1 readiness head recomputes grasp cross-attention from the same body query and visual tokens, then applies a scalar linear classifier. This repeated attention computation is an implementation fact; it is not a recurrent or temporal feedback pathway.

The output dictionary is:

```text
body_tokens       [B,10,64]    continuous post-FSQ token coordinates
hand_ref          [B,50,14]    absolute hand joint references (radians)
phase_logits      [B,3]        OPEN, CLOSING, HOLD logits
closure_progress  [B]          sigmoid scalar in [0,1]
grasp_ready_logit [B,1]        readiness logit, V4.1 only
```

At deployment the scheduler consumes only `body_tokens` and `hand_ref`. Phase, progress, and readiness remain diagnostics and training regularizers; they do not modify the body token or hand-reference outputs.

## Parameter and representation notes

The 64 body coordinates represent SONIC's release post-FSQ token space. Offline evaluation computes scalar accuracy after rounding/clamping to the release quantization range and also reports exact equality of all 64 coordinates. The model regresses the continuous coordinates; it does not contain a discrete softmax over token values. Hand references are absolute joint angles and are applied by the existing independent PD controller.

There is no observation-history encoder, language encoder, temporal self-attention, action feedback input, diffusion sampler, or additional VA learned decoder in V4.1; the protected SONIC decoder remains outside the checkpoint. The ten-token and fifty-reference dimensions are output horizons, not additional model time layers. The native SONIC decoder, history buffer, body action scaling, motor permutation and DDS transport remain outside the checkpoint.

## Architectural rationale versus evidence

The following are design interpretations, not measured causal findings: the visual spatial tokens retain object and hand geometry; separate body/grasp queries allow body and hand pathways to attend to different image regions; cross-attention is expected to be more selective than pooled MLP fusion; and phase/progress/readiness supervision may organize the grasp transition. The repository contains no controlled physical ablation proving these mechanisms improve task success. The ablation CSVs therefore report observations only.
