# VA Training and Evaluation Protocol

## Reproducible configuration

The final V4.1 run is recorded by [`outputs/va_v4_1_kimodo_rule_20261009/training_status.json`](../../outputs/va_v4_1_kimodo_rule_20261009/training_status.json), `history.json`, and the checkpoint metadata. It trained 50 epochs with 7,530 training samples and 1,958 validation samples. The exact custom run used seed 4 (PyTorch and MUSA), AdamW, learning rate (3\times10^{-4}), batch size 32, weight decay 0, gradient clipping at norm 10, no learning-rate schedule, and matching-weight initialization from the V4 prior. Loss weights were body 1, hand 1, phase 0.1, progress 0.1, readiness trigger 0.1, hand velocity 0.1 and reopening 0.02.

The standard V4 JSON configuration in `va_research/configs/token_v4.json` describes a longer 4,000-epoch baseline; it is not the V4.1 custom 50-epoch schedule. This distinction prevents a config file from being mistaken for the recorded final run.

## Split and normalization

The episode-level seed-4 split has 50 training episodes and 13 validation episodes. Body and hand statistics are fitted only on training rows, with standard deviation clamp 0.05, and copied into checkpoint buffers. Validation does not update statistics. The dataset export itself is unnormalized. The aggregate offline evaluator includes both train and validation episode IDs and sets `validation_overlap=true`, so its metrics are useful for regression and gate checking, not an independent test estimate.

## Recorded validation history

`history.json` records only epochs 1, 10, 20, 30, 40 and 50. The complete values are in `history.json` and Figure 12. At epoch 50: validation loss 0.0029009753, hand MAE 0.0026421151, phase accuracy 0.9959142208 and readiness trigger rate 0.4689275622. The file contains no per-step history; no interpolated or fabricated curve is reported.

## Offline metrics and gate

The aggregate evaluator reports token MAE 0.0041827890, token MSE 5.0398827e-05, FSQ scalar accuracy 0.9891403913, whole-token exact accuracy 0.5590746403, hand MAE 0.0029544812 rad, moving-right-hand MAE 0.0029544812 rad, phase accuracy 0.9930438399 and progress MAE 0.0043467851. The configured offline thresholds in `va_research/configs/token_v4.json` are token MAE ≤0.01, FSQ scalar accuracy ≥0.95, hand MAE ≤0.02, moving-right MAE ≤0.03, phase accuracy ≥0.98 and progress MAE ≤0.03. The recorded `offline_gate` is true. These thresholds are offline acceptance checks and do not predict physical grasp success.

## Native learned-policy evaluation

The native run uses `outputs/va_v4_1_kimodo_rule_20261009/closed_loop/report.json`, `outputs/va_v4_1_kimodo_rule_20261009/closed_loop/provenance.json`, `outputs/va_v4_1_kimodo_rule_20261009/closed_loop/inference_timing.json`, `outputs/va_v4_1_kimodo_rule_20261009/closed_loop/control_timeline.json`, `outputs/va_v4_1_kimodo_rule_20261009/closed_loop/physical_metrics.json` and `outputs/va_v4_1_kimodo_rule_20261009/closed_loop/renderer_worker.json`. The hardware renderer was verified as Moore Threads / MTT S4000. Native execution was asynchronous, with original SONIC binary/controller and independent hand PD. The run completed 810/810 ticks and 162 plans. Inference time averaged 6.336 ms (median 6.195 ms, maximum 17.894 ms); rendering averaged 6.275 ms (median 6.212 ms, maximum 8.444 ms). There was one overlap fallback/deadline miss and no uncovered/repeated/skipped tick.

The physical criterion was not met: maximum lift 0.0198275 m (<0.10 m), final lift −0.0001078 m, maximum and final hold 0 s, 107 robot-table contact steps, 640 self-contact steps, no fall, and the cube remained in table contact. The first divergence diagnostics are preserved in `outputs/va_v4_1_kimodo_rule_20261009/video/failure_localization.json`; they should be treated as localization evidence, not a causal decomposition.

## Expert and oracle controls

Expert native replays and deterministic canonical production are controls for the data path. For example, `outputs/canonical_production/production_summary.json` records a Kimodo expert with 0.120292 m lift and 4.575 s hold and an ARDY expert with 0.137346 m lift and 4.84 s hold, with physical, dataset and expert gates true. These are not learned-policy trials. An expert-future-injected replay cannot validate VA closed loop; the V4.1 rollout explicitly records `expert_future_injected=false`.

## Test matrix and limits

The ablation artifacts all use the same dataset hash and split hash, seed 4, batch 32 and learning rate 3e-4. They pass offline gates but have no controlled physical success. A pooled-MLP fusion and proprio-only model are architecture probes; no causal conclusion should be drawn from their single-run offline values. The final evaluation hierarchy is therefore: source/schema validity → canonical trace/physical expert validity → offline regression gate → absolute-tick runtime integrity → learned physical task success. V4.1 passes the first four only when “physical expert validity” refers to demonstrations; it fails the final learned physical task gate.
