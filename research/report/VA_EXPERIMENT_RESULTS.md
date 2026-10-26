# VA Experiment Results and Failure Diagnostics

## Result taxonomy

This report separates (1) deterministic expert replay, (2) offline supervised prediction, (3) oracle/reference replay, (4) learned native closed loop, and (5) scheduler integrity. A result in one category is never promoted to another category.

## Final V4.1 result table

| category | evidence | result | interpretation |
|---|---|---|---|
| checkpoint | `outputs/va_v4_1_kimodo_rule_20261009/checkpoint.pt` | SHA `92de206b…8143f`, `v41=true`, 50 epochs | final trained artifact |
| dataset | `expert_datasets/train_dataset_v0/manifest.json` | 9,488 samples, 63 episodes, 7,530/1,958 split | training source |
| offline | `offline_metrics.json` | token MAE 0.00418279; FSQ 0.98914; hand MAE 0.00295448; phase 0.99304; progress 0.00434679; gate true | regression gate passed; overlap flagged |
| native scheduler | `closed_loop/report.json` | 810/810 ticks, 162 plans, 0 uncovered/repeated/skipped, 1 deadline/fallback | runtime integrity passed |
| learned task | same report | grasp false; max lift 0.0198275 m; hold 0 s | physical learned-policy failure |
| expert control | `outputs/canonical_production/production_summary.json` | Kimodo 0.120292 m/4.575 s; ARDY 0.137346 m/4.84 s | deterministic reference/data path can pass |

## Offline predictions

Full horizon arrays are in `offline_metrics.json`; the reusable scalar summary is in `tables/offline_evaluation.csv`. The body MAE stays near 0.004 across ten slots; hand MAE is reported for all 50 slots. The confusion matrix is `[[5051,6,1],[2,1008,56],[0,1,3363]]`. Since `validation_overlap=true`, the aggregate includes training episodes; the held-out validation curve is in `history.json` and Figure 12.

## Native timing and scheduling

The rollout records 162 inference requests, anchors from −2 through 803 and starts from 0 through 805. Inference mean/median/max are 6.336/6.195/17.894 ms; rendering mean/median/max are 6.275/6.212/8.444 ms. At actual tick 390, plan generation 77 used an overlap fallback, counted as one deadline miss. All 810 ticks still had a valid plan, so strict runtime passed. This distinction matters: scheduler coverage is a necessary integration property, not a task metric.

## Physical failure diagnostics

The report records no fall and no cube attachment/teleport. Contact evidence shows 107 robot-table contact steps involving right index, middle and thumb links and 640 self-contact steps involving thumb/wrist/finger pairs. The cube remained in table contact, and opposing contact never met the success criterion. The failure-localization artifact finds first divergence at body RMS 0.1240 rad and wrist position 0.0674 m at tick 5, hand RMS 0.1085 rad at tick 15, hand-reference MAE 0.1600 rad at tick 20, and cube position divergence 0.0545 m at tick 470. The phase comparison has no mismatch at the exact canonical anchors sampled by the video tool. These observations localize failure onset but do not prove whether image mismatch, body-token error, hand scheduling, controller dynamics, or contact geometry is the dominant cause.

## Ablations

`tables/model_ablation.csv` records the same-dataset offline probes. AUX-1 removes phase loss; AUX-2 removes progress loss; AUX-3 removes both; ARCH-1 replaces cross-attention by pooled MLP fusion; ARCH-2 removes RGB. All are 20-epoch single runs from the same prior and have no verified physical success. They are useful for future hypothesis testing, not claims of architectural superiority.

## Videos

`outputs/va_v4_1_kimodo_rule_20261009/video/expert_vs_v4.mp4` is a 161-frame, 10 FPS comparison with exact canonical sample IDs and actual learner ticks; it is not padded. `video/video_metadata.json` records the alignment rule and `failure_localization.json` records the first threshold crossings. The clip is a diagnostic artifact, not an additional success trial.
