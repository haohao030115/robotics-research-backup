# Results summary

These results are evidence-qualified; they are not a replacement for the
[technical report](TECHNICAL_REPORT.md).

| Result | Evidence | Classification |
|---|---|---|
| Kimodo/ARDY fixed-center comparison | Historical comparison summary referenced by the architecture audits | **REPORTED**; complete primary 20-seed bundle not present in the representative persistent paths |
| Rule-based references | `expert_datasets/rule_reference_right100_v3/manifest.json` and per-episode provenance | **VERIFIED** reference catalog and physical metrics |
| Canonical Kimodo merge | `outputs/canonical_va/kimodo_19_merged/training_manifest.json` | **VERIFIED** 19 episodes, 3,059 samples, one explicit exclusion |
| V4 training | `outputs/grasp_pipeline_kimodo_rule_combined_v4_20261009/training_run/` | **VERIFIED** checkpoint and passing training gate |
| V4 offline evaluation | `.../training_run/offline_metrics.json` and fresh `va-train` replay | **VERIFIED** offline gate true |
| V4 closed loop | `.../closed_eval_attempt_0003/report.json` | **VERIFIED** 810/810 runtime ticks and zero scheduler misses; physical grasp false |
| Kimodo fresh generation | `research/kimodo/` execution evidence in final audit | **VERIFIED** generation reached 487 frames; **BLOCKED** by existing IK gate |
| ARDY physical qualification | canonical replay and validation artifacts | **VERIFIED** runtime integrity; physical task result remains false for the representative run |

Read the full report for timing, model contracts, provenance chains, and
limitations.
