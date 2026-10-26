# Figure Index and LaTeX Placement Guide

All figures are standalone SVG files with English labels, vector geometry, source comments and publication-oriented captions. The companion provenance map is [`FIGURE_SOURCE_PROVENANCE.json`](FIGURE_SOURCE_PROVENANCE.json). Numerical values remain in CSV/JSON sources rather than only in artwork.

| no. | file | content, source data and interpretation | intended paper placement |
|---:|---|---|---|
| 1 | [`figures/system_architecture.svg`](figures/system_architecture.svg) | Complete generator → adapter → native SONIC/MuJoCo → factory → VA → closed-loop architecture. Sources: `data_factory/runtime`, V4.1 model and native provenance. Shows learned versus deterministic boundaries. | System overview |
| 2 | [`figures/generation_to_execution.svg`](figures/generation_to_execution.svg) | Kimodo/ARDY/rule references, five-stage generation, Unified Adapter, canonical 50 Hz reference, SONIC execution and expert qualification. Sources: shared adapter and pipeline docs. | Motion-generation and execution method |
| 3 | [`figures/data_factory.svg`](figures/data_factory.svg) | Fail-closed Canonical Data Factory stages and acceptance gates. Sources: runtime pipeline, native forensic and dataset modules. | Data-generation method |
| 4 | [`figures/temporal_alignment.svg`](figures/temporal_alignment.svg) | 50 Hz control, 200 Hz physics, 500 Hz publication, 10 Hz RGB, pre-action observation and future target offsets. Sources: native provenance and PlanBuffer. | Timing/alignment subsection |
| 5 | [`figures/dataset_structure.svg`](figures/dataset_structure.svg) | Canonical episode rows, explicit IDs, history masks, RGB/state inputs and body/hand target windows. Sources: index JSONL, manifest and V4 loader. | Dataset representation |
| 6 | [`figures/va_model_architecture.svg`](figures/va_model_architecture.svg) | Exact V4.1 layer sequence, tensor dimensions and output horizons. Sources: maintained V4/V4.1 model code and status. | Model architecture |
| 7 | [`figures/multimodal_fusion.svg`](figures/multimodal_fusion.svg) | Visual spatial tokens, normalized body queries, two cross-attention blocks and hand-state fusion. Sources: `va_research/models/v4/model.py`. | Multimodal fusion |
| 8 | [`figures/prediction_heads.svg`](figures/prediction_heads.svg) | Body-token head and hand-reference head, with output shapes `[10,64]` and `[50,14]`. Sources: model code. | Prediction heads |
| 9 | [`figures/auxiliary_tasks.svg`](figures/auxiliary_tasks.svg) | Reference-derived phase/progress/readiness labels, heads and loss connections. Sources: V4/V4.1 dataset and loss code plus history. | Auxiliary supervision |
| 10 | [`figures/closed_loop_inference.svg`](figures/closed_loop_inference.svg) | Current observation → V4.1 → absolute-tick PlanBuffer → native SONIC → hand PD/MuJoCo → next observation. Sources: runtime and rollout report. | Inference and control |
| 11 | [`figures/dataset_composition.svg`](figures/dataset_composition.svg) | 63-episode composition and explicit train/validation counts for the authoritative dataset. Sources: manifest and split. | Data coverage |
| 12 | [`figures/training_validation_curves.svg`](figures/training_validation_curves.svg) | Only recorded epochs 1, 10, 20, 30, 40 and 50; no invented per-step interpolation. Source: `history.json`. | Training diagnostics |
| 13 | [`figures/offline_prediction_results.svg`](figures/offline_prediction_results.svg) | Recorded token, FSQ, hand, phase and progress metrics. Source: `offline_metrics.json`; aggregate overlap is flagged in caption/source. | Offline evaluation |
| 14 | [`figures/native_physical_evaluation.svg`](figures/native_physical_evaluation.svg) | Native runtime integrity, renderer identity, lift/hold/contact result and failure state. Source: closed-loop report and renderer worker. | Physical evaluation and failure |
| 15 | [`figures/model_development_evolution.svg`](figures/model_development_evolution.svg) | Exploratory V4.1, auxiliary and architecture ablation metrics. Sources: retained ablation JSONs. Differences are descriptive, not causal claims. | Development history/ablations |

Captions should retain the distinction between expert replay and learned-policy execution. Figure 14 in particular must be captioned as a complete runtime with physical task failure, not as a success plot.
