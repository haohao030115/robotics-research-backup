# Phase A — repository audit and integration plan

2026-10-08. Read-only inspection of code and saved artifacts, plus NPZ shape/dependency inspection inside `va-train`. No new training, native replay, closed-loop execution, or model inference was launched in this audit.

## Reuse inventory

| Component | Existing entry/API and actual contract | Reuse and minimal adapter |
|---|---|---|
| External producer | Four canonical reference files: reference.npz, metadata.json, provenance.json, reference_validation.json | Producer stays external. Carry original source hashes/producer metadata; do not ingest private producer formats in Trainer. |
| Compiler | `build_canonical_va_episode.py`; `canonical_va.pipeline.run(reference,out,gpu,recorded_episode,mode,...)`; output must be fresh | Invoke existing CLI as a process; preserve validation and all execution semantics. V4 workflow initially selects audit retention because compact outputs are insufficient for V4. |
| Compiler batch | `python -m canonical_va.batch --references-root ... --output-root ... --backend native --gpu 0 --mode audit|production` | Existing serial namespace isolation, continue-on-failure, accepted checksum/source resume, batch_summary. A nonzero batch exit and usable accepted subset must be distinguished explicitly; no silent stage success. |
| Canonical merge | `canonical_va.aggregate.aggregate(roots,output,duplicates='first')`; `merge_kimodo19.py` older one-off | Reuse existing acceptance/dedup index. New diversity audit must not add duplicate/near-duplicate detection. Merge alone does not create V4 horizons. |
| V4 dataset | `v4.dataset.CanonicalV4(manifest)`; exactly one accepted episode per source | Reuse directly per episode. It verifies five-slot prefix and extends original canonical ID-selected tables to body [10,64], hand [50,14]. Current observation only, RGB resized 128×128, body 68D, hand 28D. New builder stores its output with source hashes, per-episode ranges and explicit split. |
| Single-episode trainer | `train_token_v4.py --manifest --output --config --device`; `v4.training.main()` | Existing overfit path, not multi-episode validation. Keep available; do not pretend it has a held-out split. |
| Multi-episode trainer | `python -m v4.train_merged`; currently hardcoded kimodo19 paths, exactly 19 episodes and 4 validation episodes | Parameterize artifact/config/output/device/split I/O around existing loop. Preserve loss, AdamW, scheduler, random-init, normalization, checkpoint score and train-gate early stop. Do not duplicate optimization loop. |
| Offline evaluation | `evaluate_token_v4.py --checkpoint --output`; `v4.evaluation.evaluate(checkpoint,output)` | Existing API evaluates only checkpoint-embedded manifest. Reuse load/predict/metrics and factor the existing per-episode evaluation body behind explicit dataset input, preserving metric definitions. Label train/validation overlap; no held-out claim for reused episodes. |
| Closed-loop | `run_token_v4_closed_loop.py --checkpoint --output`; `v4.rollout.run(checkpoint,output)` | Actual native SONIC external-token execution + original TabletopDDS.step. Existing gate reads checkpoint sibling offline_metrics.json. Scenario, seed/config and gate-artifact selection need explicit I/O parameters; checkpoint must not be edited to redirect scenario. Keep scheduler/PD/control body unchanged. |
| Closed-loop comparisons/video | `python -m v4.comparison --checkpoint --rollout --output` | Existing exact-ID/tick expert comparisons and mp4 renderer. Requires full canonical states. Expose scenario input and validate initial conditions/frames/time before comparison; do not silently reuse checkpoint scene for another scenario. |
| Diagnostics | `v4.diagnostics.plan_consistency`, `compare_kimodo19`, `write_report` | Existing saved-array timing/hand-plan diagnostics, but some fixed paths/prose and some routines rerender/reinfer. Reuse formulas where applicable; ordinary report generation reads saved artifacts and must not silently invoke expensive diagnostics. |
| Tests | canonical_va/tests; va_training/tests/test_token_v4.py, test_v4_diagnostics.py | Add separate orchestration/report tests and small fixtures. Do not launch physical runs during unit tests. |

## Baseline to preserve

Config: `va_training/configs/token_v4.json`: seed 4; max 4000 epochs; batch 32; LR 0.0003; weights [1,1,.1,.1]. `v4.losses.objectives`: body SmoothL1, hand MSE, phase cross-entropy, progress MSE. AdamW weight_decay=0, grad clip=10, LR ×.3 after epochs 1500/3000. Evaluation at epoch 1/every 50; checkpoint score is TRAIN token_mae + hand_mae + .1 progress_mae; train offline gate can stop training. Validation is reported, not checkpoint selector. Normalization is train-only mean/std clamped >=.05 in merged training. No seed/model/optimizer/normalization path may be silently replaced.

## Verified artifact incompatibilities and risks

1. **Compact production != V4 dataset.** The successful compact episode contains 161 rows, body_target [161,5,64], hand_target [161,5,14], RGB and mapping_json, but no full encoder/reference/observation tables or original reference metadata/PNG mapping file. V4 requires all of these to build 10/50 horizons. Sparse 10Hz anchors cannot reconstruct the missing 50Hz hand trajectory or all future tokens. Do not pad, interpolate, repeat or alter canonical offsets. Initial integration uses existing full accepted audit episodes and compiles new V4 sources with audit retention. Supporting compact-only sources would need a separately versioned canonical retention payload; report this instead of inventing labels.
2. **Right-hand label semantics.** `progress_labels` explicitly selects right_* joints and fails without moving right reference joints. Metadata support for left does not establish left-task baseline validity. Audit may describe both arms; V4 build must surface this unsupported baseline case, not silently redefine labels.
3. **Merged contract mismatch.** New canonical aggregator emits episodes/ranges; old train_merged expects top-level output/sources and exactly 19 sources. Bridge these through a versioned model dataset, not fake legacy paths or new optimization code.
4. **Scenario coupling.** Offline/closed/comparison use checkpoint manifest; closed also demands offline_metrics.json next to checkpoint. Support independent evaluation identities with explicit manifest/gate inputs; preserve checkpoint hash and baseline gate, and do not silently relax gate on a new scenario.
5. **Saved reports are not all generic.** Existing write_report contains experiment-specific conclusions. A deterministic generic report must calculate summaries from saved evidence and mark unavailable metrics, not copy historical prose.
6. **Render domain differences.** Hardware MTT S4000 and old llvmpipe images differ photometrically. Keep renderer provenance. Do not silently rerender existing training RGB with another renderer. Preserve original canonical alignment.
7. **Runtime vs task result.** Saved kimodo19 normal_gpu report has strict_runtime_pass=true, rollout_complete=true, failure=null, physical_grasp_success=false. This should become evaluation execution SUCCEEDED with 0% grasp success, not infrastructure FAILED. Incomplete/failed runtime remains a separate status.
8. **Hash gates.** Modifying tracked compiler/evaluator helpers changes fingerprints and may trigger full QA. Leave compiler/native files untouched where possible. No automatic source-policy refresh.
9. **One encoder cross-backend discrepancy already exists.** Preserve rejection and evidence; do not reopen indexing or lower tolerance in this integration.

## Environment and fixtures

Execution environment is existing `va-train`, Python 3.10. numpy, torch, torch_musa, mujoco, PIL, imageio and jinja2 found. matplotlib is NOT on default import path; historical evaluator explicitly adds `outputs/va_v2_diagnosis/plot_dependencies`. Generic CLI must expose/configure plotting dependency availability rather than silently hardcode that user path. Use unittest; pytest absent. Do not replace MUSA with downloaded CPU/CUDA wheels.

Native episodes need SDK/DDS, original binary/model identities and `unshare --net` with loopback enabled. Rendering uses the existing checksum-verified graphics bootstrap before importing MuJoCo. Ordinary reports do not need DDS/GPU initialization.

Usable fixtures: full accepted `outputs/canonical_va/kimodo_episode_0001_native_final`; full audit `outputs/canonical_va_productionization_20261008/audit_regression`; compact accepted production sample; `outputs/va_v4_kimodo19` saved checkpoint/history/split/offline predictions/full closed-loop physics/video. Saved training finished at epoch 50, train gate passed; split is 15 train / 4 validation, 2280 / 608 usable samples. Offline predictions hold 152 samples with body [152,10,64], hand [152,50,14]. Closed physics has 810 rows with both wrist poses, joints, cube, contacts, integration state and explicit frame/time fields.

## Proposed artifact contracts and architecture

New lightweight `grasp_pipeline` package: CLI + atomic JSON manifests/state + process adapters + deterministic reports. No robotics/model/indexing logic in orchestrator. Use `python -m grasp_pipeline` as portable CLI; optional `grasp` console alias.

Independent immutable IDs: reference batch, canonical episode, model dataset, training run, checkpoint, evaluation run, report. Experiments link these by absolute/relative artifact URI and checksums; dataset can serve many training runs; checkpoints can serve many evals. Each stage attempt writes a new directory and process log; successful artifacts get checksummed completion records and are never overwritten. Running lock / PID and interrupted attempt handling prevent duplicate launches; retry creates new attempt.

```
experiments/<id>/experiment_manifest.json
experiments/<id>/stages/<stage>/attempt_<n>/
experiments/<id>/reports/index.html
artifacts/datasets/<dataset-id>/{manifest.json,split.json,samples.npz,dataset_audit/}
artifacts/training/<run-id>/{manifest.json,checkpoints/,history.csv,metrics.json,report.html,figures/}
artifacts/evaluations/<eval-id>/{manifest.json,metrics.json,per_episode.json,report.html,figures/,rollouts/}
```

Dataset manifest records V4 baseline config/hash, source canonical manifests/checksums, producer/task metadata, original sample identities, per-episode ranges and split. No private producer schema in training. Dataset audit is descriptive, never an additional acceptance/quality gate. Missing wrist/labels/phase timing are explicitly unavailable; derived closure phases must not be called physical reach/grasp/lift labels.

Statuses: NOT_STARTED/RUNNING/SUCCEEDED/FAILED/BLOCKED/SKIPPED. SUCCEEDED requires successful process and artifact validation; dependencies checked before spawn. Error kind distinguishes execution, artifact and dependency; task performance remains metrics. Exit codes should distinguish success, execution/artifact failure, dependency/manual-stop and CLI error. `run --stop-before train` prepares only; an explicit `train` command authorizes training. `resume` must never implicitly grant train authorization after preparation.

Reports are local HTML + JSON + saved figures, no LLM/service. Saved predictions allow confusion/per-horizon/MAE diagnostics without training changes. LR can be derived from recorded baseline schedule, explicitly labeled derived; unavailable gradient norms are marked unavailable. Checkpoint selection remains the baseline train score, irrespective of new report metrics.

## Concrete implementation order

B1. CLI parsing, versioned contracts, immutable artifact IDs/atomic stage state/locks, dry-run, status/compare/resume and explicit training boundary; unit-test state before connecting heavy tasks.
B2. Thin compiler process adapter and accepted full-source index; V4 builder invokes existing CanonicalV4 unchanged, saves its exact tensors/IDs/split/provenance. Reject insufficient compact payload explicitly.
C1. Dataset descriptive audit on existing accepted full episode: task/producer/split/spatial coverage, wrist/joint/state distributions, RGB contact sheet, labels/per-horizon statistics. No duplicate/quality detection.
C2. Parameterize existing merged-training I/O, preserving loop; convert saved history/predictions to deterministic training HTML/CSV/metrics. Verify existing saved run before any new optimization.
D1. Existing offline evaluation adapter and report; explicit evaluation source and split-overlap disclosure.
D2. Existing native closed-loop adapter with scenario/config/seed/output I/O, unchanged execution; saved-rollout report/video linking, early 1–3s views and compatibility-checked expert comparisons. Validate with saved artifacts/mocks first.
E. CLI/contracts/state/interruption/resume/artifact validation unit tests; source/AST guards for untouched model/loss/controller/indexing; report determinism and wrapper dry runs. Document exactly which paths are only mocked.
Only after these checks propose a bounded real integration smoke test. User explicitly requires authorization before expensive training, production batches or closed-loop runs; none are part of Phase A.

## Phase B retention recheck (2026-10-08, additive clarification)

Rechecked the current productionized `pipeline.py`/`production.py`, accepted production summaries under `outputs/canonical_va*`, and actual NPZ files before Phase B coding. Latest accepted production found: `outputs/canonical_va_productionization_20261008/batch/kimodo_episode_0001` (later `merged*` are aggregations; `production_final` is rejected). It retains only manifest, summary, provenance and samples.npz. The provenance contains source/code hashes and runtime metadata, not a companion full-sequence payload. This is not inferred from the five-slot array shape alone: `compact_success` explicitly deletes canonical_replay/source_reference and all dataset files other than samples.npz after materialization/validation.

The same current compiler already generates complete canonical tables; supported `--mode audit` retains them. The current `audit_regression` artifact has 816 encoder tokens, 816 reference/hand rows and 816 observations. `CanonicalV4` resolves the authored target IDs and extends those tables; it does not extend compact target arrays. **Decision: Case B, no compiler change required.** Phase B uses existing full-retention outputs and new compile commands default to audit retention. Full retention costs additional storage and complete QA; compact outputs remain useful for their existing five-slot consumers. An unrelated historical full replay of the same episode ID is not an interchangeable companion for a different compact physical run.
