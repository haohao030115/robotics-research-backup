# Final reorganization verification

Date: 2026-10-15  
Environment: `va-train`, Python 3.10.12, torch 2.9.0, torch_musa 2.9.0+8432052, MTT S4000

## Authoritative ownership

| Capability | Authoritative implementation | Compatibility / external boundary |
|---|---|---|
| Kimodo generation, export, qualification | `research/kimodo/` | `kimodo-t6/` checkpoint and package assets |
| ARDY generation, adaptation, qualification | `research/ardy/` | `workspace/fuyuhan/ardy` model checkout |
| Shared contracts | `contracts/` | legacy readers remain for historical fixtures |
| Canonical compilation and dataset production | `execution/sonic_mujoco/`, `data_factory/` | protected SONIC and frozen `dl-group2` scene/runtime |
| VA V4/BCZ training, offline evaluation, rollout | `va_research/models/v4/`, `va_research/registry.py` | V2/V3 and `va_training/` retained as historical compatibility |
| Experiment execution and provenance | `experiments/` | manifests and stage logs are immutable per run |
| Qwen ARDY online research | `research/ardy/online/qwen/qwen_ardy` | old workspace path contains `SOURCE_MOVED.md` and archive links |

## Verification matrix

| Workflow | Status | Evidence |
|---|---|---|
| Kimodo generation | BLOCKED at quality gate | `python -m experiments run --family kimodo ...`; local checkpoint loaded and 487 frames generated; export rejected existing IK gate (`Offline IK target error >.02m`, measured ~6.99m) |
| ARDY generation → adaptation | PASS | `/tmp/reorg_ardy_e2e_final4/artifact/{source,candidate}`; prepare, generate and adapt exit 0 |
| ARDY isolated SONIC replay | PASS (runtime), FAIL (physical task) | 810/810 frames, no missing ticks, no foreign processes, integrity pass; physical grasp false, max lift 0.00476m vs 0.1m requirement |
| Canonical dataset production | PASS | `python -m experiments dataset --canonical-root outputs/canonical_va/kimodo_batch_20261008_v4 ...`; `/tmp/reorg_dataset_final3/run_manifest.json` is `SUCCEEDED`, dataset stage exit 0 |
| VA V4 training | PASS | `python -m experiments va train ...`; `/tmp/reorg_va_train_smoke_final/artifact/checkpoint.pt`, status `SUCCEEDED` |
| VA V4 offline evaluation | PASS | `python -m experiments va offline ...`; `/tmp/reorg_va_offline_final2` status `SUCCEEDED`, gate true |
| VA V4 closed-loop rollout | PASS (runtime), task result recorded separately | `/tmp/reorg_va_closed_final2`; 810/810 ticks, zero uncovered ticks/deadline misses, MTT renderer verified |
| Native SONIC replay | PASS | prior canonical native smoke run: 800 frames, native binary and isolated DDS |
| GPU rendering | PASS | `python -m execution.sonic_mujoco.rendering --profile auto --mode production --frames 3 ...`; `GL_VENDOR=Moore Threads`, `GL_RENDERER=MTT S4000`, `software=false`, accepted=true |
| Reorganization regression tests | PASS | `python -m unittest tests.reorganization.test_contracts tests.reorganization.test_execute tests.reorganization.test_workflows -q`; 11 tests passed |

## Repository hygiene

The complete inventory is recorded in `FULL_FILE_INVENTORY_20261015.jsonl.gz` with summary in `FULL_FILE_INVENTORY_SUMMARY.md`: 565,212 records, zero walk errors, and every top-level entry classified. Historical reports, handoffs, stale environments, generated outputs, and external checkouts are retained under explicit archive or workspace ownership. No unrelated research checkout was deleted.

## Remaining issues

Kimodo and ARDY generated motions can fail their existing physical quality gates; those thresholds were not relaxed. ARDY and VA closed-loop runtime integrity passes, while the physical grasp result is false and is reported separately. V2/V3 implementations remain compatibility-only historical families. Native physical-runtime tests require the validated `va-train` container and are not claimed from host execution.
