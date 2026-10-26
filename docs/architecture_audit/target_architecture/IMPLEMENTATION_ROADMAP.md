# Implementation roadmap

Only documentation is being produced in this task. The following stages are ordered by dependency and regression risk.

## Stage 1 — Freeze validated baselines (lowest-risk first)

**Scope:** new manifests under `manifests/` or existing run outputs; no source changes. Record commits/hashes, checkpoints, commands, container, EGL profile, SONIC release, canonical runtime contract and reports. **Dependencies:** none. **Benefit:** reproducibility baseline. **Tests:** environment checker, hash verification, reopen reports. **Rollback:** remove additive manifests. **Acceptance:** every promoted run has immutable provenance and separate gates.

## Stage 2 — Declare module ownership

**Scope:** add package ownership/readme maps and compatibility entry-point inventory for `GR00T-WholeBodyControl`, `dl-group2`, `canonical_va`, `grasp_pipeline`, `va_training`, Kimodo, ARDY and Qwen/ARDY. **Dependencies:** Stage 1. **Benefit:** prevents new cross-boundary code. **Tests:** import/dependency review and launch inventory. **Rollback:** documentation-only. **Acceptance:** every entry point has one owner and one target module.

## Stage 3 — Kimodo and ARDY facades

**Scope:** add wrappers/manifests around existing generation, export, adaptation and qualification commands; no algorithm movement. **Dependencies:** ownership and manifests. **Benefit:** first-class independent research workflows and consistent provenance. **Tests:** source array comparisons, IK/calibration evidence, native qualification fixtures. **Risk:** medium/high. **Rollback:** invoke original commands. **Acceptance:** Kimodo and ARDY each produce a canonical candidate and report without importing VA.

## Stage 4 — Canonical contract facade

**Scope:** add a contract package/facade delegating to `canonical_va.schema`, `runtime_contract.json`, `source_policy.json` and existing validators. **Dependencies:** Stage 1. **Benefit:** one explicit interface without changing accepted data. **Tests:** fixture acceptance/rejection, byte-stable metadata, canonical tests. **Risk:** medium. **Rollback:** existing validators. **Acceptance:** old and new validators agree on the full fixture corpus.

## Stage 5 — VA registry and package facades

**Scope:** register V2/V3/V4/V4.1/BCZ/oracle families and wrap current trainers/evaluators/rollouts. **Dependencies:** contract facade. **Benefit:** unambiguous checkpoint/model selection and separate training from runtime. **Tests:** checkpoint load, output shapes, offline metric parity, scheduler unit tests. **Risk:** medium. **Rollback:** family-specific scripts. **Acceptance:** every checkpoint identifies model, dataset, schema and rollout contract.

## Stage 6 — Canonical data-factory facade

**Scope:** facade around trace materialization, compiler and forensic validation; preserve ID joins and artifact formats. **Dependencies:** contract facade and runtime manifests. **Benefit:** clear trace/dataset/forensics ownership. **Tests:** `validate_export`, tensor/mapping equality, full audit report. **Risk:** high. **Rollback:** `canonical_va.pipeline`.

## Stage 7 — Unified orchestration

**Scope:** make `grasp_pipeline` a cross-cutting CLI over producer, canonical, dataset and VA facades; add immutable run records and reproduction. **Dependencies:** all prior stages. **Benefit:** consistent configuration, status and reporting. **Tests:** dry-run commands, fresh-output rules, representative Kimodo/ARDY/VA reproduction. **Risk:** medium. **Rollback:** existing entry points remain wrappers.

## Stage 8 — Redundancy review

**Scope:** only after launch evidence and regression results, identify proven duplicate wrappers or superseded tools. No deletion or retirement in this task. **Dependencies:** historical run registry and observation window. **Benefit:** reduced ambiguity. **Tests:** consumer search, subprocess/DDS launch inventory, golden replay. **Risk:** high. **Rollback:** restore archived wrapper. **Acceptance:** explicit owner confirms no deployment or historical reproduction depends on the candidate.

## Invariants and first implementable stage

All stages preserve native SONIC/release ONNX, MuJoCo stepping, DDS, joint order, wxyz, timing, FSQ, hand PD, camera, sample IDs, VA I/O, scheduler checks and physical thresholds. The first migration stage that can be implemented with minimal regression risk is **Stage 1: additive baseline manifests and run provenance**, because it changes no imports, algorithms, files, runtime configuration or artifacts and can be validated read-only.
