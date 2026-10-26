# Safe refactoring roadmap

This plan is documentation/design only. It deliberately leaves validated runtime code untouched until each stage has a golden artifact and rollback point.

## Stage 0 — Freeze and observe (P0)

**Scope:** all current repositories and validated outputs. Record Git commits where possible, source hashes where not, container/image/runtime, renderer profile, SONIC binary/ONNX, checkpoints, CLI commands and manifests. Add no imports or runtime changes. **Risk:** low. **Validation:** `bash scripts/check_environment.sh`; verify existing canonical/V4 reports and artifact hashes. **Rollback:** delete only the new manifest files.

## Stage 1 — Contract extraction (P0)

**Scope:** copy the current schemas, joint orders, wxyz/units, 50/200/500 Hz timing, sample IDs, model shapes and grasp criterion into a standalone `contracts` package. Keep existing validators as the authority and compare outputs byte-for-byte. **Risk:** medium around legacy aliases. **Validation:** canonical tests, V4 dataset tests, schema fixture corpus, exact hashes and serialized mapping equality. **Rollback:** stop using the package; existing validators remain.

## Stage 2 — Runtime manifest and path resolution (P0/P1)

**Scope:** generate one read-only manifest resolving SONIC, robot XML, scenes, ONNX, observation YAML, native binary, graphics profile and container paths. Replace only new call sites first; retain `/workspace/group2` compatibility. **Risk:** high if path precedence or asset hashes change. **Validation:** source-policy check, native build hash, EGL probe, dry-run CLI and one recorded replay. **Rollback:** restore old path helpers and manifest version.

## Stage 3 — Producer adapters (P1)

**Scope:** wrap Kimodo, ARDY and rule generation behind explicit producer/adaptation records. Do not merge model libraries or algorithms. Preserve current resampling, SLERP, joint mappings, wrist IK, calibration and mirror behavior. **Risk:** high in reference numerics. **Validation:** array equality or approved tolerance against each existing candidate, metadata comparison, per-frame IK error and one native canonical replay per producer. **Rollback:** invoke existing scripts directly.

## Stage 4 — Canonical runtime boundary (P0)

**Scope:** isolate native replay, MuJoCo bridge, renderer and trace materialization behind `ControllerBackend`; move no controller math. Keep instrumented native build separate from the frozen release binary and preserve DDS namespace behavior. **Risk:** very high. **Validation:** native physical replay, exact initial qpos/qvel, command/publication/receipt IDs, timeline and full forensic audit. **Rollback:** use current `canonical_va.pipeline` entry point.

## Stage 5 — Dataset compiler and model registry (P1)

**Scope:** centralize canonical-to-V4 joins and register V2/V3/V4/V4.1/BCZ contracts. Keep serialized dataset fields, future offsets, normalization and checkpoint keys unchanged. **Risk:** medium. **Validation:** `validate_export`, tensor byte equality, train/validation split identity, offline metrics equality and checkpoint load tests. **Rollback:** use current `CanonicalV4` and family-specific entry points.

## Stage 6 — Rollout adapters (P0)

**Scope:** define one rollout interface with native SONIC and simulation implementations; keep `PlanBuffer`, absolute ticks, protocol-v4 packer, hand PD, renderer gate and strict token-consumption checks intact. **Risk:** very high timing/DDS risk. **Validation:** scheduler unit tests, a known failing uncovered-tick case, a complete runtime trace, native closed loop in isolated DDS, and separate task-success gate. **Rollback:** current `v4.rollout`/BCZ rollout scripts.

## Stage 7 — CLI and experiment packaging (P1/P2)

**Scope:** make `grasp_pipeline` and experiment scripts thin orchestration over installed packages; eliminate new `sys.path` edits and absolute paths while preserving old commands as compatibility wrappers. Move generated outputs only after artifact registry support. **Risk:** medium. **Validation:** dry-run commands, fresh-output immutability, report generation and representative Kimodo/ARDY/rule jobs. **Rollback:** retain old scripts and output locations.

## Stage 8 — Retirement review (P2)

Retire or archive only after launch inventory, subprocess/DDS observation, external deployment review and a minimum evidence window. Legacy V2/V3/oracle/synchronous tools remain available for historical reproduction until their consumers and checkpoints are registered. No file is deleted based only on import absence.

## Required invariants throughout

Do not change controller frequencies, timing/index conventions, joint order, quaternion order, FSQ representation, hand gains, camera contract, source-policy hashes, DDS semantics, reset/standby behavior, physical success thresholds or dataset field meanings. Any intentional change needs a new schema/runtime version and side-by-side evidence.

## Executive summary

Today the system is a multi-repository stack: Kimodo, ARDY and rule producers emit motion references; local adapters turn them into 50 Hz G1 references; canonical VA runs the original SONIC native controller over isolated DDS with MuJoCo and records ID-linked traces; the compiler creates V4 horizons; MUSA VA models predict body tokens and hand references; a strict absolute-tick rollout feeds protocol-v4 tokens back to SONIC. The runtime and task gates are separate: the open BCZ run completed all 810 ticks with no scheduler gaps but failed physical grasp (0.0228 m lift, 0 s hold).

The most serious problems are reproducibility and boundary problems: untracked/version-ambiguous source, duplicated schemas and path resolution, many model-family entry points, and timing-sensitive Python/C++/DDS/EGL coupling. The frozen SONIC release, native control arithmetic, canonical timing/mappings, dataset formats, validated renderer, and grasp criterion should remain unchanged.

Reorganize first around a standalone contract package, immutable runtime/source manifests, explicit Kimodo/ARDY/rule producer adapters, and a native controller boundary. Keep upstream SONIC separate, keep generated artifacts out of source packages, and register every model/checkpoint family. Migrate in freeze → contracts → manifests → producer adapters → canonical runtime boundary → dataset/model registry → rollout adapters → CLI packaging stages, validating each stage against golden traces and retaining the current entry point as rollback.
