# Kimodo reorganization

Kimodo is a first-class research module, while its model implementation and large checkpoint remain external. The module must expose four independently testable concerns.

## Target structure

```text
research/kimodo/
  generation/          # prompt/config → native Kimodo motion
  export/              # Kimodo representation → G1/MuJoCo arrays
  adaptation/          # case/task-arm adaptation and IK
  qualification/       # native replay, physical criteria, reports
  configs/
  artifacts/           # references/reports by run ID, not imported code
  cli.py
```

## Existing implementation mapping

* Generation delegates to `kimodo/scripts/generate.py:286-403` and the Kimodo package in `kimodo-main/kimodo/`. It records model name, checkpoint hash, prompt, seed, FPS and output hash.
* Export delegates to `kimodo_to_t7_reference.py:163-244` and `MujocoQposConverter`. It records source names, canonical names, quaternion order, interpolation and derivative semantics.
* Adaptation delegates initially to `dl-group2/scripts/adapt_model_reference.py`, including arm mirroring, smooth case displacement and actual-model per-frame IK. It must not share ARDY’s calibration or overlap algorithm merely because both produce `reference.npz`.
* Qualification delegates to the existing native canonical path and `PhysicalGraspEvaluator`; a qualification report must separate reference validity, runtime completion and physical task success.

## Kimodo interface

`generate(config) -> ProducerReference` produces native samples and provenance. `export(reference, target_contract) -> BodyRootIntermediate` produces finite arrays. `adapt(intermediate, case_profile) -> CanonicalReferenceCandidate` records every transform and IK error. `qualify(candidate, runtime_profile) -> QualificationReport` invokes native SONIC/MuJoCo without changing the controller.

Required shapes at the canonical boundary are `[T,29]` body q/dq, `[T,14]` hand q/dq, `[T,3]` root position and `[T,4]` wxyz root quaternion at 50 Hz. Metadata includes producer type `kimodo`, source checkpoint/hash, source FPS, target FPS, joint lists, case, task arm, units and frame definitions.

## Compatibility and risks

Existing generation scripts and qualification snapshots remain callable. The first implementation should be a wrapper and manifest writer, not a move. The highest risks are converter joint order, root frame, quaternion interpolation, model checkout identity and adaptation numerics. Acceptance requires exact or explicitly bounded array comparisons, source hash parity, canonical schema validation and one successful native qualification replay. A failed physical grasp remains a valid failed report and must not be promoted.
