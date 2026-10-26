# ARDY reorganization

ARDY remains an independently maintained research workflow. Its segmented generation, history/overlap semantics and calibration are not generic Kimodo utilities.

## Target structure

```text
research/ardy/
  generation/          # ARDY model, prompts, segmented sampling
  overlap/             # native history and transition handling
  conversion/          # ARDY representation → MuJoCo/G1
  adaptation/          # wrist IK, clearance and calibration
  qualification/       # isolated replay and physical validation
  online/              # Qwen/staged/rolling experimental variants
  configs/
  artifacts/
  cli.py
```

## Existing implementation mapping

* `workspace/fuyuhan/ardy/` is the external model/checkpoint dependency.
* `ardy_tabletop_segmented_20261008/code/run_pipeline.py:58-74` is the current orchestration entry point.
* `generate_segmented.py:95-150` owns segmented generation, native ARDY history, overlap blending and conversion to 406-frame/25 FPS output.
* `adapt_ardy.py:40-113` owns 25→50 FPS interpolation, SLERP, planned wrist targets, refinement, IK and optional measured calibration.
* `replay_ardy_isolated.py` and `validate_grasp_run.py` own isolated native qualification and report validation.
* `research/ardy/online/qwen/qwen_ardy/**` remains an online/staged experiment area behind an explicit experimental adapter; it must not be used as the canonical ARDY producer by default.

## ARDY interface

`generate_segments(config) -> SegmentedMotion` records model/checkpoint, prompt list, seed, segment lengths, transition frames, native history frames and FPS. `convert(segmented) -> MuJoCoIntermediate` records representation and joint mapping. `adapt(intermediate, case, calibration) -> CanonicalReferenceCandidate` records clearance, orientation, blend windows, IK errors and calibration hashes. `qualify(candidate) -> QualificationReport` uses isolated DDS/native SONIC.

The canonical boundary remains 50 Hz, finite 29/14 arrays, world root and wxyz. ARDY-native 25 FPS and 406-frame semantics stop at the conversion boundary. No Kimodo overlap or calibration code is substituted.

## Acceptance and rollback

Acceptance requires reproducible segment logs, identical overlap/history metadata, no extrapolation beyond source coverage, bounded wrist IK errors, calibration source hashes, canonical schema acceptance and native qualification evidence. The first migration is a facade around `run_pipeline.py`; rollback is direct invocation of the existing script. Online Qwen/ARDY experiments require their own metrics and must be labeled experimental.
