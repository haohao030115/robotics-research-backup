# Target architecture

This document instantiates the mandatory logical architecture without changing the current Git boundaries or validated runtime behavior.

```text
Research Experiments & Orchestration
├── Kimodo Research
│   ├── Generation
│   ├── Motion Export
│   ├── Reference Adaptation
│   └── Qualification
├── ARDY Research
│   ├── Segmented Generation
│   ├── History / Overlap
│   ├── IK / Calibration
│   └── Qualification
├── Canonical Reference Interface
│   ├── Joint Order
│   ├── Coordinate Frames
│   ├── Timestamps
│   └── Metadata
├── SONIC + MuJoCo Execution
│   ├── Native Controller
│   ├── DDS
│   ├── Hand PD
│   └── Physical Validation
├── Canonical Data Factory
│   ├── Trace
│   ├── Dataset
│   └── Forensic Validation
└── VA Research
    ├── Models
    ├── Training
    ├── Closed-loop Rollout
    └── Evaluation
```

The primary path is `Kimodo/ARDY → Canonical Reference Interface → SONIC + MuJoCo Execution → Canonical Data Factory → VA Research`. The closed loop is `VA inference → SONIC controller → MuJoCo → RGB/state → VA inference`. Orchestration owns run identity, configurations, provenance, gates and reports across all modules; it is not a pipeline implementation layer.

## Physical package layout

```text
workspace/
  contracts/                         # canonical reference and run contracts
  research/kimodo/                   # adapters and Kimodo experiment wrappers
  research/ardy/                     # adapters and ARDY experiment wrappers
  execution/canonical_runtime/       # native SONIC/MuJoCo/DDS bridge
  data_factory/                      # trace, compiler, forensic validation
  va_research/                       # model registry, training, rollout, eval
  experiments/                       # cross-cutting CLI/config/reporting
  manifests/                         # immutable source/runtime/checkpoint manifests
  artifacts/                         # generated outputs, never imported as code
```

This is a logical organization. `GR00T-WholeBodyControl` remains an external/frozen dependency; Kimodo and ARDY model repositories remain independently maintained; current repositories can expose compatibility wrappers until migration is proven.

## Boundary contracts

| Boundary | Input | Output | Invariants |
|---|---|---|---|
| Producer → canonical reference | Producer-native motion, names, source metadata | `CanonicalReference` | 50 Hz, 29 body +14 hand, world root, wxyz, finite arrays, explicit case/task arm |
| Reference → execution | Validated reference + runtime manifest | `Trace` | native SONIC/DDS semantics, 200 Hz physics, 50 Hz control, 500 Hz command writer |
| Trace → data factory | ID-linked observations, references, tokens, RGB, physics | canonical dataset | no timestamp guessing; explicit IDs; active phases only; future offsets preserved |
| Dataset → VA | manifest, tensors, split, provenance | model/checkpoint | model family and I/O contract explicit; train-only normalization |
| VA → rollout | policy outputs | protocol-v4 tokens/hand references | 64D finite token, authored absolute tick, strict consumption checks |
| Execution → validation | trace and physics | separate gates | runtime completion and physical grasp success are distinct |

The protected semantics are unchanged: source hashes, joint order, quaternion convention, camera, FSQ, hand PD, scheduler and grasp thresholds.
