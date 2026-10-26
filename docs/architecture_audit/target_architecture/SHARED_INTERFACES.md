# Shared interfaces

These are small data contracts, not a new framework. They define ownership and prevent cycles while preserving current implementations.

## Canonical reference

`CanonicalReference` contains `time_s`, `body_ref_q`, `body_ref_dq`, `root_ref_pos`, `root_ref_quat`, `hand_ref_q`, optional hand velocity, joint names, units, `quaternion_convention=wxyz`, `root_pose_frame=world`, `reference_hz=50`, case, task arm, producer, source hashes and qualification provenance. Validation is the current `canonical_va.schema.validate` contract (`schema.py:19-251`).

## Producer interfaces

```text
ReferenceProducer.generate(config) -> ProducerReference
ReferenceExporter.export(reference, target_contract) -> IntermediateReference
ReferenceAdapter.adapt(intermediate, case_profile) -> CanonicalReferenceCandidate
Qualifier.run(candidate, runtime_profile) -> QualificationReport
```

Producer-specific fields remain in namespaced metadata. Kimodo and ARDY do not call one another or import VA.

## Execution and trace

```text
ControllerBackend.reset(reference, runtime_profile) -> RuntimeHandle
ControllerBackend.step(input) -> Observation
TraceWriter.record(observation, command, token, publication, receipt)
TraceValidator.validate(trace) -> ValidationReport
```

`NativeSonicBackend` wraps the existing native controller/DDS path. `SimulationBackend` is an explicit diagnostic backend. Neither changes SONIC formulas, DDS, hand PD, timing or MuJoCo physics. Trace joins use IDs, never nearest timestamps.

## Dataset and policy

```text
DatasetCompiler.compile(Trace, DatasetContract) -> DatasetArtifact
PolicyRegistry.load(ModelId, checkpoint, DatasetContract) -> Policy
Policy.predict(RGB, BodyState68, HandState28) -> PolicyPlan
RolloutBackend.run(PolicyPlan, ControllerBackend) -> RuntimeReport
```

V4 `PolicyPlan` is body tokens `[10,64]`, hand references `[50,14]`, phase logits `[3]`, progress scalar; BCZ adds closure intent. Native SONIC consumes a finite 64D protocol-v4 token at an authored absolute execution tick.

## Error and provenance rules

Every interface fails closed on missing metadata, nonfinite arrays, shape/order mismatch, stale hashes or invalid timing. Every artifact has a run ID, schema version, source/checkpoint hashes, command, environment/runtime manifest and separate gate results. Runtime feedback is represented by observations/events; packages remain acyclic: contracts → producers/runtime/training → orchestration, with rollout depending on runtime only through the backend interface.
