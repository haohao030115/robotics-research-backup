# Canonical Data Factory Technical Reference

This document describes the implemented Canonical Data Factory and the evidence
contract used to convert a native SONIC/MuJoCo grasp execution into VA training
samples. The implementation is in `data_factory/runtime/` and
`data_factory/dataset/`; the accepted production artifacts are under
`outputs/unified_adapter_v1/canonical_compilation_20261017/`.

## Scope and execution boundary

The factory begins with a producer reference episode exported on the canonical
G1 interface. Producer generation is already complete at this boundary. The
factory validates the reference, executes or revalidates the original native
SONIC/DDS path, materializes ID-linked traces, performs physical and forensic
gates, and only then exports VA samples. The transaction is fail-closed:
`data_factory/runtime/pipeline.py` defines the required flags at lines 8--11,
validates and hashes the source at lines 59--65, runs the native replay at lines
69--80, runs basic checks at lines 82--84, and invokes full forensic validation
at lines 85--97. Failed traces remain available for audit; only
`expert_valid == true` episodes reach `compile_samples` (lines 98--106).

The batch wrapper serializes episodes and creates a separate network namespace
for each native invocation (`data_factory/runtime/batch.py`, lines 12--48).
Existing accepted outputs may be skipped only when their source hashes and
episode identity still match; the wrapper never overwrites an accepted output.

## Canonical source-reference schema

`data_factory/runtime/schema.py` is the source gate. It requires an explicit
producer type (`kimodo`, `rule_based`, or `ardy`), a `wxyz` root quaternion in
the world frame, and canonical units

\[
\{t\!:\!\mathrm{s},\ q_{body}\!:\!\mathrm{rad},\
\dot q_{body}\!:\!\mathrm{rad/s},\ p_{root}\!:\!\mathrm{m},\
q_{root}\!:\!1,\ q_{hand}\!:\!\mathrm{rad}\}.
\]

The body and hand orders are read from SONIC and must contain exactly 29 and 14
joints, respectively (`schema.py`, lines 100--110). Every reference array must
be finite and have a common length \(N\geq 7\):

| Array | Shape | Units |
|---|---:|---|
| `time_s` | `[N]` | s |
| `body_ref_q`, `body_ref_dq` | `[N,29]` | rad, rad/s |
| `root_ref_pos` | `[N,3]` | m |
| `root_ref_quat` | `[N,4]` | unitless, unit norm |
| `hand_ref_q` | `[N,14]` | rad |

The time origin is zero and \(t_{k+1}-t_k=1/50\) s exactly within the
configured tolerance (`schema.py`, lines 112--190). The initial hand reference
must be zero. A case identifier, finite block XY/yaw, task arm, and one phase
label per frame are also required. Phases are `active`, `reset`, `standby`, or
`inactive`; only active segments can produce samples.

## Native runtime and temporal identity

Accepted metadata records the original controller and simulator, not a
synchronous substitute. Native replay uses the original G1Deploy observation,
policy-command, and low-command writer, the original `TabletopDDS.step`, and
the original hand PD path. Production metadata declares 200 Hz physics, 50 Hz
control decisions, 500 Hz command publication, and 10 Hz RGB. The accepted
episodes use MTT S4000 rendering with a 320 x 240 `head_rgb` camera. Timing is
explicitly asynchronous: command publication, DDS receipt, and physics
consumption are joined by native IDs rather than forced into a four-step
lockstep (`canonical_replay/metadata.json`, `runtime.timing`; also
`data_factory/runtime/native_forensic.py`, lines 114--144).

Each native decision receives a unique observation, control-tick, decision,
reference-frame, token, command, and hand-reference identity. The observation
is the **pre-action physical snapshot selected by native LowState**. Physics may
advance while inference and transport are in flight. Each command retains its
publication and receipt records, the consuming physics-step IDs, and any
intervening steps. The physics table distinguishes `native_DDS` body control
from startup WBC and stores pre/post integration states. The hand cursor is
recorded separately at every physics step. This representation prevents a late
or duplicated log row from silently changing the action associated with an
observation (`data_factory/runtime/native_truth.py`, lines 31--101).

RGB frames are generated for active reference indices divisible by five. The
factory restores the exact integration state before rendering, stores the
observation and simulation IDs, and records a snapshot and image SHA-256. The
forensic validator re-renders every saved frame and requires pixel error zero
(`native_forensic.py`, lines 56--68). There is no nearest-timestamp repair or
row-index assumption.

## Forensic gates

Forensics verify all links that matter for a reproducible training example:

* native table lengths, unique event IDs, finite arrays, and the exact 1,762-D
  encoder and 994-D decoder input schemas (`native_forensic.py`, lines 40--55);
* RGB observation identity, simulation timestamp, snapshot hash, and exact
  replayed pixels (lines 56--68);
* independent ONNX Runtime CPU reconstruction of native encoder and decoder
  values, with a 1e-5 maximum error tolerance (lines 71--80);
* cross-decision equality of observation/tick/reference/token IDs, exact
  previous-action causality, and the current q/dq snapshot bound into the
  decoder input; reference windows are checked against the authored 50 Hz
  source without an index shift (lines 81--113);
* publication-to-receipt-to-consumption causality, native DDS identity, and
  the distinction between selected DDS commands and effective startup WBC
  control (lines 114--144);
* exact reconstruction using the original `TabletopDDS.step` and hand PD,
  requiring maximum state error at most 1e-8 and agreement with the recorded
  physical metrics (lines 145--180);
* body and hand future targets sharing the same reference IDs at offsets
  \(\{2,3,4,5,6\}\) control ticks (lines 182--190).

The basic production gate additionally requires complete native execution,
complete reference-frame coverage, no missing cursor ticks, 200 Hz physics
spacing, contiguous simulation-step IDs, finite arrays, valid RGB hashes, and
the physical criterion of at least 0.10 m lift, at least 2 s continuous hold,
valid opposing-hand contact, no cube/table contact during the qualifying hold,
and no fall (`data_factory/runtime/production.py`, lines 49--98).

## Canonical episode materialization

`native_truth.materialize` creates the canonical replay tables. For every native
decision it stores a pre-action observation, the authored reference row, exact
SONIC encoder input and post-FSQ 64-D token, exact decoder input and native
29-D command, and command/timing evidence. RGB files are linked through
`frames.jsonl`; all rows retain explicit IDs. Training-retention artifacts keep
the source reference, canonical metadata, encoder table, reference table,
observation table, sample mapping, packed RGB, and validation reports.

## VA supervision compiler

The VA compiler is deliberately separate from replay. `plan_samples` joins
observations, references, and tokens by explicit IDs (`compiler.py`, lines
9--22). For an active anchor index \(t\), it requires active reference frames
from \(t\) through \(t+6\), then uses history \([t-2,t-1,t]\), clipped to the
first active index, with a three-element validity mask. The target offsets are

\[
\Delta = \{2,3,4,5,6\},
\]

which correspond to 40--120 ms at 50 Hz. Every body-token target and hand
target must point to the same future reference-frame IDs; missing IDs, phase
boundaries, duplicate ticks, and incomplete horizons are rejected
(`compiler.py`, lines 23--48).

`compile_samples` stores the five-slot prefix used by the canonical sample
contract (`compiler.py`, lines 51--70):

* `body_target`: post-FSQ SONIC tokens `[5,64]`;
* `hand_target`: absolute hand references `[5,14]`;
* three observations for each of `body_q[29]`, `body_dq[29]`, `hand_q[14]`,
  `hand_dq[14]`, `root_pos[3]`, `root_quat[4]`, and `root_velocity[6]`;
* `history_valid_mask[3]` and the RGB frame bound to the anchor observation.

The V4 loader extends the verified contiguous IDs to body `[10,64]` and hand
`[50,14]` targets. It uses only the final/current state entry for the model:

\[
x_{body}=[q_{29},\dot q_{29},q_{root,4},\omega_{root,3},g_{3}]\in\mathbb{R}^{68},
\quad x_{hand}=[q_{hand,14},\dot q_{hand,14}]\in\mathbb{R}^{28}.
\]

RGB is resized from 320 x 240 to 128 x 128. Nine tail rows per 161-row episode
are excluded because a complete ten-token/fifty-hand future horizon is not
available (`va_research/models/v4/dataset.py`, lines 82--125).

The closure labels are reference-derived. The loader projects each hand
reference onto the initial-to-maximum-norm displacement of right-hand joints
whose peak displacement exceeds 0.05 rad, clips the scalar to `[0,1]`, and
labels OPEN for progress <= 0.05, CLOSING for intermediate progress, and HOLD
for progress >= 0.95 (`va_research/models/v4/dataset.py`, lines 28--40). This label is
not an independently sensed physical phase.

## Accepted production episodes

The production compilation command and manifest are
`outputs/unified_adapter_v1/canonical_compilation_20261017/manifest.json` and
the two `factory_command.txt` files. It retained two fresh, strict, right-arm
center episodes:

| Episode | Producer | Source frames | Canonical rows | Complete V4 rows | Max lift | Hold |
|---|---|---:|---:|---:|---:|---:|
| `kimodo_unified_center_shared_v2` | Kimodo | 810 | 161 | 152 | 0.157920 m | 4.895 s |
| `ardy_unified_center_shared_v4` | ARDY | 810 | 161 | 152 | 0.144612 m | 4.855 s |

Both passed the reference, renderer, physical, RGB, encoder, decoder,
command-trace, state-transition, no-off-by-one, future-target, schema, and
dataset gates. Each has 3,240 active physics steps, 810 control ticks, 816
native decisions, 162 RGB frames, zero RGB pixel error, zero encoder
reconstruction error, decoder error at most 7.63e-6, and zero reconstructed
state error. The new mixed V4 dataset therefore contains 304 complete samples
(152 from each episode); the historical 304-row mixed dataset is explicitly
excluded. Source hashes and sample hashes are recorded in
`outputs/unified_adapter_v1/canonical_compilation_20261017/manifest.json` and
the per-episode `validation_summary.json` files.

## Dataset aggregation and immutability

`data_factory/dataset/dataset.py` consumes only accepted full-retention
episodes. It rejects failed episodes and unsupported task arms, enforces an
episode-disjoint split, and serializes existing CanonicalV4 tensors exactly;
it never fabricates horizons or labels (lines 31--70 and 72--150). Final V4
arrays are checked for exact keys, shapes, dtypes, finite values, contiguous
episode ranges, and a sealed SHA-256 completion record (lines 152--170 and
`data_factory/dataset/contracts.py`, lines 46--57).

The resulting data factory contract is therefore stronger than a conventional
row-wise export: an example is publishable only when source semantics, native
timing, visual identity, SONIC reconstruction, physical task success, target
alignment, and artifact hashes all agree.
