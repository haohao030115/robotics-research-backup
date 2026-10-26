# Canonical expert replay and VA compiler

Production uses the **original native SONIC controller and original MuJoCo DDS runtime**. There is no synchronous fallback. See [NATIVE_EXECUTION_AUDIT.md](NATIVE_EXECUTION_AUDIT.md) for the complete execution-chain and reference-consumption audit, source equivalence evidence, instrumentation diff and reproducible commands.

## Production entry

From an environment with the existing SONIC/DDS/MuJoCo dependencies:

```bash
cd /workspace/group2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
python3 -B build_canonical_va_episode.py \
  --reference /path/to/canonical_reference \
  --output /path/to/fresh_episode \
  --backend native --gpu 0 --mode production
```

The isolated logging executable must first be built with `python3 -B -m canonical_va.native_build`. This copies original sources, records the precise logging-only patch and builds a separate executable; the shared original executable is unchanged. Existing build/output directories are not overwritten.

Do not share the DDS network namespace with another replay. A busy shared domain is rejected before physics. In `va-train`, `unshare --net` plus `ip link set lo up` creates an isolated loopback environment; see the audit for the full actual test command. The controller retains original domain 0, topics and transport.

## Input and producer independence

`reference.npz`, `metadata.json`, `provenance.json`, `reference_validation.json` are required. Required numeric finite arrays are `time_s [T]`, `body_ref_q [T,29]`, `body_ref_dq [T,29]`, `root_ref_pos [T,3]`, `root_ref_quat [T,4]`, `hand_ref_q [T,14]`. Canonical joint names/order, explicit units, complete reference, normalized wxyz quaternions and strictly increasing 50 Hz zero-based time are validated before starting physics. Supported producer labels are kimodo, rule_based and ardy; execution semantics never branch on that label.

The existing schema also supports legacy producer schema names, `root_quat_order`, legacy unit keys and `cube_initial_pos`. In audit mode original source files are retained byte-for-byte; production retains their hashes, and required reference arrays are serialized losslessly for the original native CSV loader. No source IK, retargeting, hand closure adjustment or trajectory regeneration occurs.

`root_ref_pos[0]` is used for reset; later root positions are retained in native reference files but are **not consumed by the mode-0 encoder**. Body q/dq and root quaternion use the original native encoder construction. Hand reference drives the existing independent named-joint PD, at the original playback cursor time. Details and source positions are in the consumption audit.

## Native identity and forensic truth

PRE_ACTION observations are exact physical states published to DDS and actually selected by the native controller. They may precede command application by multiple physical steps; inference and DDS latency are retained, not hidden.

`LowState.reserve[0]` tags the snapshot. Native decision traces save exact encoder input, current 64D post-FSQ token, exact decoder input/history, previous action, current raw action and mapped motor fields. `LowCmd.reserve` carries decision and publication identities. The original simulator lock selects the actual received message; the recorder retains which physics steps consumed it. The original publisher remains asynchronous at 500 Hz, controller at 50 Hz, and simulator at 200 Hz. Zero or variable consumption intervals are recorded honestly, and unconsumed active commands fail validation.

Original startup balance WBC, native initialization, standby ticks and hand timing are preserved. Physics records save full integration states, applied control and actual hand cursor at every step. RGB is losslessly rendered offline from the exact observation state, then independently re-rendered for verification.

Outputs include raw native binary traces/receipt records and physics, plus canonical observations/references/encoder/decoder/commands/physics/timeline tables, PNGs, provenance, validation reports and REPORT.md. The original executable and source hashes, instrumented build hash, exact patch, network namespace and launch command are retained.

## Gates and VA compiler

Forensic checks include RGB pixel reconstruction, independent frozen ONNX reconstruction of same-decision encoder/decoder outputs, exact physical-snapshot-to-native-input linkage, previous/current-action separation, DDS publication/receipt/consumption edges, original `TabletopDDS.step()` reconstruction, canonical physical metrics and ID-based future mapping.

VA uses RGB at 10 Hz observation anchors, three control observations ending at the anchor, and future offsets `[2,3,4,5,6]` on the 50 Hz reference/control timeline. Body token IDs and hand intent IDs resolve to the same future reference IDs. Start history repeats the first active observation with an explicit mask. Reset/standby/inactive transitions and incomplete tails are excluded. No row-number matching is used.

Only physical success AND dataset validity permits export and a training manifest. `build_va_samples.py --episode EPISODE_ROOT --output NEW_DATASET` is the separate compiler; it reruns the native forensic gates. Use the batch and aggregation entry points below for production.

## Historical code

The old synchronous replay and verifier are archived as non-executable `.py.txt` files under `diagnostics/`. `runtime.py` contains snapshot/camera helpers and historical diagnostic methods; `policy.py` retains frozen-model/parameter readers and the historical builder. Production never calls its Policy or Runtime control methods. No SONIC observation, history, inference scheduling, action mapping or PD implementation in those old Python classes governs native replay.

The old `pilot_0000` report is retained as the prior failed synchronous experiment. It must not be confused with the new native test.

## Tests

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
python3 -B -m unittest discover -s canonical_va/tests -v
```

Tests cover schema rejection, producer-independent mapping, shuffled table rows, future shifts, boundaries, failed export gates, native binary trace layout, real receipt identity/causality, production backend selection and the additive source instrumentation patch. Tests do not execute a new physical rollout.

Audit-only `--recorded-episode PREVIOUS_NATIVE_EPISODE` revalidates immutable native evidence and compiles into a fresh output without new physics. It checks the original reference and metadata bytes, hashes every reused file, verifies the recording-time archived implementation against its original provenance, and records the new verifier's hashes separately. This supports verification-code fixes without retuning or rerunning an already successful physical replay.

## Modes, graphics and batch production (v2)

`--mode audit` (default) retains complete DDS traces, snapshots, recording source, images and encoder/decoder/physics/pixel reconstruction. `--mode production` always starts a new native execution. It validates the reference, completion, physical success, ID/timing/provenance and exported samples. Full reconstruction runs for a new code/runtime/renderer qualification, a failure, or deterministic sampled QA (`--qa-rate 0.1`). Failed outputs retain evidence. Successful production outputs contain only:

```
va_dataset/samples.npz
training_manifest.json
validation_summary.json
provenance.json
```

The NPZ preserves all original numeric arrays and adds `rgb [N,240,320,3] uint8` and lossless `mapping_json`; compact consumers read RGB directly from NPZ. Future offsets remain +2..+6; hands remain all 14 canonical joints. `task_arm` accepts left/right (legacy default right); manifests include case, XY, yaw, sample count, versions, source hashes and optional `--position-bin-size 0.05` bins.

`--render-profile auto` selects the checksum-verified MThreads graphics stage described in `graphics_profile.json`, then reexecutes Python so the dynamic loader receives the library paths. `current` probes the existing environment. `mthreads` requires the verified stage. Each invocation actually renders and records GL vendor/renderer/version. Production rejects llvmpipe/softpipe/software rasterizer unless `--allow-software-rendering` is explicitly provided; audit records a warning. Driver warnings remain visible. The stage is an existing audited host-driver copy under `outputs/mujoco_gpu_rendering/driver_stage`, not a portable bundled driver; a new host requires a reviewed graphics profile.

```bash
cd /workspace/group2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -B -m canonical_va.batch \
  --references-root /path/to/references --output-root /path/to/batch \
  --backend native --gpu 0 --mode production --qa-rate 0.1

python3 -B -m canonical_va.aggregate \
  --roots /path/to/batch /path/to/older_batch --output /path/to/fresh_dataset

python3 -B -m canonical_va.rendering --profile auto --mode production \
  --output /path/to/renderer.json
```

Batch is serial and creates one network namespace per episode. It continues after failures, writes explicit reasons and PASS/FAIL/SKIP totals, and skips only accepted, checksum-verified outputs with identical source bytes and episode identity. Existing failures are retained; retry into a fresh output root. It returns exit code 2 if any episode fails. Aggregation reports duplicates, keeps the first accepted compatible episode in root order (or fails with `--duplicates error`), rejects incompatible numeric dtypes/keys/trailing shapes, and losslessly widens Unicode IDs/JSON. Global manifests retain each source path and sample range. Legacy unversioned episodes are labeled as such.

`source_policy.json` pins critical runtime sources and AST symbols: controller/models, DDS, joint ordering, scene/camera/physics, observation and sample mapping. Mixed evaluator modules keep mapping symbols and module bindings pinned while evaluator/helper changes warn and record actual hashes. CLI/forensic/report changes warn; any tracked source change invalidates full-QA qualification. The manifest records compiler version, runtime version/hash and actual source hashes. The policy is a reviewed baseline; never regenerate it merely to bypass a failure.

Stage timings include startup, native rollout, RGB, trace materialization, compilation, validation, disk writing and total. Nested serialization inside existing rollout/materialization stays charged to its enclosing stage; measured disk writing covers PNG/sample packaging and compaction. See the dated productionization report for actual measurements and remaining renderer photometric differences.
