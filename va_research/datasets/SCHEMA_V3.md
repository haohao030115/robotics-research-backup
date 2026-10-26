# Schema-v3 adapter contract

The real export has not arrived. This is the supported model-side contract,
not a claim about the future control stack's file layout.

`data.manifest` points to `training_manifest.json`, containing an `episodes` list
or a top-level list. Preferred fields:

```json
{"episodes": [
  {"id": "episode_001", "episode_path": "episode_001/states.npz",
   "rgb_path": "episode_001/images.npz", "split": "train"},
  {"id": "episode_002", "episode_path": "episode_002/states.npz",
   "rgb_path": "episode_002/images.npz", "split": "val"}
]}
```

Paths are relative to the manifest, or absolute. A directory `episode_path`
requires explicit `data.episode_state_filename`. `rgb_path` points to an NPZ.
Legacy states/images aliases are supported; conflicting aliases are rejected.
Explicit IDs are recommended; otherwise episode_path becomes the ID.

State keys are explicit `data.keys` settings. Actual body_q [T,29] and hand_q
[T,14] supply observations. Reference body_ref_q [T,29] and hand_ref_q [T,14]
supply future targets. Timestamp units are seconds, with a 50Hz state grid.
Optional dq keys can be null; dq is not a current model input.

RGB is head_rgb [N,height,width,3], uint8 or finite float [0,1], with synchronized
capture timestamps. rgb_reference_frame_indices can be an integer list in each
manifest entry or an array in images.npz (configurable key). If both are present,
they must agree. The mapping must mean state index at RGB capture on the same
state/reference grid, not a future waypoint. Timestamps, monotonicity and bounds
are checked. Unknown clock or mapping semantics require explicit adaptation
after inspecting the real export.

At anchor a, observation is actual q[a], target is reference[a:a+20] for H=20.
Samples with a+20 > T are filtered. No action chunk crosses an episode boundary.

Splits are episode-level. Supply all explicit splits or none. source_episode_id
optionally groups fragments of the same source episode; one source group cannot
span train and val. Shared source files under different IDs are rejected. If an
exporter copies or fragments episodes, it must preserve source identity; arbitrary
new filenames alone cannot prove that recordings are distinct episodes.

Training-only normalization uses resized image channels, actual q anchors and
complete target chunks. State/action statistics are per coordinate, with separate
body/hand subsets. Overlapping target frames are weighted by sample occurrence.
Stats are stored in checkpoint and normalization.json and inverted at inference.

```bash
python3 scripts/check_dataset.py --config configs/my_real_run.json \
  --output outputs/my_real_run/sanity.json
```

The report includes RGB/state counts, valid/tail counts, first/last indices,
alignment examples, actual q/action min/max/mean/std and per-joint statistics.
Synthetic reports carry synthetic=true. Real-data verification is still pending.
