# Unified adapter implementation

`research/shared_adapter` is the producer-neutral boundary used by Kimodo and ARDY. Native outputs are converted to the canonical 50 Hz body array before `UnifiedMotionAdapter.adapt()` is called. The adapter records native, canonical, planned, and final stages in `motion_stages.npz` and records bounded task correction and a configuration hash in `adapter_provenance.json`.

The correction is derived from declared cube displacement relative to the reference scene. It is bounded by the reach corridor and blended with native motion; an out-of-corridor correction is rejected. Legacy behavior remains the default and can be selected explicitly with `adapter="legacy"` in either workflow facade. Kimodo export accepts `--adapter unified`; ARDY accepts the same option through `run_pipeline.py`.

Native SONIC, MuJoCo, DDS, joint order, and V4 contracts remain unchanged.
