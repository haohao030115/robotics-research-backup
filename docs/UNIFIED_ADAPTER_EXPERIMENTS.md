# Unified adapter experiments

The implementation includes a deterministic offline smoke experiment for adapter and hand behavior. Native experiments require the validated `va-train` container, MTT renderer, and isolated DDS namespace described in `docs/ENVIRONMENT.md`. In the current validated container, `scripts/check_environment.sh` passes and the live EGL probe identifies `MTT S4000` with `software=false`; the Mesa `musa_dri.so` loader warning is an optional stale helper. The independent non-rendering physics capability result is preserved at `outputs/unified_adapter_v2/joint_optimization_20261017/analysis/nonrender_physics_probe_v1.json`.

Run the pure-array check in the validated container:

```bash
docker exec va-train bash -lc 'cd /workspace/group2 && PYTHONPATH=. python3 - <<"PY"
import numpy as np
from research.shared_adapter import UnifiedMotionAdapter, HandMotionGenerator
q=np.zeros((810,29)); r=UnifiedMotionAdapter().adapt(q, task={"cube_xy":[.03,-.02]}, seed=7, producer="kimodo")
h,m=HandMotionGenerator().generate(np.arange(810)/50, seed=7)
print(r.corrections); print(h.shape, m)
PY'
```

Physical success, spatial success rates, SONIC tracking, and accepted V4 throughput must be recorded by the native production runner before claiming those measurements.
