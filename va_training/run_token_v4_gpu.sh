#!/usr/bin/env bash
set -euo pipefail
# Run inside the existing va-train container; preserve validated graphics setup.
exec unshare --net bash -c '
    set -euo pipefail
    ip link set lo up
    source /workspace/group2/outputs/mujoco_gpu_rendering/enable_mthreads_egl.sh
    export PYTHONPATH=/workspace/group2/va_training:/workspace/group2${PYTHONPATH:+:$PYTHONPATH}
    exec "$@"
' bash python3 -B /workspace/group2/va_training/run_token_v4_closed_loop.py "$@"
