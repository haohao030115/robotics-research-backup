#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/group2
PROJECT=$ROOT/va_training
OUT=${1:-$ROOT/outputs/va_v2_video_repro}
BASE=$ROOT/outputs/va_v2
DIAG=$ROOT/outputs/va_v2_diagnosis
EPISODE=xy_random_v1_random_0000
GL=$ROOT/outputs/mujoco_gpu_rendering/enable_mthreads_egl.sh
run() { docker exec va-train env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$@"; }
physics() { run unshare --net bash -c 'ip link set lo up; source "$1"; shift; exec "$@"' bash "$GL" "$@"; }
physics python3 -B "$PROJECT/scripts/diagnose_token_v2.py" \
  --checkpoint "$DIAG/chunk5_training/checkpoints/latest.pt" --episode-id "$EPISODE" \
  --oracle-report "$BASE/oracle_0000/report.json" --overfit-report "$BASE/overfit_0000/report.json" \
  --offline-report "$DIAG/chunk5_training/report.json" --experiment learned --chunk-mode --output "$OUT/learner"
# Oracle uses existing GT/GT runner, not a learned K1 policy or a reference animation.
physics python3 -B "$PROJECT/scripts/diagnose_token_v2.py" \
  --checkpoint "$BASE/heldout/checkpoints/latest.pt" --episode-id "$EPISODE" \
  --oracle-report "$BASE/oracle_0000/report.json" --overfit-report "$BASE/overfit_0000/report.json" \
  --offline-report "$BASE/heldout/report.json" --condition A --output "$OUT/expert"
run bash -c 'source "$1"; shift; exec "$@"' bash "$GL" \
  python3 -B "$PROJECT/scripts/render_closed_loop_video.py" --learner "$OUT/learner" \
  --expert "$OUT/expert" --output "$OUT/videos"
