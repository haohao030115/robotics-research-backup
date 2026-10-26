#!/usr/bin/env bash
set -euo pipefail
PROJECT=/workspace/group2/va_training
ROOT=/workspace/group2
OUT=${1:-/workspace/group2/outputs/va_v3_repro}
CHECKPOINT=$OUT/train/best.pt
GL=$ROOT/outputs/mujoco_gpu_rendering/enable_mthreads_egl.sh
run() { docker exec va-train env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$@"; }
physics() { run unshare --net bash -c 'ip link set lo up; source "$1"; shift; exec "$@"' bash "$GL" "$@"; }
run python3 -B "$PROJECT/train_token_v3.py" --config "$PROJECT/configs/token_v3.json" --output "$OUT/train"
run python3 -B "$PROJECT/scripts/evaluate_token_v3.py" --checkpoint "$CHECKPOINT" --output "$OUT/offline_val.json"
for device in cpu musa; do
  run python3 -B "$PROJECT/scripts/benchmark_token_v3.py" --checkpoint "$CHECKPOINT" --device "$device" --output "$OUT/benchmark_$device.json"
done
# Strict timing failures intentionally return 2 and retain the failure report.
status=0
physics python3 -B "$PROJECT/scripts/run_token_v3_closed_loop.py" --checkpoint "$CHECKPOINT" \
  --episode-id xy_random_v1_random_0000 --device musa --output "$OUT/strict" || status=$?
if [ "$status" -ne 0 ] && [ "$status" -ne 2 ]; then exit "$status"; fi
# Always distinguish the full failure-diagnostic trajectory from strict mode.
status=0
physics python3 -B "$PROJECT/scripts/run_token_v3_closed_loop.py" --checkpoint "$CHECKPOINT" \
  --episode-id xy_random_v1_random_0000 --device musa --output "$OUT/learner" --diagnostic-continue-on-miss --vision-worker thread --render-height 240 --render-width 320 || status=$?
if [ "$status" -ne 0 ] && [ "$status" -ne 2 ]; then exit "$status"; fi
# Existing independent GT-token/GT-hand physical protocol, unchanged.
physics python3 -B "$PROJECT/scripts/diagnose_token_v2.py" \
  --checkpoint "$ROOT/outputs/va_v2/heldout/checkpoints/latest.pt" --episode-id xy_random_v1_random_0000 \
  --oracle-report "$ROOT/outputs/va_v2/oracle_0000/report.json" \
  --overfit-report "$ROOT/outputs/va_v2/overfit_0000/report.json" \
  --offline-report "$ROOT/outputs/va_v2/heldout/report.json" --condition A --output "$OUT/expert"
run bash -c 'source "$1"; shift; exec "$@"' bash "$GL" \
  python3 -B "$PROJECT/scripts/render_token_v3_video.py" --learner "$OUT/learner" --expert "$OUT/expert" --output "$OUT/videos"
