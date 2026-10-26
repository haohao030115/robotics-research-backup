#!/usr/bin/env bash
set -euo pipefail

exec 9>/tmp/ljy_method_b_online_hold.lock
if ! flock -n 9; then
  echo "已有一条 Method B 仿真正在运行；请等待它结束，不要并行启动。" >&2
  exit 2
fi

RESEARCH_ROOT="${RESEARCH_ROOT:-/workspace/group2}"
ROOT="${RESEARCH_ROOT}/research/ardy/online/qwen"
SONIC_ROOT="${SONIC_ROOT:-${RESEARCH_ROOT}/GR00T-WholeBodyControl}"
ARDY_REPO="${ARDY_REPO:-${RESEARCH_ROOT}/workspace/fuyuhan/ardy}"
QWEN_MODEL="${QWEN_MODEL:-${RESEARCH_ROOT}/workspace/ljy/models/Qwen3-VL-2B-Instruct}"
cd "$ROOT"
export PYTHONPATH="$ROOT:$SONIC_ROOT"
export SONIC_DEPLOY="${SONIC_DEPLOY:-${RESEARCH_ROOT}/workspace/ljy/sonic_hot_reload/gear_sonic_deploy}"
# noVNC desktop is the existing TigerVNC X server on :1.  Physics/rendering
# stays on EGL; an OpenCV monitor window presents the live MuJoCo camera.
export DISPLAY="${DISPLAY:-:1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
ARDY_GPU="${ARDY_GPU:-5}"

RUN_ID="method_b_online_hold_$(date +%Y%m%d_%H%M%S)"
python -m qwen_ardy.method_b_online_continuation \
  --source "$SONIC_ROOT/gear_sonic_deploy/reference/ardy_grasp_sides/target_block" \
  --reference-root "${RUNS_ROOT:-${RESEARCH_ROOT}/outputs/ardy_online}/${RUN_ID}_refs" \
  --output-dir "${RUNS_ROOT:-${RESEARCH_ROOT}/outputs/ardy_online}/${RUN_ID}" \
  --scene "$SONIC_ROOT/gear_sonic/data/robot_model/model_data/g1/scene_43dof_blocks_ardy_sides.xml" \
  --ardy-repo "$ARDY_REPO" \
  --qwen-model "$QWEN_MODEL" \
  --qwen-device musa:7 \
  --ardy-gpu "$ARDY_GPU" \
  --viewer \
  "$@"

