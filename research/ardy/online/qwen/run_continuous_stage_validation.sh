#!/usr/bin/env bash
set -euo pipefail

RESEARCH_ROOT="${RESEARCH_ROOT:-/workspace/group2}"
ROOT="${RESEARCH_ROOT}/research/ardy/online/qwen"
SONIC="${SONIC_ROOT:-${RESEARCH_ROOT}/GR00T-WholeBodyControl}"
RUN_DIR="${1:-${RUNS_ROOT:-${RESEARCH_ROOT}/outputs/ardy_online}/continuous_stage_$(date +%Y%m%d_%H%M%S)}"

cd "$ROOT"
export PYTHONPATH="$ROOT:$SONIC${PYTHONPATH:+:$PYTHONPATH}"
export SONIC_ROOT="$SONIC"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python -m qwen_ardy.continuous_stage_validation \
  --arms-at-sides \
  --robot-x 0.24 --robot-y -0.03 \
  --hand-kp 6 \
  --motion-dir "$SONIC/gear_sonic_deploy/reference/ardy_grasp_sides" \
  --scene "$SONIC/gear_sonic/data/robot_model/model_data/g1/scene_43dof_blocks_ardy_sides.xml" \
  --output-dir "$RUN_DIR" \
  "${@:2}"

echo "Result: $RUN_DIR/result.json"
