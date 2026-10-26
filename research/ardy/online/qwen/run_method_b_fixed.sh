#!/usr/bin/env bash
set -euo pipefail

RESEARCH_ROOT="${RESEARCH_ROOT:-/workspace/group2}"
ROOT="${RESEARCH_ROOT}/research/ardy/online/qwen"
SONIC="${SONIC_ROOT:-${RESEARCH_ROOT}/GR00T-WholeBodyControl}"
ARDY="${ARDY_REPO:-${RESEARCH_ROOT}/workspace/fuyuhan/ardy}"
STAMP=$(date +%Y%m%d_%H%M%S)
BUILD_DIR="${1:-${RUNS_ROOT:-${RESEARCH_ROOT}/outputs/ardy_online}/method_b_$STAMP}"
VALIDATION_DIR="${BUILD_DIR}_validation"

cd "$ROOT"
export PYTHONPATH="$ROOT:$SONIC${PYTHONPATH:+:$PYTHONPATH}"
export SONIC_ROOT="$SONIC"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python -m qwen_ardy.build_method_b_reference \
  --source "$SONIC/gear_sonic_deploy/reference/ardy_grasp_sides/target_block" \
  --scene "$SONIC/gear_sonic/data/robot_model/model_data/g1/scene_43dof_blocks_ardy_sides.xml" \
  --ardy-repo "$ARDY" \
  --sonic-root "$SONIC" \
  --qwen-model "$ROOT/../models/Qwen3-VL-2B-Instruct" \
  --qwen-device musa:7 \
  --ardy-gpu 6 \
  --output-dir "$BUILD_DIR"

python -m qwen_ardy.continuous_stage_validation \
  --arms-at-sides --robot-x 0.24 --robot-y -0.03 --hand-kp 6 \
  --motion-dir "$BUILD_DIR/reference" \
  --scene "$SONIC/gear_sonic/data/robot_model/model_data/g1/scene_43dof_blocks_ardy_sides.xml" \
  --output-dir "$VALIDATION_DIR" \
  "${@:2}"

echo "Method B plan: $BUILD_DIR/method_b_plan.json"
echo "Simulation result: $VALIDATION_DIR/result.json"
