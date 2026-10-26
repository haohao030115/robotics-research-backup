#!/usr/bin/env bash
set -euo pipefail

RESEARCH_ROOT="${RESEARCH_ROOT:-/workspace/group2}"
ROOT="${RESEARCH_ROOT}/research/ardy/online/qwen"
FUYUHAN="${RESEARCH_ROOT}/workspace/fuyuhan"
QWEN_MODEL="${QWEN_MODEL:-${RESEARCH_ROOT}/workspace/ljy/models/Qwen3-VL-2B-Instruct}"
export PYTHONPATH="$ROOT:$FUYUHAN/GR00T-WholeBodyControl:$FUYUHAN/ardy"
export ARDY_REPO="$FUYUHAN/ardy"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/root/.Xauthority}"

RUN_DIR="${RUNS_ROOT:-${RESEARCH_ROOT}/outputs/ardy_online}/fixed_stage_$(date +%Y%m%d_%H%M%S)"

if [[ ! -d "$QWEN_MODEL" ]]; then
  echo "ERROR: local Qwen model directory not found: $QWEN_MODEL" >&2
  exit 2
fi

python -m qwen_ardy.fixed_stage_grasp \
  --task-prompt "Please grasp the wooden block and lift it" \
  --qwen-model "$QWEN_MODEL" \
  --qwen-device auto \
  --source "$FUYUHAN/GR00T-WholeBodyControl/gear_sonic_deploy/reference/ardy_grasp_sides/target_block" \
  --scene "$FUYUHAN/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof_blocks_ardy_sides.xml" \
  --output-dir "$RUN_DIR" \
  --sonic-root "$FUYUHAN/GR00T-WholeBodyControl" \
  --viewer

echo "Result: $RUN_DIR/result.json"
