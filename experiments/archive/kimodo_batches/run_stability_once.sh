#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/assets/unitree_sdk2_python:$ROOT/code:$ROOT/code/canonical/scripts"
export SONIC_REPO=/workspace/group2/GR00T-WholeBodyControl
export CHECKPOINT_DIR=/workspace/group2/kimodo-t6/g1-integration/checkpoints
export TEXT_ENCODERS_DIR="$ROOT/assets/text_encoders"
export TEXT_ENCODER_MODE=local TEXT_ENCODER_DEVICE=cpu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$ROOT/assets/hf_cache"
export XDG_CACHE_HOME="$ROOT/qualification/cache"
export TMPDIR=/tmp/kimodo_task8_dev_v3_run05
export TORCH_HOME="$ROOT/qualification/cache/torch"
exec /workspace/group2/workspace/amy/kimodo-work/.venv/bin/python "$ROOT/code/development_full_once.py"
