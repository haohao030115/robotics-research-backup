#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export ARDY_PHYSICAL_GPU="${ARDY_PHYSICAL_GPU:-7}"
export MUSA_VISIBLE_DEVICES="$ARDY_PHYSICAL_GPU"
export CUDA_VISIBLE_DEVICES="$ARDY_PHYSICAL_GPU"
# Only this child simulator and SONIC use the new loopback network namespace.
# The replay implementation is owned by runtime_support; this shell file is
# the isolated-DDS compatibility entry point used by the qualification flow.
if [[ $# -ne 3 ]]; then
  echo "usage: $0 <workspace-root> <candidate> <output>" >&2
  exit 2
fi
exec unshare --net bash -c 'ip link set lo up; exec python "$1/runtime_support/replay_ardy_isolated.py" --candidate "$3" --out "$4"' bash "$TASK_ROOT" "$1" "$2" "$3"
