#!/usr/bin/env bash
set -u

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fail=0
say() { printf '%-7s %s\n' "$1" "$2"; return 0; }

if [[ -f /.dockerenv || "${container:-}" != "" ]]; then say PASS "runtime: container"; else say WARNING "runtime: host; use va-train for MUSA training/native work"; fi
if [[ "$root" == /workspace/group2* ]]; then say PASS "workspace: $root"; else say WARNING "workspace: $root (host path; container path is /workspace/group2)"; fi
for p in canonical_va va_training dl-group2 GR00T-WholeBodyControl outputs; do
  if [[ -d "$root/$p" ]]; then say PASS "tree: $p"; else say FAIL "missing tree: $p"; fail=1; fi
done
for p in canonical_va/rendering.py canonical_va/graphics_profile.json outputs/mujoco_gpu_rendering/enable_mthreads_egl.sh; do
  if [[ -f "$root/$p" ]]; then say PASS "file: $p"; else say FAIL "missing file: $p"; fail=1; fi
done

py="${PYTHON:-python3}"
if command -v "$py" >/dev/null 2>&1; then
  say PASS "python: $($py -c 'import sys; print(sys.executable)' 2>/dev/null || echo unavailable)"
  if "$py" - <<'PY' >/tmp/codex_env_torch.$$ 2>&1
import torch
print(torch.__version__)
print(__import__('torch_musa').__version__)
print(torch.musa.is_available(), torch.musa.device_count())
PY
  then say PASS "torch/MUSA: $(tr '\n' ' ' </tmp/codex_env_torch.$$)"; else say WARNING "torch/MUSA import unavailable in this interpreter; check va-train"; fi
  rm -f /tmp/codex_env_torch.$$
else say FAIL "python3 not found"; fi

if [[ -f "$root/canonical_va/graphics_profile.json" ]]; then
  say PASS "graphics profile: present (run --egl for live probe)"
fi
if command -v unshare >/dev/null 2>&1; then say PASS "DDS capability: unshare available (not exercised)"; else say WARNING "DDS capability: unshare unavailable"; fi
printf 'result: %s\n' "$([[ $fail -eq 0 ]] && echo PASS || echo FAIL)"
exit "$fail"
