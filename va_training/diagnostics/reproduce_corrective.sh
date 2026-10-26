#!/usr/bin/env bash
# Host shell; all outputs separate from original expert data and checkpoints.
set -euo pipefail
ROOT=/workspace/group2
PROJECT=$ROOT/va_training
DIAG=$ROOT/outputs/va_v2_diagnosis
OUT=${1:-$ROOT/outputs/va_v2_corrective}
GL=$ROOT/outputs/mujoco_gpu_rendering/enable_mthreads_egl.sh
CHECKPOINT=$DIAG/chunk5_training/checkpoints/latest.pt
run() { docker exec va-train env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$@"; }
physics() { run unshare --net bash -c 'ip link set lo up; source "$1"; shift; exec "$@"' bash "$GL" "$@"; }
if [[ ${CORRECTIVE_SKIP_PREPARE:-0} != 1 ]]; then
  run python3 -B "$PROJECT/scripts/prepare_corrective_experiment.py" --prototype "$DIAG/relabel_prototype" \
    --rollout "$DIAG/phase1_D" --checkpoint "$CHECKPOINT" --output "$OUT/round1"
fi
if [[ ! -f /home/group2/${OUT#"$ROOT/"}/round1/rgb_audit.json ]]; then
  run bash -c 'source "$1"; shift; exec "$@"' bash "$GL" \
    python3 -B "$PROJECT/scripts/audit_corrective_rgb.py" --prototype "$DIAG/relabel_prototype" \
    --rollout "$DIAG/phase1_D" --output "$OUT/round1/rgb_audit.json"
fi
models=(original nominal_ft body25 body50 hand25 hand50 combined25 combined50 combined25_scratch)
for model in "${models[@]}"; do
  branch=none; fraction=0; extra=(); steps=500; lr=.0001
  case "$model" in
    original) extra=(--evaluate-only);;
    body25) branch=body; fraction=.25;; body50) branch=body; fraction=.5;;
    hand25) branch=hand; fraction=.25;; hand50) branch=hand; fraction=.5;;
    combined25) branch=combined; fraction=.25;; combined50) branch=combined; fraction=.5;;
    combined25_scratch) branch=combined; fraction=.25; steps=2500; lr=.001; extra=(--from-initialization);;
  esac
  if [[ ! -f /home/group2/${OUT#"$ROOT/"}/models/$model/report.json ]]; then
    run python3 -B "$PROJECT/scripts/train_corrective_experiment.py" --checkpoint "$CHECKPOINT" \
      --corrective "$OUT/round1/corrective_k5.npz" --output "$OUT/models/$model" \
      --branch "$branch" --fraction "$fraction" --steps "$steps" --lr "$lr" "${extra[@]}"
  fi
done
if [[ ${CORRECTIVE_TRAIN_ONLY:-0} == 1 ]]; then exit 0; fi
for model in "${models[@]}"; do
  for episode in xy_random_v1_random_0000 xy_random_v1_random_0001; do
    if [[ ! -f /home/group2/${OUT#"$ROOT/"}/rollouts/${model}_${episode}/report.json ]]; then
      physics python3 -B "$PROJECT/scripts/diagnose_token_v2.py" --checkpoint "$OUT/models/$model/checkpoints/latest.pt" \
        --episode-id "$episode" --oracle-report "$ROOT/outputs/va_v2/oracle_0000/report.json" \
        --overfit-report "$ROOT/outputs/va_v2/overfit_0000/report.json" --offline-report "$OUT/models/$model/report.json" \
        --experiment learned --chunk-mode --output "$OUT/rollouts/${model}_${episode}"
    fi
  done
done
run env PYTHONPATH="$DIAG/plot_dependencies" python3 -B "$PROJECT/scripts/analyze_corrective_experiment.py" --root "$OUT"
# At most one additional dataset, and only after measured round1 improvement.
best=$(run python3 -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["round2_model"] or "")' "$OUT/analysis/report.json")
if [[ -n "$best" && ! -f /home/group2/${OUT#"$ROOT/"}/round2/audit.json ]]; then
  physics python3 -B "$PROJECT/scripts/diagnose_token_v2.py" --checkpoint "$OUT/models/$best/checkpoints/latest.pt" \
    --episode-id xy_random_v1_random_0000 --oracle-report "$ROOT/outputs/va_v2/oracle_0000/report.json" \
    --overfit-report "$ROOT/outputs/va_v2/overfit_0000/report.json" --offline-report "$OUT/models/$best/report.json" \
    --experiment learned --chunk-mode --output "$OUT/round2_source_rollout"
  run bash -c 'source "$1"; shift; exec "$@"' bash "$GL" \
    python3 -B "$PROJECT/scripts/prototype_token_relabel.py" --rollout "$OUT/round2_source_rollout" --output "$OUT/round2_prototype"
  run python3 -B "$PROJECT/scripts/prepare_corrective_experiment.py" --prototype "$OUT/round2_prototype" \
    --rollout "$OUT/round2_source_rollout" --checkpoint "$OUT/models/$best/checkpoints/latest.pt" --output "$OUT/round2" --round corrective_round2
fi
run env VA_V2_REAL_CONFIG="$PROJECT/configs/token_v2.json" bash -c \
  'cd /workspace/group2/va_training; set -o pipefail; python3 -B -m unittest discover -s tests -v 2>&1 | tee "$1"' bash "$OUT/regression_tests.log"
run python3 -B "$PROJECT/scripts/report_corrective_experiment.py" --root "$OUT"
