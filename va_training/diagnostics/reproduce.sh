#!/usr/bin/env bash
# Host shell. Pass a fresh CONTAINER output directory as argument 1.
set -euo pipefail
ROOT=${VA_WORKSPACE_IN_CONTAINER:-/workspace/group2}
PROJECT=$ROOT/va_training
OUT=${1:-$ROOT/outputs/va_v2_diagnosis_repro}
BASE=$ROOT/outputs/va_v2
CHECKPOINT=$BASE/heldout/checkpoints/latest.pt
ORACLE=$BASE/oracle_0000/report.json
OVERFIT=$BASE/overfit_0000/report.json
OFFLINE=$BASE/heldout/report.json
EPISODE=xy_random_v1_random_0000
HELDOUT=xy_random_v1_random_0001
GL=$ROOT/outputs/mujoco_gpu_rendering/enable_mthreads_egl.sh
run() { docker exec va-train env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$@"; }
physics() { run unshare --net bash -c 'ip link set lo up; source "$1"; shift; exec "$@"' bash "$GL" "$@"; }
isolate() { run unshare --net bash -c 'ip link set lo up; exec "$@"' bash "$@"; }
common=(--checkpoint "$CHECKPOINT" --episode-id "$EPISODE" --oracle-report "$ORACLE" --overfit-report "$OVERFIT" --offline-report "$OFFLINE")

# Phase 1: equal-condition factorial; abort other conditions if A cannot execute.
physics python3 -B "$PROJECT/scripts/diagnose_token_v2.py" "${common[@]}" --condition A --output "$OUT/phase1/A"
run python3 -B -c 'import json,sys; assert json.load(open(sys.argv[1]))["closed_loop_pass"], "Stop: oracle task failed"' "$OUT/phase1/A/report.json"
for condition in B C D; do
  physics python3 -B "$PROJECT/scripts/diagnose_token_v2.py" "${common[@]}" --condition "$condition" --output "$OUT/phase1_$condition"
done

# Phase 2, 7 (frozen checkpoint comparison), 10: all held-out samples and exact decoder.
run python3 -B "$PROJECT/scripts/analyze_token_control.py" --checkpoint "$CHECKPOINT" --oracle-episode "$EPISODE" --output "$OUT/offline_control_verified"

# Phase 3 plus pure scheduling controls. Separate processes avoid DDS reinitialization.
isolate python3 -B "$PROJECT/scripts/token_corruption_suite.py" "${common[@]}" \
  --predictions "$OUT/offline_control_verified/nominal_predictions.npz" \
  --residuals "$OUT/offline_control_verified/heldout_predictions.npz" \
  --output "$OUT/corruption" --seed 20261005

# Phase 4: overfit, train (already phase1 D), and held-out, through the same live path.
physics python3 -B "$PROJECT/scripts/diagnose_token_v2.py" \
  --checkpoint "$BASE/overfit_0000/checkpoints/latest.pt" --episode-id "$EPISODE" \
  --oracle-report "$ORACLE" --overfit-report "$OVERFIT" --offline-report "$OFFLINE" \
  --experiment learned --output "$OUT/overfit_closed_loop"
physics python3 -B "$PROJECT/scripts/diagnose_token_v2.py" --checkpoint "$CHECKPOINT" \
  --episode-id "$HELDOUT" --oracle-report "$ORACLE" --overfit-report "$OVERFIT" \
  --offline-report "$OFFLINE" --experiment learned --output "$OUT/heldout_closed_loop"

# Phase 6: native timestamps, all 800 body DDS target matches, and hand application time.
physics python3 -B "$PROJECT/scripts/diagnose_token_v2.py" "${common[@]}" \
  --condition D --record-dds --output "$OUT/timing_dds"

# Phase 7: controlled independent retraining, with retrospective offset 0 isolated.
run python3 -B "$PROJECT/scripts/train_token_alignment.py" --config "$PROJECT/configs/token_v2.json" \
  --oracle-report "$ORACLE" --overfit-report "$OVERFIT" --offsets 0 2 3 --output "$OUT/alignment_retraining"

# Phase 8: run only after the timing audit establishes missed next-control deadlines.
run python3 -B -c 'import json,sys; t=json.load(open(sys.argv[1])); a=[x for x in t if x["anchor"]>=0 and "applied_index" in x]; assert a and sum(x["applied_index"]>x["execution_index"] for x in a)/len(a)>.5' "$OUT/timing_dds/timing.json"
run python3 -B "$PROJECT/train_token_v2.py" --config "$PROJECT/configs/token_v2_diagnostic_chunk5.json" \
  --oracle-report "$ORACLE" --overfit-report "$OVERFIT" --stage heldout --output "$OUT/chunk5_training"
for episode in "$EPISODE" "$HELDOUT"; do
  physics python3 -B "$PROJECT/scripts/diagnose_token_v2.py" \
    --checkpoint "$OUT/chunk5_training/checkpoints/latest.pt" --episode-id "$episode" \
    --oracle-report "$ORACLE" --overfit-report "$OVERFIT" --offline-report "$OUT/chunk5_training/report.json" \
    --experiment learned --chunk-mode --output "$OUT/chunk5_$episode"
done

# Privileged hand-state counterfactual: still execute predictions, change current observation only.
physics python3 -B "$PROJECT/scripts/diagnose_token_v2.py" "${common[@]}" \
  --condition B --expert-hand-observation --output "$OUT/hand_observation_counterfactual"

# Phase 9: minimal separate corrective-label prototype, no DAgger training/aggregation.
run bash -c 'source "$1"; shift; exec "$@"' bash "$GL" \
  python3 -B "$PROJECT/scripts/prototype_token_relabel.py" --rollout "$OUT/phase1_D" \
  --target-offset 2 --output "$OUT/relabel_prototype"

# Independent plot dependencies do not replace the container's scientific runtime.
run python3 -m pip install --target "$OUT/plot_dependencies" matplotlib
run env PYTHONPATH="$OUT/plot_dependencies" python3 -B "$PROJECT/scripts/analyze_token_rollouts.py" \
  --rollouts "$OUT/phase1/A" "$OUT/phase1_B" "$OUT/phase1_C" "$OUT/phase1_D" \
  "$OUT/overfit_closed_loop" "$OUT/heldout_closed_loop" "$OUT/chunk5_$EPISODE" "$OUT/chunk5_$HELDOUT" \
  "$OUT/hand_observation_counterfactual" "$OUT/timing_dds" --output "$OUT/trajectory_analysis"
run env VA_V2_REAL_CONFIG="$PROJECT/configs/token_v2.json" bash -c \
  'cd /workspace/group2/va_training; set -o pipefail; python3 -B -m unittest discover -s tests -v 2>&1 | tee "$1"' bash "$OUT/regression_tests.log"
run env PYTHONPATH="$OUT/plot_dependencies" python3 -B "$PROJECT/scripts/report_token_diagnosis.py" \
  --root "$OUT" --output "$OUT/REPORT.md"
