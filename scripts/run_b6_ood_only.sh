#!/usr/bin/env bash
# Continue B6 after the currently-running `global test` inference finishes.
# Remaining work per GPU/seed (11 evals):
#   global:    ood_compositional, ood_concept, ood_frame, ood_tailrisk, ood_shifted_cal_test  (5)
#   cvar_cell: test, ood_compositional, ood_concept, ood_frame, ood_tailrisk, ood_shifted_cal_test  (6)
#
# Usage: bash scripts/run_b6_ood_only.sh <gpu_id> <seed> <wait_pid>
set -u

GPU=$1
SEED=$2
WAIT_PID=$3
PY=/LOCAL2/psqhe8/anaconda3/envs/biasmap/bin/python
OUT=results/sprint3/b6_test_ood

cd /LOCAL2/psqhe8/BiasMap-CP

echo "[GPU $GPU seed $SEED] waiting for test PID $WAIT_PID to finish..." | tee -a /tmp/b6_screen_gpu${GPU}.log
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 15
done
echo "[GPU $GPU seed $SEED] test PID $WAIT_PID done. Starting remaining sweep at $(date)" | tee -a /tmp/b6_screen_gpu${GPU}.log

run_one() {
  local m=$1
  local sp=$2
  local dst=$OUT/${m}_seed${SEED}_${sp}
  if [[ -f "$dst/b6_results.json" ]]; then
    echo "[GPU $GPU] SKIP (exists): $m seed$SEED $sp" | tee -a /tmp/b6_screen_gpu${GPU}.log
    return 0
  fi
  mkdir -p "$dst"
  echo "[GPU $GPU] === $m seed$SEED $sp === $(date)" | tee -a /tmp/b6_screen_gpu${GPU}.log
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_b4_recalibration.py \
      --method "$m" --seed "$SEED" --eval_split "$sp" --output_dir "$OUT" \
      >> "$dst/run.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[GPU $GPU] FAILED rc=$rc: $m seed$SEED $sp" | tee -a /tmp/b6_screen_gpu${GPU}.log
  else
    echo "[GPU $GPU] DONE: $m seed$SEED $sp" | tee -a /tmp/b6_screen_gpu${GPU}.log
  fi
}

# Remaining global splits (test is being run by the orphaned python we're waiting on)
for sp in ood_compositional ood_concept ood_frame ood_tailrisk ood_shifted_cal_test; do
  run_one global "$sp"
done

# All cvar_cell splits (including test — which was never started)
for sp in test ood_compositional ood_concept ood_frame ood_tailrisk ood_shifted_cal_test; do
  run_one cvar_cell "$sp"
done

echo "[GPU $GPU seed $SEED] ALL B6 RUNS COMPLETE at $(date)" | tee -a /tmp/b6_screen_gpu${GPU}.log
