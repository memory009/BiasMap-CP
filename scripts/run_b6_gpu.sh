#!/usr/bin/env bash
# Run all 12 B6 evaluations on a single GPU (2 checkpoints x 6 splits).
# Usage: bash scripts/run_b6_gpu.sh <gpu_id> <seed>
set -u

GPU=$1
SEED=$2
PY=/LOCAL2/psqhe8/anaconda3/envs/biasmap/bin/python
OUT=results/sprint3/b6_test_ood

SPLITS=(test ood_compositional ood_concept ood_frame ood_tailrisk ood_shifted_cal_test)

for m in global cvar_cell; do
  for sp in "${SPLITS[@]}"; do
    dst=$OUT/${m}_seed${SEED}_${sp}
    mkdir -p "$dst"
    echo "[GPU $GPU] === $m seed$SEED $sp ==="
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_b4_recalibration.py \
        --method $m --seed $SEED --eval_split $sp --output_dir $OUT \
        >> "$dst/run.log" 2>&1
    if [[ $? -ne 0 ]]; then
      echo "[GPU $GPU] FAILED: $m seed$SEED $sp (see $dst/run.log)"
    else
      echo "[GPU $GPU] DONE: $m seed$SEED $sp"
    fi
  done
done
echo "[GPU $GPU] ALL 12 RUNS COMPLETE"
