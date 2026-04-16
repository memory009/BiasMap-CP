#!/usr/bin/env bash
# B6 launcher: 36 evaluations across 3 GPUs.
# Run each block in a SEPARATE terminal on the corresponding GPU.
#
# Assignment:
#   GPU 0: global seed1 + cvar_cell seed1, all 6 splits (12 runs)
#   GPU 1: global seed2 + cvar_cell seed2, all 6 splits (12 runs)
#   GPU 2: global seed3 + cvar_cell seed3, all 6 splits (12 runs)
#
# Wall time estimate: ~9-10h per GPU.
# Recal cache is pre-seeded by prep_b6.sh (Phase A skipped).

PY=/LOCAL2/psqhe8/anaconda3/envs/biasmap/bin/python
OUT=results/sprint3/b6_test_ood

SPLITS=(test ood_compositional ood_concept ood_frame ood_tailrisk ood_shifted_cal_test)

print_block() {
  local gpu=$1
  local seed=$2
  echo "# ========== GPU $gpu  (seed $seed) =========="
  for m in global cvar_cell; do
    for sp in "${SPLITS[@]}"; do
      echo "CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_b4_recalibration.py --method $m --seed $seed --eval_split $sp --output_dir $OUT 2>&1 | tee $OUT/${m}_seed${seed}_${sp}/run.log"
    done
  done
  echo ""
}

print_block 0 1
print_block 1 2
print_block 2 3
