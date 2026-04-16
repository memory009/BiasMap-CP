#!/usr/bin/env bash
# Seed B6 run dirs with B4-cached recal_outputs so Phase A is skipped.
set -euo pipefail

B4_ROOT="results/sprint3/b4_recalibration"
B6_ROOT="results/sprint3/b6_test_ood"

METHODS=(global cvar_cell)
SEEDS=(1 2 3)
SPLITS=(test ood_compositional ood_concept ood_frame ood_tailrisk ood_shifted_cal_test)

n=0
for m in "${METHODS[@]}"; do
  for s in "${SEEDS[@]}"; do
    src="$B4_ROOT/${m}_seed${s}/recal_outputs.jsonl"
    if [[ ! -f "$src" ]]; then
      echo "ERROR: missing $src" >&2
      exit 1
    fi
    for sp in "${SPLITS[@]}"; do
      dst_dir="$B6_ROOT/${m}_seed${s}_${sp}"
      mkdir -p "$dst_dir"
      cp "$src" "$dst_dir/recal_outputs.jsonl"
      n=$((n+1))
    done
  done
done
echo "Seeded $n B6 run dirs under $B6_ROOT"
