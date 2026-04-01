#!/usr/bin/env python3
"""
B1: Diagnosis Lock  (CPU-only, < 2 hours)

1. Build Mondrian partition on diag_cal
2. Compute BiasMap diagnostics per model
3. Bootstrap stability analysis
4. A7 ranking comparison: CP vs loss/error/entropy (oracle = repair_val)

Usage:
    python scripts/run_b1_diagnosis.py
"""
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diagnosis.mondrian_partition import MondrianPartition
from src.diagnosis.bias_map import BiasMapDiagnosis
from src.diagnosis.stability import bootstrap_stability
from src.diagnosis.ranking_comparison import compare_rankings

# ── paths ──────────────────────────────────────────────────────────────
SPLITS = Path("data/splits")
S1     = Path("results/sprint1")
OUT    = Path("results/sprint2/b1_diagnosis")

MODELS   = ["Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct",
            "llava-1.5-7b-hf", "blip2-opt-2.7b"]
DATASETS = ["vsr", "gqa", "whatsup"]
ALPHA    = 0.1
MIN_SUPPORT = 30
SHRINKAGE   = 50


# ── helpers ────────────────────────────────────────────────────────────
def load_split(name: str):
    samples = []
    with open(SPLITS / f"{name}.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def load_outputs(model: str, split: str):
    """Load cached model outputs across all datasets for a given split."""
    nc: dict[str, float] = {}
    corr: dict[str, bool] = {}
    probs: dict[str, dict] = {}
    for ds in DATASETS:
        p = S1 / model / ds / f"{split}.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                rec = json.loads(line)
                sid = rec["sample_id"]
                nc[sid] = rec["nonconformity_score"]
                corr[sid] = rec["correct"]
                probs[sid] = rec.get("probabilities", {})
    return nc, corr, probs


# ── main ───────────────────────────────────────────────────────────────
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # 1. Load splits
    diag_cal   = load_split("diag_cal")
    repair_val = load_split("repair_val")
    print(f"diag_cal: {len(diag_cal)},  repair_val: {len(repair_val)}")

    # 2. Build Mondrian partition on diag_cal
    partition = MondrianPartition(MIN_SUPPORT, SHRINKAGE)
    partition.build(diag_cal)
    partition.save(OUT / "partition.json")
    print(f"Partition saved ({len(partition.cells)} cells)")

    # 3. Per-model analysis
    summary_rows = []
    for model in MODELS:
        print(f"\n{'='*60}\n  {model}\n{'='*60}")
        mdir = OUT / model
        mdir.mkdir(exist_ok=True)

        # -- load diag_cal outputs --
        nc, corr, probs = load_outputs(model, "diag_cal")
        if not nc:
            print("  ⚠ no diag_cal outputs, skipping")
            continue

        # -- global CP threshold --
        all_nc = np.array(list(nc.values()))
        n = len(all_nc)
        q = min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0)
        cal_threshold = float(np.quantile(all_nc, q))
        print(f"  global threshold: {cal_threshold:.4f}  (n={n})")

        # -- cell diagnostics --
        diag = BiasMapDiagnosis(partition, ALPHA)
        cell_diags = diag.compute_cell_diagnostics(nc, corr, cal_threshold)
        with open(mdir / "diagnostics.json", "w") as f:
            json.dump(cell_diags, f, indent=2)
        print(f"  {len(cell_diags)} cells diagnosed")

        # -- rankings (all methods) --
        print("  Rankings (top-5 worst cells):")
        for method in ["cp_composite", "cp_set_size", "loss", "error_rate", "entropy"]:
            ranked = diag.rank_cells(cell_diags, method)
            top5 = [(c, f"{s:.3f}") for c, s in ranked[:5]]
            print(f"    {method:15s}: {top5}")

        # -- bootstrap stability --
        print("  Running bootstrap stability …")
        stab = bootstrap_stability(
            diag_cal, nc, corr, cal_threshold,
            n_bootstrap=3,
            min_support=MIN_SUPPORT,
            shrinkage_threshold=SHRINKAGE,
        )
        with open(mdir / "stability.json", "w") as f:
            json.dump(stab, f, indent=2)
        print(f"    CV={stab['cell_count_cv']:.3f}  "
              f"Jaccard={stab['top10_jaccard_mean']:.3f}  "
              f"RankCorr={stab['rank_correlation_mean']:.3f}  "
              f"PASSED={stab['passed']}")

        # -- A7: ranking comparison (oracle = repair_val) --
        rv_nc, rv_corr, _ = load_outputs(model, "repair_val")
        if rv_nc:
            # Build a lightweight partition on repair_val for oracle
            rv_partition = MondrianPartition(min_support=10, shrinkage_threshold=20)
            rv_partition.build(repair_val)

            oracle_loss: dict[str, float] = {}
            for cid, cell in rv_partition.cells.items():
                cell_c = [float(rv_corr.get(sid, True))
                          for sid in cell.sample_ids if sid in rv_corr]
                if cell_c:
                    oracle_loss[cid] = 1.0 - np.mean(cell_c)

            # We need both partitions to share cell IDs for comparison.
            # Because diag_cal and repair_val use different sample sets,
            # the cell IDs from the partitions are structurally identical
            # (same feature keys).  For cells only in one partition we skip.
            rank_result = compare_rankings(cell_diags, oracle_loss, ALPHA)
            with open(mdir / "ranking_comparison.json", "w") as f:
                json.dump(rank_result, f, indent=2)
            print(f"  A7: CP_ρ={rank_result['cp_spearman']:.3f}  "
                  f"best_simple_ρ={rank_result['best_simple_spearman']:.3f}  "
                  f"CP_wins={rank_result['cp_wins']}")
        else:
            print("  ⚠ no repair_val outputs, skipping A7")
            rank_result = None

        summary_rows.append({
            "model": model,
            "n_cells": len(cell_diags),
            "stability_cv": stab["cell_count_cv"],
            "stability_jaccard": stab["top10_jaccard_mean"],
            "stability_passed": stab["passed"],
            "a7_cp_spearman": rank_result["cp_spearman"] if rank_result else None,
            "a7_best_simple": rank_result["best_simple_spearman"] if rank_result else None,
            "a7_cp_wins": rank_result["cp_wins"] if rank_result else None,
        })

    # 4. Summary
    print(f"\n{'='*60}")
    print("B1 SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':30s} {'Cells':>5} {'CV':>6} {'Jacc':>6} {'Stab':>5} "
          f"{'CP_ρ':>6} {'Sim_ρ':>6} {'CP>?':>5}")
    for r in summary_rows:
        print(f"{r['model']:30s} {r['n_cells']:5d} "
              f"{r['stability_cv']:6.3f} {r['stability_jaccard']:6.3f} "
              f"{'✅' if r['stability_passed'] else '❌':>5s} "
              f"{r['a7_cp_spearman'] or 0:6.3f} "
              f"{r['a7_best_simple'] or 0:6.3f} "
              f"{'✅' if r.get('a7_cp_wins') else '❌':>5s}")

    with open(OUT / "b1_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    # 5. Gate decisions
    print("\n--- DECISION GATES ---")
    any_stable = any(r["stability_passed"] for r in summary_rows)
    any_cp_wins = any(r.get("a7_cp_wins") for r in summary_rows)
    print(f"  K2 (partition stable for ≥1 model): {'✅ PASS' if any_stable else '❌ FAIL — STOP'}")
    print(f"  K3 (CP ranking ≥1 model wins):      {'✅ PASS' if any_cp_wins else '⚠️ DOWNGRADE'}")
    if any_stable:
        print("\n→ Proceed to B2 (targeted vs global repair)")
    else:
        print("\n⛔ Partition unstable. Review min_support / features before continuing.")


if __name__ == "__main__":
    main()
