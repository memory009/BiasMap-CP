#!/usr/bin/env python3
"""Aggregate B6 results across 3 seeds x 2 methods x 6 splits."""
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = Path("results/sprint3/b6_test_ood")
METHODS = ["global", "cvar_cell"]
SEEDS = [1, 2, 3]
SPLITS = ["test", "ood_compositional", "ood_concept",
          "ood_frame", "ood_tailrisk", "ood_shifted_cal_test"]

METRICS = [
    "cvar_coverage_gap",
    "mondrian_marginal_coverage",
    "worst_cell_coverage",
    "repair_val_accuracy",   # = accuracy on eval_split (field name preserved)
    "w10_cvar_err",
    "global_mean_set_size",
    "n_cells_evaluated",
]


def load_one(method, seed, split):
    path = ROOT / f"{method}_seed{seed}_{split}" / "b6_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    # agg[(method, split)][metric] = [v_seed1, v_seed2, v_seed3]
    agg = defaultdict(lambda: defaultdict(list))
    missing = []
    for m in METHODS:
        for sp in SPLITS:
            for s in SEEDS:
                d = load_one(m, s, sp)
                if d is None:
                    missing.append(f"{m}_seed{s}_{sp}")
                    continue
                for k in METRICS:
                    if k in d:
                        agg[(m, sp)][k].append(d[k])

    if missing:
        print(f"WARNING: {len(missing)} missing runs:")
        for x in missing:
            print(f"  {x}")

    # Per-(method, split) mean ± std
    summary = {}
    for (m, sp), metrics in agg.items():
        summary[f"{m}|{sp}"] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
            for k, v in metrics.items()
        }

    # Headline comparison: cvar_cell vs global on cvar_coverage_gap per split
    print(f"\n{'='*100}")
    print("B6 RESULTS — CP Coverage Gap (cvar_cell vs global, 3 seeds)")
    print(f"{'='*100}")
    print(f"{'Split':<25s} {'global gap (μ±σ)':<22s} {'cvar_cell gap (μ±σ)':<24s} {'Δ':<10s} {'paired-t p':<12s} {'winner':<10s}")
    print("-" * 100)

    win_count = 0
    gap_rows = []
    for sp in SPLITS:
        g = agg[("global", sp)].get("cvar_coverage_gap", [])
        c = agg[("cvar_cell", sp)].get("cvar_coverage_gap", [])
        if len(g) < 2 or len(c) < 2:
            print(f"{sp:<25s} (insufficient data: global={len(g)}, cvar_cell={len(c)})")
            continue
        gm, gs = np.mean(g), np.std(g)
        cm, cs = np.mean(c), np.std(c)
        delta = cm - gm
        try:
            t, p = stats.ttest_rel(c, g)
        except Exception:
            p = float("nan")
        winner = "cvar_cell" if cm < gm else "global"
        if cm < gm:
            win_count += 1
        print(f"{sp:<25s} {gm:.4f}±{gs:.4f}    {cm:.4f}±{cs:.4f}     {delta:+.4f}  {p:<12.4f} {winner:<10s}")
        gap_rows.append({"split": sp, "global_mean": gm, "global_std": gs,
                         "cvar_cell_mean": cm, "cvar_cell_std": cs,
                         "delta": delta, "p_value": float(p), "winner": winner})

    print("-" * 100)
    print(f"CVaR-cell wins (smaller gap) on {win_count}/{len(SPLITS)} splits")

    # Also print accuracy + marginal cov + w10 cvar err
    print(f"\n{'='*100}")
    print("Auxiliary Metrics (accuracy, marginal_cov, w10_cvar_err, worst_cell_cov, n_cells)")
    print(f"{'='*100}")
    hdr = f"{'Method':<12s} {'Split':<25s} {'acc':<10s} {'mondrian_cov':<15s} {'worst_cell':<13s} {'w10_err':<10s} {'n_cells':<8s}"
    print(hdr)
    print("-" * 100)
    for sp in SPLITS:
        for m in METHODS:
            s = summary.get(f"{m}|{sp}", {})
            acc = s.get("repair_val_accuracy", {}).get("mean", float("nan"))
            mc = s.get("mondrian_marginal_coverage", {}).get("mean", float("nan"))
            wc = s.get("worst_cell_coverage", {}).get("mean", float("nan"))
            w10 = s.get("w10_cvar_err", {}).get("mean", float("nan"))
            nc = s.get("n_cells_evaluated", {}).get("mean", float("nan"))
            print(f"{m:<12s} {sp:<25s} {acc:<10.4f} {mc:<15.4f} {wc:<13.4f} {w10:<10.4f} {nc:<8.1f}")
        print()

    # Gate check (relaxed on OOD)
    print(f"\n{'='*100}")
    print("B6 Gate Check")
    print(f"{'='*100}")
    iid_row = next((r for r in gap_rows if r["split"] == "test"), None)
    iid_pass = iid_row and iid_row["winner"] == "cvar_cell" and iid_row["p_value"] < 0.10
    ood_wins = sum(1 for r in gap_rows if r["split"] != "test" and r["winner"] == "cvar_cell")
    ood_pass = ood_wins >= 3

    # Coverage floor
    floor_violations = []
    for sp in SPLITS:
        for m in METHODS:
            s = summary.get(f"{m}|{sp}", {})
            mc = s.get("mondrian_marginal_coverage", {}).get("mean", 1.0)
            floor = 0.90 if sp == "test" else 0.85
            if mc < floor:
                floor_violations.append(f"{m}/{sp}: mondrian_cov={mc:.4f} < {floor}")

    print(f"  IID test gate:       {'PASS' if iid_pass else 'FAIL'}  ", end="")
    if iid_row:
        print(f"(delta={iid_row['delta']:+.4f}, p={iid_row['p_value']:.4f})")
    else:
        print("(no test data)")
    print(f"  OOD >=3/5 gate:      {'PASS' if ood_pass else 'FAIL'}  ({ood_wins}/5 OOD splits)")
    print(f"  Coverage floor:      {'PASS' if not floor_violations else 'FAIL'}")
    for v in floor_violations:
        print(f"    - {v}")

    overall = iid_pass and ood_pass and not floor_violations
    print(f"\n  B6 OVERALL: {'PASS' if overall else 'FAIL'}")

    # Save summary
    out = {
        "summary": summary,
        "gap_comparison": gap_rows,
        "ood_win_count": ood_wins,
        "iid_pass": bool(iid_pass),
        "ood_pass": bool(ood_pass),
        "coverage_floor_violations": floor_violations,
        "b6_overall_pass": bool(overall),
        "missing_runs": missing,
    }
    out_path = ROOT / "b6_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
