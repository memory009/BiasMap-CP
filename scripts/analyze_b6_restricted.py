#!/usr/bin/env python3
"""Post-hoc B6 analysis on the 34 repair_val-support cells used in B2/B4.

This script reconstructs the exact training-side eligible cell set by:
1. Loading the saved Mondrian partition from B1
2. Re-counting support on `repair_val`
3. Keeping cells with support >= MIN_CELL_SUPPORT

It then re-evaluates existing B6 outputs on a chosen eval split after
filtering samples to that eligible-cell set.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diagnosis.mondrian_partition import MondrianPartition

ROOT = Path("results/sprint3/b6_test_ood")
PARTITION_PATH = Path("results/sprint2/b1_diagnosis/partition.json")
REPAIR_VAL_PATH = Path("data/splits/repair_val.jsonl")

METHODS = ["global", "cvar_cell"]
SEEDS = [1, 2, 3]
MIN_CELL_SUPPORT = 20
MIN_CELL_SIZE = 30


def compute_cvar(values, alpha=0.1):
    if not values:
        return 0.0
    sorted_desc = sorted(values, reverse=True)
    k = max(1, int(np.ceil(alpha * len(sorted_desc))))
    return float(np.mean(sorted_desc[:k]))


class SplitCP:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.threshold = None

    def coverage(self, nc_scores):
        if self.threshold is None:
            return 0.0
        return float(np.mean([score <= self.threshold for score in nc_scores]))

    def predict_set(self, probs):
        if self.threshold is None:
            raise ValueError("Must set threshold before predicting")
        return [answer for answer, prob in probs.items() if (1.0 - prob) <= self.threshold]

    def mean_set_size(self, prob_dicts):
        if not prob_dicts:
            return 0.0
        return float(np.mean([len(self.predict_set(probs)) for probs in prob_dicts]))


class MondrianCP:
    def __init__(self, alpha=0.1, min_cell_size=30):
        self.alpha = alpha
        self.min_cell_size = min_cell_size
        self.cell_thresholds = {}
        self.global_threshold = None

    def predict_set(self, probs, cell_label):
        threshold = self.cell_thresholds.get(cell_label, self.global_threshold)
        if threshold is None:
            threshold = 1.0
        return [answer for answer, prob in probs.items() if (1.0 - prob) <= threshold]

    def per_cell_coverage(self, nc_scores, cell_labels):
        by_cell = defaultdict(list)
        for score, label in zip(nc_scores, cell_labels):
            by_cell[label].append(score)
        result = {}
        for cell, scores in by_cell.items():
            threshold = self.cell_thresholds.get(cell, self.global_threshold)
            if threshold is None:
                continue
            result[cell] = float(np.mean([score <= threshold for score in scores]))
        return result

    def per_cell_set_size(self, prob_dicts, cell_labels):
        by_cell = defaultdict(list)
        for probs, label in zip(prob_dicts, cell_labels):
            by_cell[label].append(len(self.predict_set(probs, label)))
        return {cell: float(np.mean(sizes)) for cell, sizes in by_cell.items()}


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def reconstruct_eligible_cells():
    partition = MondrianPartition.load(PARTITION_PATH)
    counts = Counter()
    with open(REPAIR_VAL_PATH) as f:
        for line in f:
            sample = json.loads(line)
            cell_id = partition.get_cell_by_features(sample)
            if cell_id is not None:
                counts[cell_id] += 1

    eligible = sorted([cid for cid, n in counts.items() if n >= MIN_CELL_SUPPORT])
    return {
        "partition_min_support": partition.min_support,
        "reporting_min_support": MIN_CELL_SUPPORT,
        "n_cells_total": len(counts),
        "n_cells_eligible": len(eligible),
        "eligible_cells": eligible,
        "repair_val_cell_support": dict(sorted(counts.items())),
    }


def evaluate_one(method: str, seed: int, split: str, eligible_cells):
    run_dir = ROOT / f"{method}_seed{seed}_{split}"
    outputs_path = run_dir / f"{split}_outputs.jsonl"
    thresholds_path = run_dir / "mondrian_thresholds.json"
    if not outputs_path.exists():
        raise FileNotFoundError(f"Missing outputs: {outputs_path}")
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Missing thresholds: {thresholds_path}")

    outputs = load_jsonl(outputs_path)
    thresholds = load_json(thresholds_path)

    restricted = [row for row in outputs if row.get("cell_id") in eligible_cells]
    present_cells = sorted({row["cell_id"] for row in restricted if row.get("cell_id")})
    missing_cells = sorted(set(eligible_cells) - set(present_cells))

    global_cp = SplitCP(alpha=thresholds["alpha"])
    global_cp.threshold = thresholds["global_threshold"]

    mondrian_cp = MondrianCP(alpha=thresholds["alpha"], min_cell_size=MIN_CELL_SIZE)
    mondrian_cp.global_threshold = thresholds["global_threshold"]
    mondrian_cp.cell_thresholds = thresholds["cell_thresholds"]

    nc_scores = [row["nc_score"] for row in restricted]
    cell_labels = [row["cell_id"] or "unknown" for row in restricted]
    prob_dicts = [row["probabilities"] for row in restricted]

    marginal_coverage = global_cp.coverage(nc_scores) if restricted else 0.0
    global_mean_set_size = global_cp.mean_set_size(prob_dicts) if restricted else 0.0
    per_cell_coverage = mondrian_cp.per_cell_coverage(nc_scores, cell_labels) if restricted else {}
    per_cell_set_size = mondrian_cp.per_cell_set_size(prob_dicts, cell_labels) if restricted else {}

    cell_errors_all = defaultdict(list)
    for row in restricted:
        if row.get("cell_id"):
            cell_errors_all[row["cell_id"]].append(1.0 - float(row["correct"]))
    cell_mean_error = {
        cid: float(np.mean(errs))
        for cid, errs in cell_errors_all.items()
        if len(errs) >= MIN_CELL_SUPPORT
    }
    cell_mean_error_all = {
        cid: float(np.mean(errs))
        for cid, errs in cell_errors_all.items()
    }

    target_coverage = 1.0 - thresholds["alpha"]
    coverage_shortfalls = {}
    for cell_id, cov in per_cell_coverage.items():
        shortfall = max(0.0, target_coverage - cov)
        coverage_shortfalls[cell_id] = {
            "coverage": float(cov),
            "shortfall": float(shortfall),
            "set_size": float(per_cell_set_size.get(cell_id, 0.0)),
            "error_rate": float(cell_mean_error.get(cell_id, -1.0)),
            "error_rate_all": float(cell_mean_error_all.get(cell_id, -1.0)),
            "n_eval_samples": len(cell_errors_all.get(cell_id, [])),
        }

    all_coverages = list(per_cell_coverage.values())
    worst_cell_coverage = min(all_coverages) if all_coverages else 0.0
    worst_cell_id = min(per_cell_coverage, key=per_cell_coverage.get) if per_cell_coverage else "N/A"
    mean_cell_coverage = float(np.mean(all_coverages)) if all_coverages else 0.0
    cvar_coverage_gap = (
        compute_cvar([max(0.0, target_coverage - c) for c in all_coverages], alpha=0.1)
        if all_coverages else 1.0
    )

    mondrian_marginal = float(np.mean([
        row["nc_score"] <= mondrian_cp.cell_thresholds.get(
            row["cell_id"] or "unknown", mondrian_cp.global_threshold or 1.0
        )
        for row in restricted
    ])) if restricted else 0.0

    accuracy = float(np.mean([row["correct"] for row in restricted])) if restricted else 0.0

    w10_cvar_err = compute_cvar(list(cell_mean_error.values()), alpha=0.1) if cell_mean_error else 1.0
    w10_cvar_err_all = compute_cvar(list(cell_mean_error_all.values()), alpha=0.1) if cell_mean_error_all else 1.0

    return {
        "method": method,
        "seed": seed,
        "split": split,
        "alpha": thresholds["alpha"],
        "eligible_cell_count_from_repair_val": len(eligible_cells),
        "eligible_cells_present_in_eval": len(present_cells),
        "eligible_cells_missing_from_eval": missing_cells,
        "n_eval_samples_restricted": len(restricted),
        "repair_val_accuracy": accuracy,
        "w10_cvar_err": w10_cvar_err,
        "w10_cvar_err_all_restricted_cells": w10_cvar_err_all,
        "cell_mean_errors": cell_mean_error,
        "cell_mean_errors_all_restricted": cell_mean_error_all,
        "marginal_coverage": float(marginal_coverage),
        "mondrian_marginal_coverage": float(mondrian_marginal),
        "global_mean_set_size": float(global_mean_set_size),
        "mean_cell_coverage": mean_cell_coverage,
        "worst_cell_coverage": float(worst_cell_coverage),
        "worst_cell_id": worst_cell_id,
        "cvar_coverage_gap": float(cvar_coverage_gap),
        "n_cells_evaluated": len(per_cell_coverage),
        "n_cells_w10_err_support20": len(cell_mean_error),
        "per_cell_detail": coverage_shortfalls,
    }


def aggregate(results):
    summary = defaultdict(lambda: defaultdict(list))
    for row in results:
        key = f"{row['method']}|{row['split']}"
        for metric in [
            "cvar_coverage_gap",
            "mondrian_marginal_coverage",
            "worst_cell_coverage",
            "repair_val_accuracy",
            "w10_cvar_err",
            "w10_cvar_err_all_restricted_cells",
            "global_mean_set_size",
            "n_cells_evaluated",
            "n_cells_w10_err_support20",
            "n_eval_samples_restricted",
        ]:
            summary[key][metric].append(row[metric])

    out = {}
    for key, metric_map in summary.items():
        out[key] = {
            metric: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": len(values),
            }
            for metric, values in metric_map.items()
        }
    return out


def paired_compare(results, split):
    by_method = {method: [] for method in METHODS}
    for method in METHODS:
        rows = sorted(
            [row for row in results if row["method"] == method and row["split"] == split],
            key=lambda row: row["seed"],
        )
        by_method[method] = rows

    global_gap = [row["cvar_coverage_gap"] for row in by_method["global"]]
    cvar_gap = [row["cvar_coverage_gap"] for row in by_method["cvar_cell"]]
    global_err = [row["w10_cvar_err"] for row in by_method["global"]]
    cvar_err = [row["w10_cvar_err"] for row in by_method["cvar_cell"]]

    gap_test = stats.ttest_rel(cvar_gap, global_gap) if len(global_gap) >= 2 else None
    err_test = stats.ttest_rel(cvar_err, global_err) if len(global_err) >= 2 else None

    return {
        "split": split,
        "gap_delta_mean": float(np.mean(cvar_gap) - np.mean(global_gap)),
        "gap_p_value": float(gap_test.pvalue) if gap_test is not None else None,
        "gap_seed_wins": int(sum(c < g for c, g in zip(cvar_gap, global_gap))),
        "w10_err_delta_mean": float(np.mean(cvar_err) - np.mean(global_err)),
        "w10_err_p_value": float(err_test.pvalue) if err_test is not None else None,
        "w10_err_seed_wins": int(sum(c < g for c, g in zip(cvar_err, global_err))),
        "global_gap_by_seed": global_gap,
        "cvar_gap_by_seed": cvar_gap,
        "global_w10_err_by_seed": global_err,
        "cvar_w10_err_by_seed": cvar_err,
    }


def main():
    parser = argparse.ArgumentParser(description="Restricted B6 analysis on the 34 repair_val-support cells.")
    parser.add_argument("--split", default="test", help="Eval split name inside results/sprint3/b6_test_ood")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path (default: results/sprint3/b6_test_ood/b6_restricted_<split>_summary.json)",
    )
    args = parser.parse_args()

    eligible_info = reconstruct_eligible_cells()
    eligible_cells = eligible_info["eligible_cells"]
    if eligible_info["n_cells_eligible"] != 34:
        print(
            f"WARNING: reconstructed eligible cell count is {eligible_info['n_cells_eligible']} (expected 34)"
        )

    per_run = []
    for method in METHODS:
        for seed in SEEDS:
            per_run.append(evaluate_one(method, seed, args.split, eligible_cells))

    summary = aggregate(per_run)
    comparison = paired_compare(per_run, args.split)

    out = {
        "eligible_cell_source": eligible_info,
        "per_run": per_run,
        "summary": summary,
        "comparison": comparison,
    }

    output_path = Path(args.output) if args.output else ROOT / f"b6_restricted_{args.split}_summary.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 100)
    print(f"B6 RESTRICTED ANALYSIS ({args.split})")
    print("=" * 100)
    print(f"Eligible cells reconstructed from repair_val: {eligible_info['n_cells_eligible']}")
    print()
    print(
        f"{'Method':<12s} {'gap (μ±σ)':<20s} {'w10_err (μ±σ)':<20s} "
        f"{'acc (μ±σ)':<20s} {'cells':<10s}"
    )
    print("-" * 100)
    for method in METHODS:
        key = f"{method}|{args.split}"
        row = summary[key]
        print(
            f"{method:<12s} "
            f"{row['cvar_coverage_gap']['mean']:.4f}±{row['cvar_coverage_gap']['std']:.4f}   "
            f"{row['w10_cvar_err']['mean']:.4f}±{row['w10_cvar_err']['std']:.4f}   "
            f"{row['repair_val_accuracy']['mean']:.4f}±{row['repair_val_accuracy']['std']:.4f}   "
            f"{row['n_cells_evaluated']['mean']:.1f}"
        )

    print("-" * 100)
    print(
        f"Gap delta (cvar-global): {comparison['gap_delta_mean']:+.4f}, "
        f"p={comparison['gap_p_value']:.4f}, wins={comparison['gap_seed_wins']}/3"
    )
    print(
        f"W10 err delta (cvar-global): {comparison['w10_err_delta_mean']:+.4f}, "
        f"p={comparison['w10_err_p_value']:.4f}, wins={comparison['w10_err_seed_wins']}/3"
    )
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
