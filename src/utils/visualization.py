"""Visualization utilities for BiasMap-CP."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional


def plot_bias_map(bias_map: Dict[str, Dict],
                  output_path: str,
                  title: str = "BiasMap: Per-Relation Conditional Uncertainty",
                  target_coverage: float = 0.9):
    """Plot the BiasMap: per-cell set size and coverage gap."""
    cells = sorted(bias_map.keys())
    if not cells:
        return

    set_sizes = [bias_map[c].get("mean_set_size", 0) for c in cells]
    coverages = [bias_map[c].get("coverage", 0) for c in cells]
    ns = [bias_map[c].get("n", 0) for c in cells]
    cvar_vals = [bias_map[c].get("cvar_10", 0) for c in cells]
    coverage_gaps = [max(0, target_coverage - cov) for cov in coverages]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Plot 1: Mean set size per cell
    ax = axes[0]
    colors = sns.color_palette("RdYlGn_r", len(cells))
    bars = ax.barh(cells, set_sizes, color=colors)
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.7, label="ideal=1")
    ax.set_xlabel("Mean Prediction Set Size")
    ax.set_title("Uncertainty (Set Size) by Relation Type")
    for bar, n in zip(bars, ns):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"n={n}", va="center", fontsize=8)
    ax.legend()

    # Plot 2: Coverage vs target
    ax = axes[1]
    bar_colors = ["red" if gap > 0.05 else "orange" if gap > 0 else "green"
                  for gap in coverage_gaps]
    ax.barh(cells, coverages, color=bar_colors)
    ax.axvline(x=target_coverage, color="blue", linestyle="--",
               label=f"target={target_coverage}")
    ax.set_xlabel("Empirical Coverage")
    ax.set_title("Coverage by Relation Type")
    ax.set_xlim(0, 1.05)
    red_patch = mpatches.Patch(color="red", label="coverage gap > 5%")
    orange_patch = mpatches.Patch(color="orange", label="coverage gap > 0%")
    green_patch = mpatches.Patch(color="green", label="target met")
    ax.legend(handles=[green_patch, orange_patch, red_patch])

    # Plot 3: CVaR
    ax = axes[2]
    ax.barh(cells, cvar_vals,
            color=sns.color_palette("Reds", len(cells)))
    ax.set_xlabel("CVaR₁₀ of Nonconformity Scores")
    ax.set_title("Tail Risk (CVaR) by Relation Type")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"BiasMap saved to {output_path}")


def plot_relation_accuracy(per_relation: Dict[str, Dict],
                            output_path: str,
                            model_name: str = "Model",
                            dataset_name: str = "Dataset"):
    """Bar plot of per-relation accuracy."""
    relations = sorted(per_relation.keys())
    accs = [per_relation[r].get("accuracy", 0) for r in relations]
    ns = [per_relation[r].get("n", 0) for r in relations]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(relations, accs,
                  color=sns.color_palette("RdYlGn", len(relations)))
    ax.axhline(y=np.mean(accs), color="blue", linestyle="--",
               label=f"mean={np.mean(accs):.3f}")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{model_name} on {dataset_name}: Per-Relation Accuracy")
    ax.set_xticklabels(relations, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"n={n}", ha="center", fontsize=8)
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Accuracy plot saved to {output_path}")


def plot_all_models_comparison(summary: Dict,
                                output_dir: str,
                                split_name: str = "test"):
    """Multi-model comparison heatmap."""
    # summary[model][dataset][split] = {accuracy, ece, ...}
    models = sorted(summary.keys())
    datasets = sorted({ds for m in summary.values() for ds in m.keys()})

    for metric in ["accuracy", "ece", "brier"]:
        matrix = []
        for model in models:
            row = []
            for ds in datasets:
                val = summary.get(model, {}).get(ds, {}).get(split_name, {}).get(metric, np.nan)
                row.append(val)
            matrix.append(row)

        matrix = np.array(matrix, dtype=float)
        fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.5), max(4, len(models) * 0.8)))
        sns.heatmap(matrix, annot=True, fmt=".3f", xticklabels=datasets,
                    yticklabels=models, cmap="RdYlGn",
                    vmin=0, vmax=1, ax=ax)
        ax.set_title(f"{metric.upper()} — {split_name} split")
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"heatmap_{metric}_{split_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Heatmap saved to {out_path}")
