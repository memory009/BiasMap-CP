"""Bootstrap stability analysis for Mondrian partition + cell ranking."""
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List
from .mondrian_partition import MondrianPartition
from .bias_map import BiasMapDiagnosis


def bootstrap_stability(
    samples: List[dict],
    nc_scores: Dict[str, float],
    correctness: Dict[str, bool],
    cal_threshold: float,
    n_bootstrap: int = 3,
    min_support: int = 30,
    shrinkage_threshold: int = 50,
    seed: int = 42,
) -> dict:
    """
    Resample diag_cal with replacement, rebuild partition + ranking,
    and measure stability across resamples.
    """
    rng = np.random.RandomState(seed)
    all_rankings: List[List[str]] = []
    all_cell_counts: List[int] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(samples), size=len(samples), replace=True)
        boot = [samples[i] for i in idx]
        boot_ids = {s["id"] for s in boot}

        part = MondrianPartition(min_support, shrinkage_threshold)
        part.build(boot)

        diag = BiasMapDiagnosis(part, alpha=0.1)
        b_nc = {k: v for k, v in nc_scores.items() if k in boot_ids}
        b_corr = {k: v for k, v in correctness.items() if k in boot_ids}
        cell_diags = diag.compute_cell_diagnostics(b_nc, b_corr, cal_threshold)

        ranking = diag.rank_cells(cell_diags, method="cp_composite")
        all_rankings.append([c for c, _ in ranking])
        all_cell_counts.append(len(part.cells))

    counts = np.array(all_cell_counts)
    cv = float(counts.std() / counts.mean()) if counts.mean() > 0 else 0.0

    # Top-10 Jaccard across bootstrap pairs
    top_k = 10
    top_sets = [set(r[:top_k]) for r in all_rankings]
    jaccards = []
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            inter = len(top_sets[i] & top_sets[j])
            union = len(top_sets[i] | top_sets[j])
            jaccards.append(inter / union if union else 0.0)
    mean_jaccard = float(np.mean(jaccards)) if jaccards else 0.0

    # Spearman over common cells
    common = set.intersection(*(set(r) for r in all_rankings)) if all_rankings else set()
    rank_corrs = []
    if len(common) >= 5:
        for i in range(len(all_rankings)):
            for j in range(i + 1, len(all_rankings)):
                r1 = {c: k for k, c in enumerate(all_rankings[i]) if c in common}
                r2 = {c: k for k, c in enumerate(all_rankings[j]) if c in common}
                ordered = sorted(common)
                corr, _ = spearmanr([r1[c] for c in ordered],
                                    [r2[c] for c in ordered])
                rank_corrs.append(float(corr))
    mean_corr = float(np.mean(rank_corrs)) if rank_corrs else 0.0

    passed = cv <= 0.20 and mean_jaccard >= 0.50

    return {
        "n_bootstrap": n_bootstrap,
        "cell_counts": counts.tolist(),
        "cell_count_cv": cv,
        "top10_jaccard_mean": mean_jaccard,
        "top10_jaccard_scores": [float(j) for j in jaccards],
        "rank_correlation_mean": mean_corr,
        "rank_correlations": rank_corrs,
        "common_cells_count": len(common),
        "passed": passed,
        "threshold": "cell_count_cv<=0.20 AND top10_jaccard>=0.50",
    }
