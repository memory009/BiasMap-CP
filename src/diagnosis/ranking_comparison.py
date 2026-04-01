"""
A7: Compare CP-based cell ranking against simpler heuristics.
Oracle = per-cell mean loss on repair_val (never used for diagnosis).
"""
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List, Optional
from .bias_map import _score


_METHODS = ["cp_composite", "cp_set_size", "loss", "error_rate", "entropy", "random"]


def compare_rankings(
    diag_diagnostics: Dict[str, dict],
    oracle_loss: Dict[str, float],
    alpha: float = 0.1,
    methods: Optional[List[str]] = None,
) -> dict:
    """
    Rank cells by each method (from diag_cal diagnostics) and compare
    against oracle ranking (repair_val loss, worst-first).
    """
    methods = methods or _METHODS

    # Oracle ranking (worst first)
    oracle_ranked = sorted(oracle_loss.items(), key=lambda x: -x[1])
    oracle_order = {cid: i for i, (cid, _) in enumerate(oracle_ranked)}

    results: Dict[str, dict] = {}

    for method in methods:
        # Method ranking
        method_ranked = sorted(
            diag_diagnostics.items(),
            key=lambda kv: -_score(kv[1], method, alpha),
        )
        method_order = {cid: i for i, (cid, _) in enumerate(method_ranked)}

        common = set(oracle_order) & set(method_order)
        if len(common) < 5:
            results[method] = {"spearman": 0.0, "top_k_recall": {}, "n_common": len(common)}
            continue

        ordered = sorted(common)
        corr, pval = spearmanr(
            [oracle_order[c] for c in ordered],
            [method_order[c] for c in ordered],
        )

        top_k_recall = {}
        for k in (5, 10, 20):
            if k > len(common):
                continue
            o_top = {c for c, _ in oracle_ranked[:k] if c in common}
            m_top = {c for c, _ in method_ranked[:k] if c in common}
            top_k_recall[f"top_{k}"] = len(o_top & m_top) / max(len(o_top), 1)

        results[method] = {
            "spearman": float(corr),
            "spearman_pval": float(pval),
            "top_k_recall": top_k_recall,
            "n_common": len(common),
        }

    cp_rho = results.get("cp_composite", {}).get("spearman", 0.0)
    best_simple = max(
        results.get(m, {}).get("spearman", 0.0) for m in ("loss", "error_rate", "entropy")
    )

    return {
        "rankings": results,
        "cp_wins": cp_rho > best_simple,
        "cp_spearman": cp_rho,
        "best_simple_spearman": best_simple,
        "margin": cp_rho - best_simple,
    }
