"""
BiasMap: per-cell diagnostics from cached model outputs.
Ranks cells by composite failure score.
"""
import numpy as np
from typing import Dict, List, Tuple
from .mondrian_partition import MondrianPartition


class BiasMapDiagnosis:

    def __init__(self, partition: MondrianPartition, alpha: float = 0.1):
        self.partition = partition
        self.alpha = alpha

    # ------------------------------------------------------------------
    def compute_cell_diagnostics(
        self,
        nc_scores: Dict[str, float],
        correctness: Dict[str, bool],
        cal_threshold: float,
    ) -> Dict[str, dict]:
        """Per-cell: coverage, set_size, cvar, nc_variance, etc."""
        cell_diags: Dict[str, dict] = {}

        for cell_id, cell in self.partition.cells.items():
            ids_present = [sid for sid in cell.sample_ids if sid in nc_scores]
            if not ids_present:
                continue

            nc = np.array([nc_scores[sid] for sid in ids_present])
            corr = np.array([float(correctness.get(sid, False))
                             for sid in ids_present])
            loss = 1.0 - corr

            # Global-threshold set membership (binary: set size 1 or 2)
            in_set_global = nc <= cal_threshold
            set_sizes_global = np.where(in_set_global, 1.0, 2.0)

            # Cell-wise (Mondrian) threshold
            if len(nc) >= 2:
                n = len(nc)
                q = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
                cell_thr = float(np.quantile(nc, q))
            else:
                cell_thr = cal_threshold
            set_sizes_mondrian = np.where(nc <= cell_thr, 1.0, 2.0)

            # CVaR of loss (worst-alpha tail)
            sorted_loss = np.sort(loss)[::-1]
            k = max(1, int(np.ceil(len(sorted_loss) * self.alpha)))
            cvar = float(sorted_loss[:k].mean())

            cell_diags[cell_id] = {
                "cell_id": cell_id,
                "support": len(nc),
                "accuracy": float(corr.mean()),
                "mean_loss": float(loss.mean()),
                "mean_set_size_global": float(set_sizes_global.mean()),
                "mean_set_size_mondrian": float(set_sizes_mondrian.mean()),
                "coverage_global": float(in_set_global.mean()),
                "coverage_mondrian": float((nc <= cell_thr).mean()),
                "coverage_gap": float(max(0, (1 - self.alpha) - in_set_global.mean())),
                "cvar_loss": cvar,
                "nc_mean": float(nc.mean()),
                "nc_variance": float(nc.var()),
                "cell_threshold": cell_thr,
                "is_backed_off": cell.is_backed_off,
            }

        # James-Stein shrinkage for small cells
        global_mean = float(np.mean([d["mean_loss"] for d in cell_diags.values()]))
        for d in cell_diags.values():
            n = d["support"]
            if n < self.partition.shrinkage_threshold:
                s = max(0.0, 1.0 - (self.partition.shrinkage_threshold - n)
                        / self.partition.shrinkage_threshold)
                d["mean_loss_shrunk"] = s * d["mean_loss"] + (1 - s) * global_mean
                d["cvar_loss_shrunk"] = s * d["cvar_loss"] + (1 - s) * global_mean
            else:
                d["mean_loss_shrunk"] = d["mean_loss"]
                d["cvar_loss_shrunk"] = d["cvar_loss"]

        return cell_diags

    # ------------------------------------------------------------------
    def rank_cells(
        self,
        diagnostics: Dict[str, dict],
        method: str = "cp_composite",
    ) -> List[Tuple[str, float]]:
        """Rank cells worst-first by *method*.

        Methods: cp_composite, cp_set_size, loss, error_rate, entropy, random
        """
        scores: Dict[str, float] = {}
        for cid, d in diagnostics.items():
            scores[cid] = _score(d, method, self.alpha)
        return sorted(scores.items(), key=lambda x: -x[1])


# helpers ---------------------------------------------------------------

def _score(d: dict, method: str, alpha: float) -> float:
    if method == "cp_composite":
        return (0.4 * d["mean_set_size_global"] / 2.0
                + 0.3 * d["coverage_gap"] / max(1 - alpha, 1e-9)
                + 0.3 * d["cvar_loss"])
    if method == "cp_set_size":
        return d["mean_set_size_global"]
    if method == "loss":
        return d["mean_loss_shrunk"]
    if method == "error_rate":
        return 1.0 - d["accuracy"]
    if method == "entropy":
        return d["nc_variance"]
    if method == "random":
        return float(np.random.random())
    raise ValueError(f"Unknown method: {method}")
