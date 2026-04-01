"""Conformal prediction implementations for BiasMap-CP.

Includes:
- SplitCP: standard split conformal prediction
- MondrianCP: Mondrian (conditional) conformal prediction
- APS: Adaptive Prediction Sets
- RAPS: Regularized APS
"""
import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
from collections import defaultdict


class SplitCP:
    """Standard split conformal prediction.

    Given calibration nonconformity scores, computes a threshold q̂ such that
    P(Y ∈ C(X)) ≥ 1 - alpha on new examples.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.threshold: Optional[float] = None

    def calibrate(self, nc_scores: List[float]) -> float:
        """Compute conformal threshold from calibration nonconformity scores."""
        scores = np.array(nc_scores)
        n = len(scores)
        # Finite-sample correction: (1 + 1/n) * (1 - alpha) quantile
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        quantile_level = min(quantile_level, 1.0)
        self.threshold = float(np.quantile(scores, quantile_level))
        return self.threshold

    def predict_set(self, probs: Dict[str, float]) -> List[str]:
        """Return prediction set: all answers with nc_score ≤ threshold."""
        if self.threshold is None:
            raise ValueError("Must calibrate before predicting")
        return [ans for ans, p in probs.items()
                if (1.0 - p) <= self.threshold]

    def coverage(self, nc_scores: List[float]) -> float:
        """Empirical coverage on a test set."""
        if self.threshold is None:
            return 0.0
        return float(np.mean([s <= self.threshold for s in nc_scores]))

    def mean_set_size(self, prob_dicts: List[Dict[str, float]]) -> float:
        """Average prediction set size."""
        return float(np.mean([len(self.predict_set(p)) for p in prob_dicts]))


class MondrianCP:
    """Mondrian (conditional) conformal prediction.

    Creates cell-wise calibration: one SplitCP per group/cell.
    Groups can be relation_type, scene_type, or any combination.
    """

    def __init__(self, alpha: float = 0.1, min_cell_size: int = 30):
        self.alpha = alpha
        self.min_cell_size = min_cell_size
        self.cell_thresholds: Dict[str, float] = {}
        self.global_threshold: Optional[float] = None

    def calibrate(self, nc_scores: List[float],
                  cell_labels: List[str]) -> Dict[str, float]:
        """Calibrate per-cell thresholds with hierarchical backoff for sparse cells."""
        by_cell = defaultdict(list)
        for score, label in zip(nc_scores, cell_labels):
            by_cell[label].append(score)

        # Global fallback threshold
        global_cp = SplitCP(self.alpha)
        self.global_threshold = global_cp.calibrate(nc_scores)

        for cell, scores in by_cell.items():
            if len(scores) >= self.min_cell_size:
                cp = SplitCP(self.alpha)
                self.cell_thresholds[cell] = cp.calibrate(scores)
            else:
                # Conservative: use global threshold (or parent cell)
                self.cell_thresholds[cell] = self.global_threshold

        return self.cell_thresholds

    def predict_set(self, probs: Dict[str, float],
                    cell_label: str) -> List[str]:
        """Return prediction set using cell-specific threshold."""
        threshold = self.cell_thresholds.get(cell_label, self.global_threshold)
        if threshold is None:
            threshold = 1.0
        return [ans for ans, p in probs.items()
                if (1.0 - p) <= threshold]

    def per_cell_coverage(self, nc_scores: List[float],
                           cell_labels: List[str]) -> Dict[str, float]:
        """Compute per-cell empirical coverage."""
        by_cell = defaultdict(list)
        for score, label in zip(nc_scores, cell_labels):
            by_cell[label].append(score)

        result = {}
        for cell, scores in by_cell.items():
            threshold = self.cell_thresholds.get(cell, self.global_threshold)
            if threshold is None:
                continue
            result[cell] = float(np.mean([s <= threshold for s in scores]))
        return result

    def per_cell_set_size(self, prob_dicts: List[Dict[str, float]],
                           cell_labels: List[str]) -> Dict[str, float]:
        """Compute per-cell mean prediction set size."""
        by_cell = defaultdict(list)
        for probs, label in zip(prob_dicts, cell_labels):
            by_cell[label].append(len(self.predict_set(probs, label)))
        return {cell: float(np.mean(sizes)) for cell, sizes in by_cell.items()}

    def bias_map(self, nc_scores: List[float],
                 prob_dicts: List[Dict[str, float]],
                 cell_labels: List[str]) -> Dict[str, Dict]:
        """Compute full BiasMap: coverage + set_size + CVaR per cell."""
        from .metrics import compute_cvar
        by_cell = defaultdict(lambda: {"nc_scores": [], "probs": []})
        for score, probs, label in zip(nc_scores, prob_dicts, cell_labels):
            by_cell[label]["nc_scores"].append(score)
            by_cell[label]["probs"].append(probs)

        result = {}
        for cell, data in by_cell.items():
            threshold = self.cell_thresholds.get(cell, self.global_threshold) or 1.0
            scores = data["nc_scores"]
            result[cell] = {
                "n": len(scores),
                "threshold": threshold,
                "coverage": float(np.mean([s <= threshold for s in scores])),
                "mean_set_size": float(np.mean([
                    len(self.predict_set(p, cell)) for p in data["probs"]
                ])),
                "mean_nc_score": float(np.mean(scores)),
                "cvar_10": compute_cvar(scores, alpha=0.1),
                "worst_coverage_gap": float(max(0.0, (1 - self.alpha) - np.mean([s <= threshold for s in scores]))),
            }
        return result


class APS:
    """Adaptive Prediction Sets (Romano et al., 2020).

    Nonconformity score: sum of sorted probabilities until true class is included.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.threshold: Optional[float] = None

    def compute_nc_score(self, probs: Dict[str, float], true_answer: str) -> float:
        """APS nonconformity score for a single example."""
        sorted_probs = sorted(probs.values(), reverse=True)
        true_prob = probs.get(true_answer, 0.0)
        cumsum = 0.0
        for p in sorted_probs:
            cumsum += p
            if p <= true_prob + 1e-9:
                break
        return float(cumsum)

    def calibrate(self, nc_scores: List[float]) -> float:
        n = len(nc_scores)
        quantile_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.threshold = float(np.quantile(nc_scores, quantile_level))
        return self.threshold

    def predict_set(self, probs: Dict[str, float]) -> List[str]:
        if self.threshold is None:
            raise ValueError("Must calibrate first")
        sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        pred_set = []
        cumsum = 0.0
        for ans, p in sorted_items:
            pred_set.append(ans)
            cumsum += p
            if cumsum >= self.threshold:
                break
        return pred_set


class RAPS:
    """Regularized APS (Angelopoulos et al., 2021).

    Adds penalty for set size to discourage large prediction sets.
    """

    def __init__(self, alpha: float = 0.1, k_reg: int = 1, lam_reg: float = 0.01):
        self.alpha = alpha
        self.k_reg = k_reg
        self.lam_reg = lam_reg
        self.threshold: Optional[float] = None

    def compute_nc_score(self, probs: Dict[str, float], true_answer: str) -> float:
        sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        answers = [a for a, _ in sorted_items]
        if true_answer not in answers:
            return float("inf")
        rank = answers.index(true_answer)
        cumsum = sum(p for _, p in sorted_items[:rank + 1])
        penalty = self.lam_reg * max(0, rank + 1 - self.k_reg)
        return float(cumsum + penalty)

    def calibrate(self, nc_scores: List[float]) -> float:
        scores = [s for s in nc_scores if np.isfinite(s)]
        n = len(scores)
        quantile_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.threshold = float(np.quantile(scores, quantile_level))
        return self.threshold

    def predict_set(self, probs: Dict[str, float]) -> List[str]:
        if self.threshold is None:
            raise ValueError("Must calibrate first")
        sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        pred_set = []
        cumsum = 0.0
        for rank, (ans, p) in enumerate(sorted_items):
            pred_set.append(ans)
            cumsum += p
            penalty = self.lam_reg * max(0, rank + 1 - self.k_reg)
            if cumsum + penalty >= self.threshold:
                break
        return pred_set
