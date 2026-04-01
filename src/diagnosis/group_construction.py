"""
Group construction strategies for B2-v3.

Wraps MondrianPartition cells into coarser groups, exposing the same
`get_cell_by_features(sample) -> group_id` interface so the existing
CellCVaRWeighter and training loop work unchanged.

Two strategies:
  cluster_cvar  — KMeans on (mean_loss, loss_std, log_support)
  lossgroup_cvar — Equal-count bins sorted by mean_loss
"""
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional


class GroupPartition:
    """Wraps MondrianPartition: maps samples to coarser groups of cells."""

    def __init__(self, mondrian, cell_to_group: Dict[str, str],
                 group_info: Optional[dict] = None):
        """
        Args:
            mondrian: MondrianPartition (original fine-grained partition)
            cell_to_group: {mondrian_cell_id: group_id}
            group_info: optional metadata per group (for logging)
        """
        self.mondrian = mondrian
        self.cell_to_group = cell_to_group
        self.group_info = group_info or {}

        # Reverse mapping: group_id -> [cell_ids]
        self.groups: Dict[str, List[str]] = defaultdict(list)
        for cid, gid in cell_to_group.items():
            self.groups[gid].append(cid)

        # Compatibility: code that checks `partition.cells` or `len(partition.cells)`
        self.cells = {gid: None for gid in self.groups}

    def get_cell_by_features(self, sample: dict) -> Optional[str]:
        """Map sample -> Mondrian cell -> group."""
        cid = self.mondrian.get_cell_by_features(sample)
        if cid is None:
            return None
        return self.cell_to_group.get(cid)

    def get_group_composition(self) -> Dict[str, List[str]]:
        """Return {group_id: [mondrian_cell_ids]}."""
        return dict(self.groups)

    def summary(self) -> str:
        lines = [f"GroupPartition: {len(self.groups)} groups from "
                 f"{len(self.cell_to_group)} Mondrian cells"]
        for gid in sorted(self.groups.keys()):
            members = self.groups[gid]
            info = self.group_info.get(gid, {})
            mean_l = info.get("mean_loss", "?")
            if isinstance(mean_l, float):
                mean_l = f"{mean_l:.4f}"
            lines.append(f"  {gid}: {len(members)} cells, mean_loss={mean_l}")
        return "\n".join(lines)


def build_cluster_groups(mondrian, cell_losses: dict,
                         n_groups: int = 4,
                         cell_loss_lists: Optional[dict] = None):
    """Cluster Mondrian cells into groups via KMeans on cell-level features.

    Features: [mean_loss, loss_std, log1p(support)]  (standardised).
    Falls back to loss_std=0 if cell_loss_lists not provided.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    eligible = sorted(cell_losses.keys())
    n = len(eligible)
    if n <= n_groups:
        mapping = {cid: f"cluster_{i}" for i, cid in enumerate(eligible)}
        return GroupPartition(mondrian, mapping)

    features = []
    for cid in eligible:
        mean_loss = cell_losses[cid]
        support = mondrian.cells[cid].support if cid in mondrian.cells else 30
        std_loss = 0.0
        if cell_loss_lists and cid in cell_loss_lists:
            std_loss = float(np.std(cell_loss_lists[cid]))
        features.append([mean_loss, std_loss, np.log1p(support)])

    X = np.array(features)
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    mapping = {cid: f"cluster_{int(lab)}" for cid, lab in zip(eligible, labels)}

    info = {}
    for gid in sorted(set(mapping.values())):
        members = [c for c, g in mapping.items() if g == gid]
        info[gid] = {
            "members": members,
            "n_cells": len(members),
            "mean_loss": float(np.mean([cell_losses[c] for c in members])),
            "loss_range": [
                float(min(cell_losses[c] for c in members)),
                float(max(cell_losses[c] for c in members)),
            ],
        }

    return GroupPartition(mondrian, mapping, group_info=info)


def build_lossgroup_groups(mondrian, cell_losses: dict, n_groups: int = 4):
    """Bin Mondrian cells into equal-count groups sorted by mean loss.

    Bin 0 = lowest loss (easiest), bin n_groups-1 = highest loss (hardest).
    """
    sorted_cells = sorted(cell_losses.items(), key=lambda x: x[1])  # ascending
    n = len(sorted_cells)

    if n <= n_groups:
        mapping = {cid: f"lossbin_{i}" for i, (cid, _) in enumerate(sorted_cells)}
        return GroupPartition(mondrian, mapping)

    bin_size = n / n_groups
    mapping = {}
    for i, (cid, _) in enumerate(sorted_cells):
        bin_idx = min(int(i / bin_size), n_groups - 1)
        mapping[cid] = f"lossbin_{bin_idx}"

    info = {}
    for gid in sorted(set(mapping.values())):
        members = [c for c, g in mapping.items() if g == gid]
        info[gid] = {
            "members": members,
            "n_cells": len(members),
            "mean_loss": float(np.mean([cell_losses[c] for c in members])),
            "loss_range": [
                float(min(cell_losses[c] for c in members)),
                float(max(cell_losses[c] for c in members)),
            ],
        }

    return GroupPartition(mondrian, mapping, group_info=info)
