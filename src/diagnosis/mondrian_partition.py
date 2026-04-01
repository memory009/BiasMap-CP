"""
Pre-registered Mondrian partition over discrete features.
Features: relation_type × depth_ambiguity × dataset_source.

Cells with support < min_support are backed off to parent
(relation_type × depth_ambiguity).
"""
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class MondrianCell:
    cell_id: str
    features: Dict[str, str]
    sample_ids: List[str] = field(default_factory=list)
    support: int = 0
    parent_id: Optional[str] = None
    is_backed_off: bool = False


class MondrianPartition:
    """
    Pre-registered partition: relation_type × depth_ambiguity × dataset.
    Cells with support < min_support back off to parent
    (relation_type × depth_ambiguity).
    """

    def __init__(self, min_support: int = 30, shrinkage_threshold: int = 50):
        self.min_support = min_support
        self.shrinkage_threshold = shrinkage_threshold
        self.cells: Dict[str, MondrianCell] = {}
        self.sample_to_cell: Dict[str, str] = {}

    # ------------------------------------------------------------------
    def build(self, samples: List[dict]) -> Dict[str, MondrianCell]:
        """Build partition from a list of sample dicts (loaded from JSONL)."""
        # Finest level: relation_type | depth_ambiguity | dataset
        fine_cells: Dict[str, List[str]] = defaultdict(list)
        for s in samples:
            rel = s.get("relation_type", "unknown")
            depth = str(s.get("depth_ambiguity", False))
            dataset = s.get("dataset", "unknown")
            key = f"{rel}|{depth}|{dataset}"
            fine_cells[key].append(s["id"])

        # Parent level: relation_type | depth_ambiguity
        parent_members: Dict[str, List[str]] = defaultdict(list)
        for key, ids in fine_cells.items():
            parent_key = "|".join(key.split("|")[:2])
            parent_members[parent_key].extend(ids)

        self.cells.clear()
        self.sample_to_cell.clear()
        backed_off = 0

        for key, ids in fine_cells.items():
            parts = key.split("|")
            parent_key = f"{parts[0]}|{parts[1]}"

            if len(ids) >= self.min_support:
                cell = MondrianCell(
                    cell_id=key,
                    features={"relation_type": parts[0],
                              "depth_ambiguity": parts[1],
                              "dataset": parts[2]},
                    sample_ids=list(ids),
                    support=len(ids),
                    parent_id=parent_key,
                    is_backed_off=False,
                )
                self.cells[key] = cell
                for sid in ids:
                    self.sample_to_cell[sid] = key
            else:
                backed_off += 1
                if parent_key not in self.cells:
                    pp = parent_key.split("|")
                    self.cells[parent_key] = MondrianCell(
                        cell_id=parent_key,
                        features={"relation_type": pp[0],
                                  "depth_ambiguity": pp[1],
                                  "dataset": "merged"},
                        sample_ids=[],
                        support=0,
                        parent_id=None,
                        is_backed_off=True,
                    )
                self.cells[parent_key].sample_ids.extend(ids)
                self.cells[parent_key].support += len(ids)
                for sid in ids:
                    self.sample_to_cell[sid] = parent_key

        print(f"Partition built: {len(self.cells)} cells "
              f"({backed_off} fine cells backed off)")
        return self.cells

    # ------------------------------------------------------------------
    def get_cell(self, sample_id: str) -> Optional[str]:
        return self.sample_to_cell.get(sample_id)

    def get_cell_by_features(self, sample: dict) -> Optional[str]:
        """Map any sample to a cell by its features (works for samples
        not in the original partition, e.g. train samples)."""
        rel = sample.get("relation_type", "unknown")
        depth = str(sample.get("depth_ambiguity", False))
        dataset = sample.get("dataset", "unknown")
        # Try finest level first
        fine_key = f"{rel}|{depth}|{dataset}"
        if fine_key in self.cells:
            return fine_key
        # Fall back to parent
        parent_key = f"{rel}|{depth}"
        if parent_key in self.cells:
            return parent_key
        return None

    def get_cell_stats(self) -> Dict[str, dict]:
        return {
            cid: {
                "cell_id": cid,
                "features": cell.features,
                "support": cell.support,
                "is_backed_off": cell.is_backed_off,
            }
            for cid, cell in self.cells.items()
        }

    # ------------------------------------------------------------------
    def save(self, path: Path):
        data = {
            "min_support": self.min_support,
            "shrinkage_threshold": self.shrinkage_threshold,
            "cells": self.get_cell_stats(),
            "sample_to_cell": self.sample_to_cell,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "MondrianPartition":
        with open(path) as f:
            data = json.load(f)
        part = cls(data["min_support"], data["shrinkage_threshold"])
        part.sample_to_cell = data["sample_to_cell"]
        # Rebuild cell objects
        for cid, stats in data["cells"].items():
            sids = [sid for sid, c in part.sample_to_cell.items() if c == cid]
            part.cells[cid] = MondrianCell(
                cell_id=cid,
                features=stats["features"],
                sample_ids=sids,
                support=stats["support"],
                parent_id=None,
                is_backed_off=stats["is_backed_off"],
            )
        return part
