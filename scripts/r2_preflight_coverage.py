#!/usr/bin/env python3
"""
R2 preflight — per-hard-cell named-object match coverage.

Goal: before writing the full R2 slot+gate code, report whether there are
enough usable samples per hard-4 cell for the gate+slot path. A sample is
"usable" if the GQA scene graph has >=1 named object that can be matched to
the question's semantic parse (i.e., n_match >= 1 under D1's matching logic).

We report per cell:
- |cell|                 : number of worst-cell samples in train split
- |cell with n_match>=1| : samples with at least one matched named object
- match_rate             : |matched| / |cell|
- mean_n_match           : average matched objects (capped at K=6) among matched
- |cell with n_match>=2| : samples with at least one pair (D1 pair-aux has signal)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Reuse pilot script's constants + helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.pilot_depth_object_level import (
    WORST_CELLS,
    SPLITS_DIR,
    GQA_QUESTIONS_FILE,
    GQA_SCENE_GRAPHS,
    get_cell_id,
    load_worst_cell_samples,
)

K_MAX = 6


def extract_named_objects_from_semantic(raw_entry):
    """Collect all named-object IDs referenced in the GQA semantic parse.

    Returns a list of object IDs in mention order (with repeats removed).
    """
    semantic = raw_entry.get("semantic", [])
    ordered_ids: list[str] = []
    seen = set()
    for step in semantic:
        arg = step.get("argument", "")
        for m in re.finditer(r"\((\d+)\)", arg):
            oid = m.group(1)
            if oid not in seen:
                seen.add(oid)
                ordered_ids.append(oid)
    return ordered_ids


def match_named_objects(raw_entry, sg_objects):
    """Return matched objects (id, name, bbox) list, up to K_MAX.

    Matches require the object ID to exist in the scene graph and to have a
    valid bbox. This is the D1 n_match signal.
    """
    ids = extract_named_objects_from_semantic(raw_entry)
    matched = []
    for oid in ids:
        if oid in sg_objects:
            sg = sg_objects[oid]
            bbox = (sg.get("x"), sg.get("y"), sg.get("w"), sg.get("h"))
            if None not in bbox and bbox[2] and bbox[3]:
                matched.append(
                    {
                        "id": oid,
                        "name": sg.get("name", "object"),
                        "bbox": bbox,
                    }
                )
                if len(matched) >= K_MAX:
                    break
    return matched


def main():
    print("=" * 70)
    print("R2 preflight — per-hard-cell named-object match coverage")
    print(f"Hard-4 cells: {WORST_CELLS}")
    print(f"K_MAX = {K_MAX}")
    print("=" * 70)

    # Load worst-cell samples
    train_samples = load_worst_cell_samples("train")
    print(f"\nLoaded {len(train_samples)} train samples across hard-4 cells")

    # Load GQA raw questions + scene graphs (V2 cache)
    print(f"Loading GQA raw questions from {GQA_QUESTIONS_FILE} ...")
    with open(GQA_QUESTIONS_FILE) as f:
        raw_questions = json.load(f)
    print(f"  {len(raw_questions)} raw GQA questions")

    print(f"Loading GQA scene graphs from {GQA_SCENE_GRAPHS} ...")
    with open(GQA_SCENE_GRAPHS) as f:
        scene_graphs = json.load(f)
    print(f"  {len(scene_graphs)} scene graphs")

    # Per-cell accumulators
    per_cell = defaultdict(
        lambda: {
            "n": 0,
            "n_match_ge1": 0,
            "n_match_ge2": 0,
            "sum_n_match": 0,
            "miss_reason": Counter(),
        }
    )

    for s in train_samples:
        cell = get_cell_id(s)
        per_cell[cell]["n"] += 1

        raw_id = s.get("id", "")
        # Split generator prefixes GQA ids with "gqa_"; raw_questions is keyed
        # by the bare numeric id (e.g. "061046996").
        qid = raw_id.split("_", 1)[1] if raw_id.startswith("gqa_") else raw_id
        if qid not in raw_questions:
            per_cell[cell]["miss_reason"]["no_raw_entry"] += 1
            continue
        raw_entry = raw_questions[qid]
        image_id = raw_entry.get("imageId") or raw_entry.get("image_id")
        if image_id is None or image_id not in scene_graphs:
            per_cell[cell]["miss_reason"]["no_scene_graph"] += 1
            continue
        sg_objects = scene_graphs[image_id].get("objects", {})
        matched = match_named_objects(raw_entry, sg_objects)
        n_match = len(matched)
        if n_match >= 1:
            per_cell[cell]["n_match_ge1"] += 1
            per_cell[cell]["sum_n_match"] += n_match
        if n_match >= 2:
            per_cell[cell]["n_match_ge2"] += 1
        if n_match == 0:
            per_cell[cell]["miss_reason"]["zero_match"] += 1

    # Report
    print("\n" + "=" * 70)
    print("PER-CELL COVERAGE")
    print("=" * 70)
    print(
        f"{'cell':<28} {'|cell|':>7} {'>=1':>7} {'rate':>8} {'mean_k':>9} {'>=2':>7}"
    )
    print("-" * 70)
    aggregate = Counter()
    for cell in WORST_CELLS:
        d = per_cell[cell]
        n = d["n"]
        g1 = d["n_match_ge1"]
        g2 = d["n_match_ge2"]
        mean_k = (d["sum_n_match"] / g1) if g1 else 0.0
        rate = (g1 / n) if n else 0.0
        print(
            f"{cell:<28} {n:>7d} {g1:>7d} {rate:>8.3f} {mean_k:>9.2f} {g2:>7d}"
        )
        aggregate["n"] += n
        aggregate["g1"] += g1
        aggregate["g2"] += g2
        aggregate["sum_k"] += d["sum_n_match"]

    n_all = aggregate["n"]
    g1_all = aggregate["g1"]
    g2_all = aggregate["g2"]
    mean_k_all = (aggregate["sum_k"] / g1_all) if g1_all else 0.0
    rate_all = (g1_all / n_all) if n_all else 0.0
    print("-" * 70)
    print(
        f"{'ALL hard-4':<28} {n_all:>7d} {g1_all:>7d} {rate_all:>8.3f} {mean_k_all:>9.2f} {g2_all:>7d}"
    )

    print("\nMiss-reason breakdown (per cell):")
    for cell in WORST_CELLS:
        reasons = per_cell[cell]["miss_reason"]
        if reasons:
            print(f"  {cell}: {dict(reasons)}")

    # Save JSON artifact
    out_path = Path("refine-logs/r2_preflight_coverage.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "k_max": K_MAX,
        "per_cell": {
            cell: {
                "n": per_cell[cell]["n"],
                "n_match_ge1": per_cell[cell]["n_match_ge1"],
                "n_match_ge2": per_cell[cell]["n_match_ge2"],
                "match_rate": (
                    per_cell[cell]["n_match_ge1"] / per_cell[cell]["n"]
                    if per_cell[cell]["n"]
                    else 0.0
                ),
                "mean_n_match_among_matched": (
                    per_cell[cell]["sum_n_match"] / per_cell[cell]["n_match_ge1"]
                    if per_cell[cell]["n_match_ge1"]
                    else 0.0
                ),
                "miss_reason": dict(per_cell[cell]["miss_reason"]),
            }
            for cell in WORST_CELLS
        },
        "aggregate": {
            "n": n_all,
            "n_match_ge1": g1_all,
            "n_match_ge2": g2_all,
            "match_rate": rate_all,
            "mean_n_match_among_matched": mean_k_all,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out_path}")

    # Recommendation
    print("\nRECOMMENDATION:")
    if rate_all >= 0.80:
        print("  Match rate >= 80% aggregate; gate+slot path has enough samples.")
    elif rate_all >= 0.50:
        print("  Match rate in [50%, 80%); usable but expect some gate dropout.")
    else:
        print("  Match rate < 50%; reconsider matching criterion (e.g., relax to name match).")

    low_cells = [
        c for c in WORST_CELLS if per_cell[c]["n"] and per_cell[c]["n_match_ge1"] / per_cell[c]["n"] < 0.50
    ]
    if low_cells:
        print(f"  Cells below 50% match rate: {low_cells}")


if __name__ == "__main__":
    main()
