"""Sprint 1 Bonus: Preliminary conformal prediction analysis.

Validates the core hypothesis:
  "Set sizes for front/behind and near/far should be much larger than
   left/right and above/below — confirming spatial bias."

Usage: python scripts/preliminary_cp_analysis.py --model qwen2vl_2b
"""
import os
import sys
import json
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.datasets.base import SpatialQASample, ModelOutput, BaseDataset
from src.evaluation.conformal import SplitCP, MondrianCP
from src.evaluation.metrics import compute_cvar
from src.utils.visualization import plot_bias_map

SPLITS_ROOT = "/LOCAL/psqhe8/BiasMap-CP/data/splits"
RESULTS_ROOT = "/LOCAL2/psqhe8/BiasMap-CP/results/sprint1"


def load_model_outputs(model_name: str, split: str) -> list:
    """Load all ModelOutput files for a model+split across datasets."""
    pattern = os.path.join(RESULTS_ROOT, model_name, "*", f"{split}.jsonl")
    files = glob.glob(pattern)
    outputs = []
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                try:
                    outputs.append(ModelOutput.from_dict(json.loads(line)))
                except Exception:
                    pass
    return outputs


def run_preliminary_analysis(model_name: str, alpha: float = 0.1):
    print(f"\n{'='*60}")
    print(f"Preliminary CP Analysis: {model_name}")
    print(f"{'='*60}")

    # Load splits
    cal_samples = BaseDataset.load_processed(os.path.join(SPLITS_ROOT, "cal.jsonl"))
    test_samples = BaseDataset.load_processed(os.path.join(SPLITS_ROOT, "test.jsonl"))
    cal_by_id = {s.id: s for s in cal_samples}
    test_by_id = {s.id: s for s in test_samples}

    cal_outputs = load_model_outputs(model_name, "cal")
    test_outputs = load_model_outputs(model_name, "test")

    if not cal_outputs:
        print(f"No cal outputs found for {model_name}. Skipping.")
        return {}

    print(f"Cal outputs: {len(cal_outputs)}, Test outputs: {len(test_outputs)}")

    cal_nc = [o.nonconformity_score for o in cal_outputs]
    test_nc = [o.nonconformity_score for o in test_outputs]

    # Get relation labels
    cal_rels = [cal_by_id.get(o.sample_id, None) for o in cal_outputs]
    test_rels = [test_by_id.get(o.sample_id, None) for o in test_outputs]
    cal_rel_labels = [s.relation_type if s else "unknown" for s in cal_rels]
    test_rel_labels = [s.relation_type if s else "unknown" for s in test_rels]

    # 1. Global split-CP
    global_cp = SplitCP(alpha)
    global_cp.calibrate(cal_nc)
    global_coverage = global_cp.coverage(test_nc)
    global_set_size = global_cp.mean_set_size([o.probabilities for o in test_outputs])

    print(f"\nGlobal Split-CP (α={alpha}):")
    print(f"  Threshold: {global_cp.threshold:.4f}")
    print(f"  Test coverage: {global_coverage:.4f} (target: {1-alpha:.1f})")
    print(f"  Mean set size: {global_set_size:.3f}")

    # 2. Mondrian CP by relation type
    mondrian_cp = MondrianCP(alpha, min_cell_size=20)
    mondrian_cp.calibrate(cal_nc, cal_rel_labels)

    test_probs = [o.probabilities for o in test_outputs]
    bias_map = mondrian_cp.bias_map(test_nc, test_probs, test_rel_labels)

    print(f"\nMondrian CP — Per-Relation Bias Map:")
    print(f"{'Relation':<20} {'N':>6} {'Coverage':>10} {'Set Size':>10} {'CVaR10':>10} {'CovGap':>10}")
    print("-" * 70)
    for rel in sorted(bias_map.keys()):
        cell = bias_map[rel]
        print(f"{rel:<20} {cell['n']:>6} {cell['coverage']:>10.4f} "
              f"{cell['mean_set_size']:>10.3f} {cell['cvar_10']:>10.4f} "
              f"{cell['worst_coverage_gap']:>10.4f}")

    # 3. Plot bias map
    out_dir = os.path.join(RESULTS_ROOT, "preliminary_cp", model_name)
    os.makedirs(out_dir, exist_ok=True)

    plot_bias_map(
        bias_map,
        output_path=os.path.join(out_dir, "bias_map.png"),
        title=f"BiasMap — {model_name} (α={alpha})",
    )

    # 4. Hypothesis check
    easy_rels = {"left", "right", "above", "below"}
    hard_rels = {"in_front_of", "behind", "near", "far"}

    easy_sizes = [bias_map[r]["mean_set_size"] for r in easy_rels if r in bias_map]
    hard_sizes = [bias_map[r]["mean_set_size"] for r in hard_rels if r in bias_map]

    hypothesis_confirmed = False
    if easy_sizes and hard_sizes:
        avg_easy = np.mean(easy_sizes)
        avg_hard = np.mean(hard_sizes)
        hypothesis_confirmed = avg_hard > avg_easy
        print(f"\nHypothesis Check:")
        print(f"  Easy relations (left/right/above/below) avg set size: {avg_easy:.3f}")
        print(f"  Hard relations (front/behind/near/far) avg set size:  {avg_hard:.3f}")
        print(f"  Hypothesis {'CONFIRMED ✓' if hypothesis_confirmed else 'NOT confirmed ✗'}: "
              f"hard > easy by {avg_hard - avg_easy:.3f}")

    # 5. Save summary
    summary = {
        "model": model_name,
        "global_cp": {
            "threshold": global_cp.threshold,
            "test_coverage": global_coverage,
            "mean_set_size": global_set_size,
        },
        "mondrian_cp": bias_map,
        "hypothesis_confirmed": bool(hypothesis_confirmed),
        "global_cvar_10": compute_cvar(test_nc, 0.1),
    }
    with open(os.path.join(out_dir, "cp_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen2-VL-2B-Instruct",
                        help="Model name (as in results dir)")
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    if args.model == "all":
        model_dirs = [d for d in os.listdir(RESULTS_ROOT)
                      if os.path.isdir(os.path.join(RESULTS_ROOT, d))]
        for model_name in model_dirs:
            run_preliminary_analysis(model_name, args.alpha)
    else:
        run_preliminary_analysis(args.model, args.alpha)


if __name__ == "__main__":
    main()
