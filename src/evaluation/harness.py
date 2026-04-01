"""Main evaluation harness for BiasMap-CP Sprint 1."""
import os
import json
import time
from typing import List, Dict, Optional
from collections import defaultdict

from ..datasets.base import SpatialQASample, ModelOutput
from ..models.base_vlm import BaseVLM
from .metrics import compute_metrics, compute_per_relation_metrics, compute_cvar
from .conformal import SplitCP, MondrianCP, APS, RAPS


class EvaluationHarness:
    """Runs model inference + conformal analysis across datasets and splits."""

    def __init__(self, model: BaseVLM, results_dir: str,
                 alpha: float = 0.1):
        self.model = model
        self.results_dir = results_dir
        self.alpha = alpha
        os.makedirs(results_dir, exist_ok=True)

    def run(self, samples: List[SpatialQASample],
            split_name: str,
            dataset_name: str,
            batch_size: int = 8,
            resume: bool = True) -> List[ModelOutput]:
        """Run inference on samples and save results."""
        model_name = self.model.model_id.split("/")[-1]
        out_dir = os.path.join(self.results_dir, model_name, dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{split_name}.jsonl")
        logits_path = os.path.join(out_dir, f"{split_name}_logits.jsonl")

        # Resume from checkpoint
        done_ids = set()
        existing_outputs = []
        if resume and os.path.exists(out_path):
            with open(out_path) as f:
                for line in f:
                    try:
                        o = ModelOutput.from_dict(json.loads(line))
                        done_ids.add(o.sample_id)
                        existing_outputs.append(o)
                    except Exception:
                        pass
            print(f"  Resuming: {len(done_ids)} already done")

        remaining = [s for s in samples if s.id not in done_ids]
        print(f"\nRunning {model_name} on {dataset_name}/{split_name}: "
              f"{len(remaining)} samples (batch_size={batch_size})")

        new_outputs = []
        with open(out_path, "a") as f_out, open(logits_path, "a") as f_logits:
            for i in range(0, len(remaining), batch_size):
                batch = remaining[i:i + batch_size]
                t0 = time.time()
                for sample in batch:
                    try:
                        out = self.model.predict_sample(sample)
                        out.split = split_name
                    except Exception as e:
                        print(f"    Error on {sample.id}: {e}")
                        out = self.model._error_output(sample)
                        out.split = split_name
                    f_out.write(json.dumps(out.to_dict()) + "\n")
                    f_logits.write(json.dumps({
                        "id": out.sample_id,
                        "logits": out.logits,
                        "probs": out.probabilities,
                        "nc_score": out.nonconformity_score,
                    }) + "\n")
                    new_outputs.append(out)

                elapsed = time.time() - t0
                done = i + len(batch)
                print(f"  {done}/{len(remaining)} | {elapsed:.1f}s | "
                      f"acc so far: {sum(o.correct for o in new_outputs)/len(new_outputs):.3f}")

        all_outputs = existing_outputs + new_outputs

        # Save aggregate metrics
        metrics = compute_metrics(all_outputs)
        metrics_path = os.path.join(out_dir, f"{split_name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics: {metrics}")

        return all_outputs

    def run_conformal_analysis(self,
                                cal_outputs: List[ModelOutput],
                                test_outputs: List[ModelOutput],
                                cal_samples: List[SpatialQASample],
                                test_samples: List[SpatialQASample],
                                output_path: str) -> Dict:
        """Run full conformal analysis: global CP + Mondrian CP per relation_type."""
        # Build lookup maps
        cal_by_id = {s.id: s for s in cal_samples}
        test_by_id = {s.id: s for s in test_samples}

        cal_nc = [o.nonconformity_score for o in cal_outputs]
        test_nc = [o.nonconformity_score for o in test_outputs]
        cal_probs = [o.probabilities for o in cal_outputs]
        test_probs = [o.probabilities for o in test_outputs]

        cal_rels = [cal_by_id.get(o.sample_id, None) for o in cal_outputs]
        test_rels = [test_by_id.get(o.sample_id, None) for o in test_outputs]
        cal_rel_labels = [s.relation_type if s else "unknown" for s in cal_rels]
        test_rel_labels = [s.relation_type if s else "unknown" for s in test_rels]

        results = {}

        # 1. Global split-CP
        global_cp = SplitCP(self.alpha)
        global_cp.calibrate(cal_nc)
        results["global_split_cp"] = {
            "threshold": global_cp.threshold,
            "cal_coverage": global_cp.coverage(cal_nc),
            "test_coverage": global_cp.coverage(test_nc),
            "test_mean_set_size": global_cp.mean_set_size(test_probs),
        }

        # 2. APS
        aps = APS(self.alpha)
        aps_cal_nc = [aps.compute_nc_score(o.probabilities, cal_samples[i].answer)
                      for i, o in enumerate(cal_outputs)
                      if i < len(cal_samples)]
        if aps_cal_nc:
            aps.calibrate(aps_cal_nc)
            results["aps"] = {"threshold": aps.threshold}

        # 3. RAPS
        raps = RAPS(self.alpha)
        raps_cal_nc = [raps.compute_nc_score(o.probabilities, cal_samples[i].answer)
                       for i, o in enumerate(cal_outputs)
                       if i < len(cal_samples)]
        if raps_cal_nc:
            raps.calibrate(raps_cal_nc)
            results["raps"] = {"threshold": raps.threshold}

        # 4. Mondrian CP by relation_type
        mondrian_cp = MondrianCP(self.alpha, min_cell_size=30)
        mondrian_cp.calibrate(cal_nc, cal_rel_labels)
        results["mondrian_cp"] = {
            "cell_thresholds": mondrian_cp.cell_thresholds,
            "test_per_cell_coverage": mondrian_cp.per_cell_coverage(test_nc, test_rel_labels),
            "test_per_cell_set_size": mondrian_cp.per_cell_set_size(test_probs, test_rel_labels),
            "bias_map": mondrian_cp.bias_map(test_nc, test_probs, test_rel_labels),
        }

        # 5. CVaR of nc_scores
        results["cvar"] = {
            "cal_cvar_10": compute_cvar(cal_nc, 0.1),
            "test_cvar_10": compute_cvar(test_nc, 0.1),
            "test_cvar_20": compute_cvar(test_nc, 0.2),
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nConformal analysis saved to {output_path}")

        return results

    def summarize_results(self, results_dir: str) -> Dict:
        """Collect all metric files into a summary table."""
        summary = {}
        for model_dir in os.listdir(results_dir):
            model_path = os.path.join(results_dir, model_dir)
            if not os.path.isdir(model_path):
                continue
            summary[model_dir] = {}
            for dataset_dir in os.listdir(model_path):
                dataset_path = os.path.join(model_path, dataset_dir)
                if not os.path.isdir(dataset_path):
                    continue
                summary[model_dir][dataset_dir] = {}
                for fname in os.listdir(dataset_path):
                    if fname.endswith("_metrics.json"):
                        split = fname.replace("_metrics.json", "")
                        with open(os.path.join(dataset_path, fname)) as f:
                            summary[model_dir][dataset_dir][split] = json.load(f)
        return summary
