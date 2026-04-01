"""Run zero-shot VLM baselines for BiasMap-CP Sprint 1.

For each model × dataset × split:
  1. Load model
  2. Run inference, save ModelOutput (with logits/probs)
  3. Compute aggregate metrics
  4. Save to results/sprint1/{model}/{dataset}/{split}.jsonl

Usage:
  python scripts/run_baselines.py --model qwen2vl_2b --dataset all --split test
  python scripts/run_baselines.py --model all --dataset vsr --split all
  python scripts/run_baselines.py  # runs everything
"""
import os
import sys
import json
import glob
import argparse
import gc
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.datasets.base import SpatialQASample, BaseDataset
from src.models.qwen2vl import Qwen2VLModel
from src.models.llava import LLaVAModel
from src.models.blip2 import BLIP2Model
from src.evaluation.harness import EvaluationHarness

CACHE_DIR = "/LOCAL/psqhe8/hf_cache"
SPLITS_ROOT = "/LOCAL/psqhe8/BiasMap-CP/data/splits"
RESULTS_ROOT = "/LOCAL2/psqhe8/BiasMap-CP/results/sprint1"

MODEL_CONFIGS = {
    "qwen2vl_2b": {
        "cls": Qwen2VLModel,
        "hf_id": "Qwen/Qwen2-VL-2B-Instruct",
        "batch_size": 8,
    },
    "qwen2vl_7b": {
        "cls": Qwen2VLModel,
        "hf_id": "Qwen/Qwen2-VL-7B-Instruct",
        "batch_size": 4,
    },
    "llava_7b": {
        "cls": LLaVAModel,
        "hf_id": "llava-hf/llava-1.5-7b-hf",
        "batch_size": 4,
    },
    "blip2": {
        "cls": BLIP2Model,
        "hf_id": "Salesforce/blip2-opt-2.7b",
        "batch_size": 8,
    },
}

SPLIT_NAMES = [
    "test", "cal",
    "ood_frame", "ood_concept", "ood_tailrisk",
    "ood_compositional", "ood_shifted_cal_test",
]


def load_split(split_name: str):
    path = os.path.join(SPLITS_ROOT, f"{split_name}.jsonl")
    if not os.path.exists(path):
        print(f"  WARNING: split not found: {path}")
        return []
    samples = BaseDataset.load_processed(path)
    print(f"  Loaded {split_name}: {len(samples)} samples")
    return samples


def run_model(model_name: str, target_splits: list,
              max_per_split: int = None, resume: bool = True):
    cfg = MODEL_CONFIGS[model_name]
    model = cfg["cls"](
        model_id=cfg["hf_id"],
        cache_dir=CACHE_DIR,
        dtype="bfloat16",
    )
    model.load()

    harness = EvaluationHarness(
        model=model,
        results_dir=RESULTS_ROOT,
        alpha=0.1,
    )

    for split_name in target_splits:
        samples = load_split(split_name)
        if not samples:
            continue
        if max_per_split:
            samples = samples[:max_per_split]

        # Group by dataset for per-dataset metrics
        from collections import defaultdict
        by_dataset = defaultdict(list)
        for s in samples:
            by_dataset[s.dataset].append(s)

        for dataset_name, ds_samples in by_dataset.items():
            print(f"\n--- {model_name} | {dataset_name} | {split_name} ---")
            harness.run(
                samples=ds_samples,
                split_name=split_name,
                dataset_name=dataset_name,
                batch_size=cfg["batch_size"],
                resume=resume,
            )

    # Free GPU memory
    del model.model
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run_conformal_analysis(model_name: str):
    """After baselines, run conformal analysis on cal+test splits."""
    from src.evaluation.harness import EvaluationHarness
    from src.models.qwen2vl import Qwen2VLModel  # dummy model for harness

    # Load cal and test outputs
    model_results_dir = os.path.join(RESULTS_ROOT, model_name.split("/")[-1])
    if not os.path.exists(model_results_dir):
        print(f"No results for {model_name}, skipping conformal analysis.")
        return

    cal_samples = load_split("cal")
    test_samples = load_split("test")

    for dataset_name in os.listdir(model_results_dir):
        ds_dir = os.path.join(model_results_dir, dataset_name)
        cal_path = os.path.join(ds_dir, "cal.jsonl")
        test_path = os.path.join(ds_dir, "test.jsonl")

        if not os.path.exists(cal_path) or not os.path.exists(test_path):
            continue

        from src.datasets.base import ModelOutput
        cal_outputs, test_outputs = [], []
        for path, lst in [(cal_path, cal_outputs), (test_path, test_outputs)]:
            with open(path) as f:
                for line in f:
                    try:
                        lst.append(ModelOutput.from_dict(json.loads(line)))
                    except Exception:
                        pass

        if not cal_outputs or not test_outputs:
            continue

        # Filter samples to this dataset
        ds_cal = [s for s in cal_samples if s.dataset == dataset_name]
        ds_test = [s for s in test_samples if s.dataset == dataset_name]

        dummy_model = Qwen2VLModel("dummy")
        harness = EvaluationHarness(dummy_model, RESULTS_ROOT)
        cp_out_path = os.path.join(
            RESULTS_ROOT, "conformal",
            model_name, f"{dataset_name}_cp_analysis.json"
        )
        harness.run_conformal_analysis(
            cal_outputs, test_outputs, ds_cal, ds_test, cp_out_path
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=["all"] + list(MODEL_CONFIGS.keys()))
    parser.add_argument("--split", default="all",
                        choices=["all"] + SPLIT_NAMES)
    parser.add_argument("--max_per_split", type=int, default=None,
                        help="Limit samples per split (for quick testing)")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--conformal_only", action="store_true",
                        help="Only run conformal analysis on existing results")
    args = parser.parse_args()

    target_models = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    target_splits = SPLIT_NAMES if args.split == "all" else [args.split]

    if args.conformal_only:
        for model_name in target_models:
            hf_id = MODEL_CONFIGS[model_name]["hf_id"].split("/")[-1]
            run_conformal_analysis(hf_id)
        return

    for model_name in target_models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        run_model(
            model_name,
            target_splits,
            max_per_split=args.max_per_split,
            resume=not args.no_resume,
        )

    # Run conformal analysis
    print("\n=== Running Conformal Analysis ===")
    for model_name in target_models:
        hf_id = MODEL_CONFIGS[model_name]["hf_id"].split("/")[-1]
        run_conformal_analysis(hf_id)

    print("\n=== Sprint 1 Complete ===")
    print(f"Results in: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
