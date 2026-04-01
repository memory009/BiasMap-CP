"""Quick baseline run: sample 500 per split to validate pipeline end-to-end.
Usage: python scripts/run_baselines_quick.py --model qwen2vl_2b
"""
import os, sys, json, gc, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.datasets.base import SpatialQASample, ModelOutput, BaseDataset
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2vl_2b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--n_per_split", type=int, default=500)
    parser.add_argument("--splits", default="test,cal,ood_frame,ood_tailrisk",
                        help="comma-separated split names")
    args = parser.parse_args()

    import random; random.seed(42)

    cfg = MODEL_CONFIGS[args.model]
    print(f"\nLoading {args.model} ({cfg['hf_id']})...")
    model = cfg["cls"](cfg["hf_id"], cache_dir=CACHE_DIR, dtype="bfloat16")
    model.load()

    harness = EvaluationHarness(model, RESULTS_ROOT, alpha=0.1)

    for split_name in args.splits.split(","):
        path = os.path.join(SPLITS_ROOT, f"{split_name}.jsonl")
        if not os.path.exists(path):
            print(f"  Split not found: {split_name}")
            continue
        samples = BaseDataset.load_processed(path)
        random.shuffle(samples)
        samples = samples[:args.n_per_split]

        from collections import defaultdict
        by_dataset = defaultdict(list)
        for s in samples:
            by_dataset[s.dataset].append(s)

        for ds_name, ds_samples in by_dataset.items():
            harness.run(
                samples=ds_samples,
                split_name=split_name,
                dataset_name=ds_name,
                batch_size=cfg["batch_size"],
                resume=True,
            )

    del model.model; del model; gc.collect(); torch.cuda.empty_cache()
    print("\nQuick run done.")


if __name__ == "__main__":
    main()
