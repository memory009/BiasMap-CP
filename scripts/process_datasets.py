"""Process all datasets to unified SpatialQASample format.
Usage: python scripts/process_datasets.py [--dataset all|vsr|whatsup|...]
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.datasets.vsr import VSRDataset
from src.datasets.whatsup import WhatsUpDataset
from src.datasets.gqa_spatial import GQASpatialDataset
from src.utils.metadata import extract_metadata_batch

DATA_ROOT = "/LOCAL/psqhe8/BiasMap-CP/data"
RAW_ROOT = os.path.join(DATA_ROOT, "raw")
PROC_ROOT = os.path.join(DATA_ROOT, "processed")


# Option A benchmark: VSR + GQA + WhatsUp (NLVR2 dropped: two-image task, incompatible with single-image VLM)
DATASET_LOADERS = {
    "vsr": (VSRDataset, os.path.join(RAW_ROOT, "vsr")),
    "whatsup": (WhatsUpDataset, os.path.join(RAW_ROOT, "whatsup")),
    "gqa": (GQASpatialDataset, os.path.join(RAW_ROOT, "gqa")),
}

# SpatialSense is optional (requires manual download)
try:
    from src.datasets.spatialsense import SpatialSenseDataset
    DATASET_LOADERS["spatialsense"] = (SpatialSenseDataset, os.path.join(RAW_ROOT, "spatialsense"))
except ImportError:
    pass


def process_dataset(name: str, max_samples: int = None):
    cls, raw_dir = DATASET_LOADERS[name]
    if not os.path.exists(raw_dir):
        print(f"WARNING: {name} raw dir not found: {raw_dir}, skipping.")
        return []

    print(f"\n{'='*50}")
    print(f"Processing {name} from {raw_dir}")
    loader = cls(raw_dir)

    # Some loaders accept max_samples
    try:
        if max_samples:
            samples = loader.load(max_samples=max_samples)
        else:
            samples = loader.load()
    except TypeError:
        samples = loader.load()

    # Extract metadata
    print(f"  Extracting metadata for {len(samples)} samples...")
    samples = extract_metadata_batch(samples)

    # Save
    out_path = os.path.join(PROC_ROOT, f"{name}.jsonl")
    os.makedirs(PROC_ROOT, exist_ok=True)
    loader.save_processed(out_path)

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=["all", "vsr", "gqa", "whatsup"])
    parser.add_argument("--max_gqa", type=int, default=100000,
                        help="Max GQA spatial samples to extract")
    args = parser.parse_args()

    target_datasets = (list(DATASET_LOADERS.keys())
                       if args.dataset == "all"
                       else [args.dataset])

    all_samples = []
    for name in target_datasets:
        max_s = args.max_gqa if name == "gqa" else None
        samples = process_dataset(name, max_samples=max_s)
        all_samples.extend(samples)
        print(f"  {name}: {len(samples)} samples processed")

    print(f"\nTotal: {len(all_samples)} samples across {len(target_datasets)} datasets")
    print(f"Processed files in: {PROC_ROOT}")


if __name__ == "__main__":
    main()
