"""Generate standard + 5 OOD splits for BiasMap-CP.
Usage: python scripts/generate_splits.py
"""
import os
import sys
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.datasets.base import SpatialQASample, BaseDataset
from src.datasets.split_generator import generate_all_splits

DATA_ROOT = "/LOCAL/psqhe8/BiasMap-CP/data"
PROC_ROOT = os.path.join(DATA_ROOT, "processed")
SPLITS_ROOT = os.path.join(DATA_ROOT, "splits")


def load_all_processed(proc_dir: str):
    """Load all processed JSONL files."""
    all_samples = []
    jsonl_files = glob.glob(os.path.join(proc_dir, "*.jsonl"))

    for fpath in sorted(jsonl_files):
        samples = BaseDataset.load_processed(fpath)
        print(f"  {os.path.basename(fpath)}: {len(samples)} samples")
        all_samples.extend(samples)

    return all_samples


def main():
    print(f"Loading processed samples from {PROC_ROOT}")
    all_samples = load_all_processed(PROC_ROOT)

    if not all_samples:
        print("ERROR: No processed samples found. Run process_datasets.py first.")
        sys.exit(1)

    print(f"\nTotal samples: {len(all_samples)}")

    # Show dataset distribution
    from collections import Counter
    dataset_counts = Counter(s.dataset for s in all_samples)
    print("Dataset distribution:")
    for ds, cnt in sorted(dataset_counts.items()):
        print(f"  {ds}: {cnt}")

    # Generate all splits
    splits = generate_all_splits(
        all_samples,
        output_dir=SPLITS_ROOT,
        target_ood_size=4000,
        seed=42,
    )

    print(f"\nSplits saved to {SPLITS_ROOT}")
    print("Summary:")
    for name, ss in splits.items():
        print(f"  {name}: {len(ss)} samples")


if __name__ == "__main__":
    main()
