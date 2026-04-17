#!/usr/bin/env python3
"""
Zero-shot evaluation for Sprint 2 / 8B scale-up comparison.

Loads a VLM WITHOUT any LoRA checkpoint, runs loss-based and generation-based
eval on repair_val, and saves structured JSON results. Reuses eval logic
from run_b2v3.py via direct import.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_zeroshot.py
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_zeroshot.py --model_id Qwen/Qwen3-VL-2B-Instruct
"""
import argparse
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.diagnosis.mondrian_partition import MondrianPartition

# Reuse helpers from run_b2v3 (same eval logic as training runs)
from scripts.run_b2v3 import (
    load_vlm_model, load_samples, evaluate_loss_cvar, evaluate_accuracy,
    CACHE_DIR, B1_DIR,
)

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_SPLIT = "repair_val"
DEFAULT_OUT   = "results/sprint2/b2v3"


def main():
    parser = argparse.ArgumentParser(description="Zero-shot eval for Sprint 2")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    # Output dir: e.g. results/sprint2/b2v3/qwen3vl8b_zeroshot/
    tag = args.model_id.split("/")[-1].lower().replace("-instruct", "").replace("-", "")
    out_dir = Path(args.output_dir) / f"{tag}_zeroshot"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Zero-shot eval: {args.model_id}")
    print(f"Split: {args.split}")
    print(f"Output: {out_dir}")
    print("=" * 60)

    # Load model (no LoRA)
    from transformers import AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info

    print("\nLoading model (no LoRA)...")
    t0 = time.time()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_id, cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    model = load_vlm_model(
        args.model_id, cache_dir=CACHE_DIR, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Load data + partition
    print(f"\nLoading {args.split} samples...")
    eval_samples = load_samples(args.split)
    print(f"  {len(eval_samples)} samples")

    partition = MondrianPartition.load(B1_DIR / "partition.json")

    # Loss eval
    print("\nRunning loss eval...")
    t0 = time.time()
    loss_metrics = evaluate_loss_cvar(
        model, processor, process_vision_info, eval_samples, partition,
    )
    loss_time = time.time() - t0
    print(f"  overall_loss:     {loss_metrics['overall_loss']:.4f}")
    print(f"  worst_10pct_cvar: {loss_metrics['worst_10pct_cvar']:.4f}")
    print(f"  worst_cell_loss:  {loss_metrics['worst_cell_loss']:.4f}")
    print(f"  Time: {loss_time / 60:.1f}m")

    # Generation eval
    print("\nRunning generation eval...")
    t0 = time.time()
    gen_metrics = evaluate_accuracy(
        model, processor, process_vision_info, eval_samples, partition,
    )
    gen_time = time.time() - t0
    print(f"  overall_accuracy:        {gen_metrics['overall_accuracy']:.4f}")
    print(f"  worst_10pct_cvar_error:  {gen_metrics['worst_10pct_cvar_error']:.4f}")
    print(f"  worst_cell_error:        {gen_metrics['worst_cell_error']:.4f}")
    print(f"  Time: {gen_time / 60:.1f}m")

    # Save
    output = {
        "model_id": args.model_id,
        "mode": "zero_shot",
        "split": args.split,
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(eval_samples),
        "loss_metrics": loss_metrics,
        "gen_eval": gen_metrics,
        "time_loss_min": loss_time / 60,
        "time_gen_min": gen_time / 60,
    }

    out_path = out_dir / "zeroshot_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
