#!/usr/bin/env python3
"""
Compute a fixed nonconformity cache from a validated R1 LoRA checkpoint.

Produces per-sample CE loss (teacher-forced on the gold answer) for every
worst-cell sample in one or more splits. The cache is the *fixed* gating
signal for R2/R3/R4 — gate scores must NOT be recomputed from the evolving
R2 model during training.

Gate thresholds (tau_h) are calibrated from the **train split only**. Scores
for repair_val (or any other split) reuse those fixed thresholds and are NOT
recalibrated — they just get NC scores for gate-fire decisions during eval.

Parameter separation:
  ALPHA_GATE = 0.30  (CP routing parameter, used HERE)
  ALPHA_CVAR = 0.10  (CVaR evaluation parameter, kept in metrics only)

Output (one JSON per split):
  refine-logs/nc_cache_qwen3vl8b_r1_seed1_train.json
  refine-logs/nc_cache_qwen3vl8b_r1_seed1_repair_val.json
  Each contains:
    {
      "source": "r1_checkpoint",
      "checkpoint": "<path>",
      "model_id": "Qwen/Qwen3-VL-8B-Instruct",
      "split": "<split>",
      "alpha_gate": 0.30,
      "seed": 1,
      "n_samples": <int>,
      "nc_scores": {sample_id: ce_loss, ...},
      "per_cell_taus": {cell_id: tau_h, ...},     # same across all splits
      "per_cell_counts": {cell_id: int, ...},      # this split's counts
      "calibration_split": {cell_id: [ids], ...},  # only in train file
    }

Usage (AFTER R1 is validated and r1_config.json is frozen):
  CUDA_VISIBLE_DEVICES=2 python scripts/compute_nc_cache.py \\
      --checkpoint results/sprint2/pilots/pilot_depth_object_level_seed1_qwen3vl8b_r1_replication/checkpoint-best \\
      --splits train,repair_val --seed 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.pilot_depth_object_level import (
    MODEL_ID,
    CACHE_DIR,
    WORST_CELLS,
    build_prompt,
    build_answer,
    get_cell_id,
    load_image,
    load_worst_cell_samples,
)

ALPHA_GATE = 0.30
ALPHA_CVAR = 0.10
CAL_FRACTION = 0.5
RANDOM_SEED_OFFSET = 42


def build_messages(processor, sample, process_vision_info):
    """Same chat template as pilot_depth_object_level.py — returns tensors + labels."""
    prompt = build_prompt(sample)
    answer = build_answer(sample)
    image = load_image(sample["image_path"])

    msg_full = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": answer},
        ]},
    ]
    msg_prompt = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
    ]
    text_full = processor.apply_chat_template(
        msg_full, tokenize=False, add_generation_prompt=False,
    )
    text_prompt = processor.apply_chat_template(
        msg_prompt, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(msg_full)
    inputs = processor(
        text=[text_full], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    full_actual_len = inputs["input_ids"].shape[1]
    full_text_len = len(processor.tokenizer.encode(text_full))
    prompt_text_len = len(processor.tokenizer.encode(text_prompt))
    image_expansion = full_actual_len - full_text_len
    prompt_len = prompt_text_len + image_expansion

    labels = inputs["input_ids"].clone()
    labels[0, :prompt_len] = -100
    return inputs, labels


def score_split(model, processor, process_vision_info, split: str, max_samples: int | None):
    """Run teacher-forced CE loss for every worst-cell sample in `split`.
    Returns (nc_scores dict, per_cell_ids dict)."""
    samples = load_worst_cell_samples(split)
    if max_samples:
        samples = samples[:max_samples]
    print(f"\n--- Scoring {split} ({len(samples)} samples) ---")

    nc_scores: dict[str, float] = {}
    per_cell_ids: dict[str, list[str]] = defaultdict(list)
    eval_errors = 0
    t0 = time.time()

    for i, s in enumerate(samples):
        sid = s["id"]
        cell = get_cell_id(s)
        if cell not in WORST_CELLS:
            continue
        try:
            inputs, labels = build_messages(processor, s, process_vision_info)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            labels = labels.to(model.device)
            with torch.no_grad():
                out = model(**inputs, labels=labels)
            ce = float(out.loss.item())
        except Exception as e:
            eval_errors += 1
            ce = 10.0
            if eval_errors <= 5:
                print(f"  [warn] {sid}: {type(e).__name__}: {e}")
        nc_scores[sid] = ce
        per_cell_ids[cell].append(sid)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(samples) - i - 1)
            print(f"  {i+1}/{len(samples)} | {elapsed/60:.1f}m | eta {eta/60:.1f}m | err {eval_errors}")

    elapsed = (time.time() - t0) / 60
    print(f"  done: {len(nc_scores)} scored in {elapsed:.1f}m (errors: {eval_errors})")
    return nc_scores, per_cell_ids, eval_errors, elapsed


def calibrate_taus(nc_scores, per_cell_ids, rng):
    """Compute cell-conditional tau_h on the calibration portion of the train split.
    Returns (per_cell_taus, calibration_split dict)."""
    per_cell_taus: dict[str, float] = {}
    calibration_split: dict[str, list[str]] = {}
    for cell, sids in per_cell_ids.items():
        sids_shuffled = list(sids)
        rng.shuffle(sids_shuffled)
        n_cal = max(20, int(len(sids_shuffled) * CAL_FRACTION))
        cal_ids = sids_shuffled[:n_cal]
        cal_scores = np.array([nc_scores[s] for s in cal_ids], dtype=np.float64)
        tau = float(np.quantile(cal_scores, 1 - ALPHA_GATE))
        per_cell_taus[cell] = tau
        calibration_split[cell] = cal_ids
        fire_rate = float(np.mean([nc_scores[s] >= tau for s in sids]))
        print(f"  {cell:<28} |cell|={len(sids):>5} |cal|={n_cal:>5} tau={tau:.4f} fire_rate={fire_rate:.3f}")
    return per_cell_taus, calibration_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--splits", type=str, default="train,repair_val",
                        help="Comma-separated list of splits to score. "
                             "tau_h is always calibrated from the train split.")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the validated R1 LoRA checkpoint-best dir.")
    parser.add_argument("--output_dir", type=str, default="refine-logs",
                        help="Directory to write per-split NC cache JSONs.")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]
    assert "train" in splits, "train split is required for tau_h calibration."
    assert abs(ALPHA_GATE - ALPHA_CVAR) > 1e-6, \
        "ALPHA_GATE must be numerically distinct from ALPHA_CVAR."

    rng = np.random.default_rng(args.seed + RANDOM_SEED_OFFSET)
    torch.manual_seed(args.seed + RANDOM_SEED_OFFSET)

    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    print("=" * 70)
    print("NC cache — from R1 LoRA checkpoint")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Model      : {args.model_id}")
    print(f"Splits     : {splits}")
    print(f"ALPHA_GATE : {ALPHA_GATE} (separate from ALPHA_CVAR={ALPHA_CVAR})")
    print(f"Cells      : {WORST_CELLS}")
    print("=" * 70)

    # Load model with LoRA adapter (4-bit, same as eval script)
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    print(f"\nLoading {args.model_id} + LoRA from {ckpt_path} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=CACHE_DIR)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, cache_dir=CACHE_DIR,
        quantization_config=bnb_config, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, str(ckpt_path))
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    # Score train split first (needed for tau_h calibration)
    train_scores, train_cell_ids, train_errors, train_min = score_split(
        model, processor, process_vision_info, "train", args.max_samples,
    )

    # Calibrate tau_h from train only
    print("\nCalibrating tau_h on train split:")
    per_cell_taus, calibration_split = calibrate_taus(train_scores, train_cell_ids, rng)

    # Write train cache
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_cache(split_name, nc_scores, per_cell_ids, eval_errors, elapsed_min, include_cal=False):
        tag = f"nc_cache_qwen3vl8b_r1_seed{args.seed}_{split_name}.json"
        out_path = out_dir / tag
        payload = {
            "source": "r1_checkpoint",
            "checkpoint": str(ckpt_path),
            "model_id": args.model_id,
            "split": split_name,
            "alpha_gate": ALPHA_GATE,
            "alpha_cvar_note": (
                f"ALPHA_CVAR={ALPHA_CVAR} is the CVaR evaluation parameter "
                f"(numerically distinct from alpha_gate). Not applied here."
            ),
            "seed": args.seed,
            "cal_fraction": CAL_FRACTION,
            "tau_calibrated_on": "train",
            "n_samples": len(nc_scores),
            "nc_scores": nc_scores,
            "per_cell_taus": per_cell_taus,
            "per_cell_counts": {c: len(ids) for c, ids in per_cell_ids.items()},
            "eval_errors": eval_errors,
            "elapsed_min": elapsed_min,
        }
        if include_cal:
            payload["calibration_split"] = calibration_split
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"  Saved -> {out_path}")

    write_cache("train", train_scores, train_cell_ids, train_errors, train_min, include_cal=True)

    # Score remaining splits (repair_val, etc.) and write caches
    for split in splits:
        if split == "train":
            continue
        scores, cell_ids, errors, elapsed = score_split(
            model, processor, process_vision_info, split, args.max_samples,
        )
        # Show per-cell fire rates using the train-calibrated taus
        print(f"\n  Fire rates on {split} (using train-calibrated tau_h):")
        for cell in WORST_CELLS:
            sids = cell_ids.get(cell, [])
            if sids:
                tau = per_cell_taus.get(cell, float("inf"))
                fire_rate = float(np.mean([scores[s] >= tau for s in sids]))
                print(f"    {cell:<28} |cell|={len(sids):>5} fire_rate={fire_rate:.3f}")
        write_cache(split, scores, cell_ids, errors, elapsed, include_cal=False)

    print("\nDone. All caches written.")


if __name__ == "__main__":
    main()
