#!/usr/bin/env python3
"""
Phase 2 Pilot: Test-Time Repair Strategies for Worst CP Cells

Tests 3 inference-time strategies on worst Mondrian cells using
the Qwen3-VL-2B global FT checkpoint. NO training required.

Strategies:
  1. standard    — Greedy decoding (baseline, same as B2v3 eval)
  2. cot_depth   — Chain-of-thought prompt with depth reasoning instruction
  3. sc5         — Self-consistency: 5 samples at temp=0.7, majority vote

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/pilot_inference_repair.py
  CUDA_VISIBLE_DEVICES=1 python scripts/pilot_inference_repair.py --max_samples 50  # quick test
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.diagnosis.mondrian_partition import MondrianPartition

# ── Config ────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen3-VL-2B-Instruct"
CACHE_DIR  = "/LOCAL2/psqhe8/hf_cache"
CHECKPOINT = Path("results/sprint2/b2v3/b2v3_global_seed1_qwen3vl2b/checkpoint-best")
SPLITS_DIR = Path("data/splits")
B1_DIR     = Path("results/sprint2/b1_diagnosis")
OUT_DIR    = Path("results/sprint2/pilots")

WORST_CELLS = [
    "in_front_of|True|gqa",
    "inside|False|gqa",
    "under|False|gqa",
    "behind|True|gqa",
]

# Self-consistency config
SC_N_SAMPLES = 5
SC_TEMPERATURE = 0.7

# ── Prompt templates ──────────────────────────────────────────────────
# Standard (same as B2v3)
OPEN_PROMPT = (
    "Look at the image carefully. Answer the following spatial reasoning "
    "question with a short answer.\n\nQuestion: {question}\n\nAnswer:"
)

# CoT with depth reasoning instruction
COT_DEPTH_PROMPT = (
    "Look at the image carefully. Answer the following spatial reasoning question.\n\n"
    "Before answering, think step by step:\n"
    "1. Identify the objects mentioned in the question\n"
    "2. Determine which object appears closer to or farther from the camera\n"
    "3. Consider occlusion (which object blocks the other)\n"
    "4. Based on these spatial cues, determine the answer\n\n"
    "Question: {question}\n\n"
    "Think step by step, then give your final answer on the last line as just the answer word/phrase."
)


def load_image(path: str) -> Image.Image:
    if path and os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    return Image.new("RGB", (224, 224), (128, 128, 128))


def load_worst_cell_samples(split="repair_val", max_n=None):
    """Load samples only from worst cells."""
    samples = []
    with open(SPLITS_DIR / f"{split}.jsonl") as f:
        for line in f:
            s = json.loads(line)
            da = str(s.get("depth_ambiguity", False))
            cid = f"{s['relation_type']}|{da}|{s['dataset']}"
            if cid in WORST_CELLS:
                s["_cell_id"] = cid
                samples.append(s)
                if max_n and len(samples) >= max_n:
                    break
    return samples


def check_correct(response: str, answer: str) -> bool:
    """Same matching logic as B2v3."""
    response = response.strip().lower()
    answer = answer.strip().lower()
    return (
        response == answer
        or answer in response
        or (response in answer and len(response) >= 2)
    )


def extract_final_answer(text: str) -> str:
    """Extract the final answer from CoT output."""
    text = text.strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return text
    # Take last non-empty line as the answer
    last = lines[-1].lower()
    # Remove common prefixes
    for prefix in ["answer:", "final answer:", "the answer is", "so the answer is", "therefore,"]:
        if last.startswith(prefix):
            last = last[len(prefix):].strip()
    # Remove trailing period
    last = last.rstrip(".")
    return last


def run_inference(model, processor, process_vision_info, sample, prompt_text,
                  do_sample=False, temperature=1.0, max_new_tokens=16):
    """Run single inference and return response text."""
    image = load_image(sample["image_path"])
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt_text},
    ]}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
        gen_ids = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(gen_ids[0][input_len:], skip_special_tokens=True)
    return response.strip()


def strategy_standard(model, processor, pvi, sample):
    """Strategy 1: Standard greedy decoding."""
    prompt = OPEN_PROMPT.format(question=sample["question"])
    response = run_inference(model, processor, pvi, sample, prompt,
                             do_sample=False, max_new_tokens=16)
    return response.lower()


def strategy_cot_depth(model, processor, pvi, sample):
    """Strategy 2: CoT with depth reasoning instruction."""
    prompt = COT_DEPTH_PROMPT.format(question=sample["question"])
    response = run_inference(model, processor, pvi, sample, prompt,
                             do_sample=False, max_new_tokens=128)
    return extract_final_answer(response)


def strategy_sc5(model, processor, pvi, sample):
    """Strategy 3: Self-consistency with 5 samples, majority vote."""
    prompt = OPEN_PROMPT.format(question=sample["question"])
    responses = []
    for _ in range(SC_N_SAMPLES):
        resp = run_inference(model, processor, pvi, sample, prompt,
                             do_sample=True, temperature=SC_TEMPERATURE,
                             max_new_tokens=16)
        responses.append(resp.lower().strip())

    # Majority vote
    counter = Counter(responses)
    majority = counter.most_common(1)[0][0]
    return majority


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per cell (for quick testing)")
    parser.add_argument("--split", default="repair_val",
                        help="Data split to evaluate on")
    parser.add_argument("--strategies", nargs="+",
                        default=["standard", "cot_depth", "sc5"],
                        help="Strategies to run")
    parser.add_argument("--skip_sc", action="store_true",
                        help="Skip self-consistency (slowest)")
    args = parser.parse_args()

    if args.skip_sc and "sc5" in args.strategies:
        args.strategies.remove("sc5")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=" * 60)
    print(f"Phase 2 Pilot: Test-Time Repair Strategies")
    print(f"Split: {args.split}, Max samples: {args.max_samples or 'all'}")
    print(f"Strategies: {args.strategies}")
    print(f"Worst cells: {WORST_CELLS}")
    print(f"=" * 60)

    # ── Load model ────────────────────────────────────────────────────
    print("\nLoading model + LoRA checkpoint...")
    from transformers import (
        AutoProcessor, BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
    )
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, str(CHECKPOINT))
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    print(f"Model loaded on {model.device}")

    # ── Load partition ────────────────────────────────────────────────
    partition = MondrianPartition.load(B1_DIR / "partition.json")

    # ── Load samples ──────────────────────────────────────────────────
    all_samples = load_worst_cell_samples(args.split, max_n=None)
    print(f"Total worst-cell samples: {len(all_samples)}")

    # Optionally limit per cell
    if args.max_samples:
        from itertools import groupby
        limited = []
        cell_counts = Counter()
        for s in all_samples:
            if cell_counts[s["_cell_id"]] < args.max_samples:
                limited.append(s)
                cell_counts[s["_cell_id"]] += 1
        all_samples = limited
        print(f"Limited to {len(all_samples)} samples ({args.max_samples}/cell)")

    cell_dist = Counter(s["_cell_id"] for s in all_samples)
    for c, n in sorted(cell_dist.items()):
        print(f"  {c}: {n}")

    # ── Run strategies ────────────────────────────────────────────────
    strategy_fns = {
        "standard": strategy_standard,
        "cot_depth": strategy_cot_depth,
        "sc5": strategy_sc5,
    }

    all_results = {}
    for strat_name in args.strategies:
        fn = strategy_fns[strat_name]
        print(f"\n{'─' * 60}")
        print(f"Strategy: {strat_name}")
        print(f"{'─' * 60}")

        cell_correct = defaultdict(list)
        t0 = time.time()

        for i, s in enumerate(all_samples):
            try:
                response = fn(model, processor, process_vision_info, s)
                answer = s["answer"].lower().strip()
                correct = check_correct(response, answer)
            except Exception as e:
                print(f"  ERROR on sample {i}: {e}")
                response = ""
                correct = False

            cell_correct[s["_cell_id"]].append(correct)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                running = np.mean([c for cc in cell_correct.values() for c in cc])
                print(f"  [{strat_name}] {i+1}/{len(all_samples)} "
                      f"running_acc={running:.3f} elapsed={elapsed:.0f}s")

        elapsed = time.time() - t0

        # Compute per-cell accuracy
        cell_acc = {}
        for cid in WORST_CELLS:
            if cid in cell_correct:
                acc = float(np.mean(cell_correct[cid]))
                n = len(cell_correct[cid])
                cell_acc[cid] = {"accuracy": acc, "n": n}

        overall = np.mean([c for cc in cell_correct.values() for c in cc])
        worst_cell_acc = min(ca["accuracy"] for ca in cell_acc.values()) if cell_acc else 0

        result = {
            "strategy": strat_name,
            "overall_accuracy": float(overall),
            "worst_cell_accuracy": float(worst_cell_acc),
            "per_cell": cell_acc,
            "n_samples": len(all_samples),
            "elapsed_seconds": elapsed,
        }
        all_results[strat_name] = result

        print(f"\n  Results for {strat_name}:")
        print(f"    Overall accuracy: {overall:.4f}")
        print(f"    Worst cell accuracy: {worst_cell_acc:.4f}")
        for cid, ca in sorted(cell_acc.items()):
            print(f"    {cid}: {ca['accuracy']:.4f} (n={ca['n']})")
        print(f"    Time: {elapsed:.0f}s")

    # ── Compare strategies ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")

    baseline = all_results.get("standard", {})
    baseline_overall = baseline.get("overall_accuracy", 0)
    baseline_worst = baseline.get("worst_cell_accuracy", 0)

    for strat_name, result in all_results.items():
        delta_overall = result["overall_accuracy"] - baseline_overall
        delta_worst = result["worst_cell_accuracy"] - baseline_worst
        signal = "BASELINE" if strat_name == "standard" else (
            "POSITIVE" if delta_worst > 0.03 else
            "WEAK POSITIVE" if delta_worst > 0.01 else
            "NEGATIVE"
        )
        print(f"  {strat_name:15s}  overall={result['overall_accuracy']:.4f} "
              f"({delta_overall:+.4f})  worst_cell={result['worst_cell_accuracy']:.4f} "
              f"({delta_worst:+.4f})  → {signal}")

    # ── Go/No-Go assessment ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("GO/NO-GO ASSESSMENT")
    print(f"{'=' * 60}")

    for strat_name in args.strategies:
        if strat_name == "standard":
            continue
        result = all_results[strat_name]
        delta = result["worst_cell_accuracy"] - baseline_worst
        if delta > 0.03:
            verdict = "GO — worst-cell gain > 3pp"
        elif delta > 0.01:
            verdict = "WEAK — worst-cell gain 1-3pp, needs larger scale"
        elif delta > -0.01:
            verdict = "INCONCLUSIVE — within noise margin"
        else:
            verdict = "NO-GO — worst-cell accuracy decreased"
        print(f"  {strat_name}: delta={delta:+.4f} → {verdict}")

    # ── Save results ──────────────────────────────────────────────────
    output = {
        "pilot": "inference_repair",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "checkpoint": str(CHECKPOINT),
        "split": args.split,
        "worst_cells": WORST_CELLS,
        "strategies": all_results,
        "comparison": {
            "baseline": "standard",
            "baseline_overall": baseline_overall,
            "baseline_worst_cell": baseline_worst,
        },
    }

    out_path = OUT_DIR / "pilot_inference_repair.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
