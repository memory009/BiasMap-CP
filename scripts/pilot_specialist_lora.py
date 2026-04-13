#!/usr/bin/env python3
"""
Phase 2 Pilot: Cell-Specialist LoRA

Tests the CAPACITY/SPECIALIZATION hypothesis:
  "Worst cells fail because shared parameters interfere —
   a dedicated LoRA trained only on worst-cell data can do better."

Design:
  - Fresh Qwen3-VL-2B + NEW LoRA (r=4) trained ONLY on worst-cell data
  - Compared against Global FT (r=16, full 54K data) on worst cells
  - Same eval: loss-based CVaR on repair_val

Go/No-Go (based on Global FT baseline: W10%CVaR=0.4077, worst_cell=0.5196):
  POSITIVE:      worst-cell loss < 0.48 (meaningful drop from 0.52)
  WEAK POSITIVE: worst-cell loss 0.48-0.51 (some signal)
  NEGATIVE:      worst-cell loss >= 0.51 (specialist can't improve either)

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/pilot_specialist_lora.py
  CUDA_VISIBLE_DEVICES=0 python scripts/pilot_specialist_lora.py --epochs 2  # shorter
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
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.diagnosis.mondrian_partition import MondrianPartition

# ── Config ────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen3-VL-2B-Instruct"
CACHE_DIR  = "/LOCAL2/psqhe8/hf_cache"
SPLITS_DIR = Path("data/splits")
B1_DIR     = Path("results/sprint2/b1_diagnosis")
OUT_DIR    = Path("results/sprint2/pilots")

WORST_CELLS = [
    "in_front_of|True|gqa",
    "inside|False|gqa",
    "under|False|gqa",
    "behind|True|gqa",
]

# Specialist LoRA: smaller rank since less data
LORA_R       = 4
LORA_ALPHA   = 8
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

# Training: matched per-sample gradient budget
LR             = 2e-4
WEIGHT_DECAY   = 0.01
MAX_GRAD_NORM  = 1.0
MICRO_BS       = 1
GRAD_ACCUM     = 8       # smaller than global (16) since less data
WARMUP_RATIO   = 0.03
MIN_CELL_SUPPORT = 20

# ── Prompt templates (same as B2v3) ──────────────────────────────────
BINARY_PROMPT = 'Look at the image. Is the following spatial statement true or false?\n\nStatement: "{caption}"\n\nAnswer with ONLY "true" or "false".'
OPEN_PROMPT   = 'Look at the image carefully. Answer the following spatial reasoning question with a short answer.\n\nQuestion: {question}\n\nAnswer:'
SPATIAL_PROMPT = 'Look at the image carefully. Answer the following spatial reasoning question.\n\nQuestion: {question}\n\nChoose the correct answer from: {choices}\n\nAnswer with ONLY the letter or the exact answer text, nothing else.'


def build_prompt(sample):
    choices = sample.get("choices")
    answer = sample["answer"].lower().strip()
    if choices and len(choices) == 2 and set(c.lower() for c in choices) == {"true", "false"}:
        q = sample["question"]
        stmt = q.split('"')[1] if '"' in q else q
        return BINARY_PROMPT.format(caption=stmt)
    elif choices and len(choices) >= 2:
        return SPATIAL_PROMPT.format(question=sample["question"],
                                     choices=" / ".join(choices))
    elif answer in ("yes", "no"):
        return SPATIAL_PROMPT.format(question=sample["question"],
                                     choices="yes / no")
    else:
        return OPEN_PROMPT.format(question=sample["question"])


def build_answer(sample):
    return sample["answer"].lower().strip()


def load_image(path):
    if path and os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    return Image.new("RGB", (224, 224), (128, 128, 128))


def get_cell_id(sample):
    da = str(sample.get("depth_ambiguity", False))
    return f"{sample['relation_type']}|{da}|{sample['dataset']}"


def load_worst_cell_samples(split, max_n=None):
    samples = []
    with open(SPLITS_DIR / f"{split}.jsonl") as f:
        for line in f:
            s = json.loads(line)
            if get_cell_id(s) in WORST_CELLS:
                samples.append(s)
                if max_n and len(samples) >= max_n:
                    break
    return samples


def tokenize_train_example(processor, sample, process_vision_info):
    prompt = build_prompt(sample)
    answer = build_answer(sample)
    image = load_image(sample["image_path"])

    messages_full = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": answer},
        ]},
    ]
    messages_prompt = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
    ]

    text_full = processor.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False,
    )
    text_prompt = processor.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages_full)
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


def evaluate_loss(model, processor, process_vision_info, samples, partition):
    """Evaluate per-cell loss on samples."""
    model.eval()
    cell_losses = defaultdict(list)

    for i, s in enumerate(samples):
        cid = partition.get_cell_by_features(s)
        if cid is None:
            continue
        try:
            inputs, labels = tokenize_train_example(processor, s, process_vision_info)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            labels = labels.to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, labels=labels)
            cell_losses[cid].append(outputs.loss.item())
        except Exception:
            cell_losses[cid].append(10.0)

        if (i + 1) % 100 == 0:
            print(f"    eval: {i+1}/{len(samples)}")

    cell_mean = {}
    for cid, losses in cell_losses.items():
        if len(losses) >= MIN_CELL_SUPPORT:
            cell_mean[cid] = float(np.mean(losses))

    # Compute worst-10% CVaR
    if cell_mean:
        sorted_vals = sorted(cell_mean.values(), reverse=True)
        k = max(1, int(np.ceil(len(sorted_vals) * 0.1)))
        w10_cvar = float(np.mean(sorted_vals[:k]))
    else:
        w10_cvar = 1.0

    worst_cell = max(cell_mean.values()) if cell_mean else 1.0
    overall = np.mean([l for ls in cell_losses.values() for l in ls])

    model.train()
    return {
        "worst_10pct_cvar": w10_cvar,
        "worst_cell_loss": worst_cell,
        "overall_loss": float(overall),
        "n_cells": len(cell_mean),
        "cell_losses": cell_mean,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    torch.manual_seed(args.seed + 42)
    np.random.seed(args.seed + 42)

    run_dir = OUT_DIR / "pilot_specialist_lora"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 2 Pilot: Cell-Specialist LoRA")
    print(f"Target cells: {WORST_CELLS}")
    print(f"LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────
    from transformers import AutoProcessor, BitsAndBytesConfig, get_cosine_schedule_with_warmup
    from transformers import Qwen3VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, TaskType
    from qwen_vl_utils import process_vision_info

    print("\nLoading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # ── Load data ─────────────────────────────────────────────────────
    print("\nLoading worst-cell training data...")
    train_samples = load_worst_cell_samples("train")
    eval_samples = load_worst_cell_samples("repair_val")
    print(f"  Train: {len(train_samples)} (worst cells only)")
    print(f"  Eval: {len(eval_samples)} (worst cells only)")

    from collections import Counter
    train_dist = Counter(get_cell_id(s) for s in train_samples)
    for c, n in sorted(train_dist.items()):
        print(f"    {c}: {n}")

    partition = MondrianPartition.load(B1_DIR / "partition.json")

    # ── Training setup ────────────────────────────────────────────────
    n_train = len(train_samples)
    steps_per_epoch = n_train // GRAD_ACCUM
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    print(f"\n  Steps/epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Grad accum: {GRAD_ACCUM}")

    # ── Pre-training eval (zero-shot specialist baseline) ─────────────
    print("\nPre-training eval on worst cells...")
    pre_eval = evaluate_loss(model, processor, process_vision_info, eval_samples, partition)
    print(f"  Pre-train worst_10pct_cvar: {pre_eval['worst_10pct_cvar']:.4f}")
    print(f"  Pre-train worst_cell_loss: {pre_eval['worst_cell_loss']:.4f}")
    for cid, loss in sorted(pre_eval['cell_losses'].items()):
        print(f"    {cid}: {loss:.4f}")

    # ── Training loop ─────────────────────────────────────────────────
    model.train()
    best_metric = float("inf")
    best_epoch = 0
    epoch_history = []

    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 60}")

        indices = np.random.permutation(n_train)
        optimizer.zero_grad()
        epoch_loss = 0.0
        n_steps = 0

        for micro_step in range(n_train):
            idx = int(indices[micro_step])
            sample = train_samples[idx]

            try:
                inputs, labels = tokenize_train_example(processor, sample, process_vision_info)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                labels = labels.to(model.device)

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss / GRAD_ACCUM
                loss.backward()
                epoch_loss += outputs.loss.item()

            except Exception as e:
                if micro_step < 5:
                    print(f"  Error at step {micro_step}: {e}")
                continue

            if (micro_step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                n_steps += 1

                if n_steps % max(steps_per_epoch // 10, 1) == 0:
                    avg = epoch_loss / (micro_step + 1)
                    print(f"  Step {n_steps}/{steps_per_epoch}, "
                          f"avg_loss={avg:.4f}")

        avg_epoch_loss = epoch_loss / n_train
        print(f"\n  Epoch {epoch} avg train loss: {avg_epoch_loss:.4f}")

        # Eval
        print(f"  Evaluating...")
        eval_result = evaluate_loss(model, processor, process_vision_info, eval_samples, partition)
        w10 = eval_result["worst_10pct_cvar"]
        wc = eval_result["worst_cell_loss"]
        print(f"  worst_10pct_cvar: {w10:.4f}")
        print(f"  worst_cell_loss: {wc:.4f}")
        for cid, loss in sorted(eval_result['cell_losses'].items()):
            print(f"    {cid}: {loss:.4f}")

        epoch_history.append({
            "epoch": epoch,
            "train_loss": avg_epoch_loss,
            "eval": eval_result,
        })

        if w10 < best_metric:
            best_metric = w10
            best_epoch = epoch
            # Save checkpoint
            ckpt_path = run_dir / "checkpoint-best"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_path)
            processor.save_pretrained(ckpt_path)
            print(f"  ★ New best: W10%CVaR={w10:.4f}")

    elapsed = time.time() - train_start

    # ── Comparison with Global FT ─────────────────────────────────────
    GLOBAL_W10_CVAR = 0.4077
    GLOBAL_WORST_CELL = 0.5196

    best_eval = epoch_history[best_epoch - 1]["eval"]
    specialist_w10 = best_eval["worst_10pct_cvar"]
    specialist_wc = best_eval["worst_cell_loss"]

    print(f"\n{'=' * 60}")
    print("COMPARISON: Specialist vs Global FT")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25s} {'Specialist':>12s} {'Global FT':>12s} {'Delta':>10s}")
    print(f"  {'-' * 60}")
    print(f"  {'W10% CVaR':<25s} {specialist_w10:>12.4f} {GLOBAL_W10_CVAR:>12.4f} {specialist_w10 - GLOBAL_W10_CVAR:>+10.4f}")
    print(f"  {'Worst cell loss':<25s} {specialist_wc:>12.4f} {GLOBAL_WORST_CELL:>12.4f} {specialist_wc - GLOBAL_WORST_CELL:>+10.4f}")

    # Per-cell comparison
    global_cell_losses = {
        "in_front_of|True|gqa": 0.5196,
        "inside|False|gqa": 0.4165,
        "under|False|gqa": 0.4144,
        "behind|True|gqa": 0.2802,
    }
    print(f"\n  Per-cell comparison:")
    print(f"  {'Cell':<30s} {'Specialist':>12s} {'Global FT':>12s} {'Delta':>10s}")
    for cid in WORST_CELLS:
        sp = best_eval["cell_losses"].get(cid, float("nan"))
        gl = global_cell_losses.get(cid, float("nan"))
        print(f"  {cid:<30s} {sp:>12.4f} {gl:>12.4f} {sp - gl:>+10.4f}")

    # Go/No-Go
    print(f"\n{'=' * 60}")
    print("GO/NO-GO ASSESSMENT")
    print(f"{'=' * 60}")

    if specialist_wc < 0.48:
        signal = "POSITIVE"
        verdict = "Specialist significantly reduces worst-cell loss → capacity/interference IS a bottleneck"
    elif specialist_wc < 0.51:
        signal = "WEAK POSITIVE"
        verdict = "Some improvement → partial capacity effect, but info gap remains"
    else:
        signal = "NEGATIVE"
        verdict = "Specialist can't improve worst cells either → problem is information-limited, not capacity-limited"

    print(f"  Signal: {signal}")
    print(f"  Verdict: {verdict}")
    print(f"  Training time: {elapsed / 3600:.1f}h")

    # ── Save results ──────────────────────────────────────────────────
    output = {
        "pilot": "specialist_lora",
        "hypothesis": "capacity/specialization",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "worst_cells": WORST_CELLS,
        "n_train": len(train_samples),
        "n_eval": len(eval_samples),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "elapsed_hours": elapsed / 3600,
        "pre_training_eval": pre_eval,
        "epoch_history": epoch_history,
        "best_eval": best_eval,
        "comparison": {
            "global_ft_w10_cvar": GLOBAL_W10_CVAR,
            "global_ft_worst_cell": GLOBAL_WORST_CELL,
            "specialist_w10_cvar": specialist_w10,
            "specialist_worst_cell": specialist_wc,
            "delta_w10": specialist_w10 - GLOBAL_W10_CVAR,
            "delta_worst_cell": specialist_wc - GLOBAL_WORST_CELL,
        },
        "signal": signal,
        "verdict": verdict,
    }

    out_path = run_dir / "pilot_specialist_lora.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
