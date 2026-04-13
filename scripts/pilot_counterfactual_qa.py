#!/usr/bin/env python3
"""
Phase 2 Pilot 3: CP-Guided Synthetic Counterfactual QA (Minimal Version)

Tests the DATA-LEVEL AUGMENTATION hypothesis:
  "Worst cells lack contrastive training signal. Adding binary
   verification pairs (positive + negative via relation inversion)
   from existing QA provides stronger gradient signal for spatial
   relations."

Augmentation design (template-based, deterministic):
  For each parseable worst-cell GQA open-ended sample:
    Q: "What is in front of the dog?" → A: "cat"
    → Positive: "Is this true or false? 'The cat is in front of the dog.'" → "true"
    → Negative: "Is this true or false? 'The cat is behind the dog.'" → "false"

  Relation inverse pairs:
    in_front_of ↔ behind
    under ↔ on top of
    inside ↔ outside of

  This creates contrastive pairs reinforcing spatial semantics without
  new images. The model already handles binary format well (VSR loss
  5-16x lower than GQA open-ended), so binary augmentation provides
  a stronger learning signal.

Honest caveat:
  This may fail — text augmentation doesn't add new visual information,
  and the architecture-level analysis suggests CLIP's 1D patch flattening
  is the fundamental bottleneck. But this is the minimal test to close
  or confirm the data-level direction.

Go/No-Go (vs Specialist LoRA 0.5409, Global FT 0.5196):
  POSITIVE:      worst_cell < 0.48
  WEAK POSITIVE: worst_cell 0.48-0.51
  NEGATIVE:      worst_cell >= 0.51

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/pilot_counterfactual_qa.py
"""
import argparse
import json
import os
import sys
import time
import random
import re
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter
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

LORA_R       = 4
LORA_ALPHA   = 8
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

LR             = 2e-4
WEIGHT_DECAY   = 0.01
MAX_GRAD_NORM  = 1.0
MICRO_BS       = 1
GRAD_ACCUM     = 8
WARMUP_RATIO   = 0.03
MIN_CELL_SUPPORT = 20

# ── Relation parsing + inversion ──────────────────────────────────────
REL_PHRASES = {
    "in_front_of": ["in front of"],
    "behind": ["behind of", "behind"],
    "inside": ["inside of", "inside"],
    "under": ["underneath", "under"],
}

# Human-readable relation text for statement construction
REL_TEXT = {
    "in_front_of": "in front of",
    "behind": "behind",
    "inside": "inside",
    "under": "under",
}

REL_INVERSE_TEXT = {
    "in_front_of": "behind",
    "behind": "in front of",
    "inside": "outside of",
    "under": "on top of",
}

# ── Prompt templates ──────────────────────────────────────────────────
# Original QA prompts (same as specialist LoRA)
BINARY_PROMPT = 'Look at the image. Is the following spatial statement true or false?\n\nStatement: "{caption}"\n\nAnswer with ONLY "true" or "false".'
OPEN_PROMPT   = 'Look at the image carefully. Answer the following spatial reasoning question with a short answer.\n\nQuestion: {question}\n\nAnswer:'
SPATIAL_PROMPT = 'Look at the image carefully. Answer the following spatial reasoning question.\n\nQuestion: {question}\n\nChoose the correct answer from: {choices}\n\nAnswer with ONLY the letter or the exact answer text, nothing else.'


def build_prompt(sample):
    """Build prompt for original QA samples."""
    if sample.get("is_augmented"):
        return BINARY_PROMPT.format(caption=sample["statement"])

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


def load_worst_cell_samples(split):
    samples = []
    with open(SPLITS_DIR / f"{split}.jsonl") as f:
        for line in f:
            s = json.loads(line)
            if get_cell_id(s) in WORST_CELLS:
                samples.append(s)
    return samples


# ── Counterfactual augmentation ───────────────────────────────────────

def extract_reference_object(question, relation_type):
    """Extract reference object from GQA spatial question."""
    q = question.lower().rstrip("?").strip()
    for phrase in REL_PHRASES[relation_type]:
        idx = q.rfind(phrase)
        if idx != -1:
            after = q[idx + len(phrase):].strip()
            if after and len(after) < 60:
                return after
    return None


def generate_counterfactual_samples(original_samples):
    """
    Generate binary counterfactual pairs from parseable open-ended GQA samples.

    For each parseable sample:
      Original: "What is in front of the dog?" → "cat"
      → Positive: "The cat is in front of the dog." → "true"
      → Negative: "The cat is behind the dog." → "false"
    """
    aug_samples = []
    stats = {"parseable": 0, "skipped_yesno": 0, "skipped_noparse": 0,
             "skipped_answer_too_long": 0, "skipped_ref_too_long": 0}

    for s in original_samples:
        answer = s["answer"].lower().strip()
        rel = s["relation_type"]

        # Skip yes/no questions — already binary
        if answer in ("yes", "no"):
            stats["skipped_yesno"] += 1
            continue

        ref_obj = extract_reference_object(s["question"], rel)
        if ref_obj is None:
            stats["skipped_noparse"] += 1
            continue

        # Quality filter: skip overly complex noun phrases
        if len(answer.split()) > 4:
            stats["skipped_answer_too_long"] += 1
            continue
        if len(ref_obj.split()) > 6:
            stats["skipped_ref_too_long"] += 1
            continue

        stats["parseable"] += 1
        rel_text = REL_TEXT[rel]
        inv_text = REL_INVERSE_TEXT[rel]

        # Positive: "[answer] is [relation] [ref_obj]" → true
        pos_statement = f"The {answer} is {rel_text} {ref_obj}."
        aug_samples.append({
            "image_path": s["image_path"],
            "statement": pos_statement,
            "answer": "true",
            "is_augmented": True,
            "aug_type": "positive_binary",
            "source_relation": rel,
            # Keep cell metadata for tracking
            "relation_type": s["relation_type"],
            "dataset": s["dataset"],
            "depth_ambiguity": s.get("depth_ambiguity", False),
        })

        # Negative: "[answer] is [inverse_relation] [ref_obj]" → false
        neg_statement = f"The {answer} is {inv_text} {ref_obj}."
        aug_samples.append({
            "image_path": s["image_path"],
            "statement": neg_statement,
            "answer": "false",
            "is_augmented": True,
            "aug_type": "negative_binary",
            "source_relation": rel,
            "relation_type": s["relation_type"],
            "dataset": s["dataset"],
            "depth_ambiguity": s.get("depth_ambiguity", False),
        })

    return aug_samples, stats


# ── Tokenization ──────────────────────────────────────────────────────

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
    """Evaluate per-cell loss — ONLY on original QA."""
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
    parser.add_argument("--aug_ratio", type=float, default=0.3,
                        help="Fraction of augmented samples per epoch (default 0.3)")
    args = parser.parse_args()

    torch.manual_seed(args.seed + 42)
    np.random.seed(args.seed + 42)
    random.seed(args.seed + 42)

    run_dir = OUT_DIR / "pilot_counterfactual_qa"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 2 Pilot 3: Counterfactual QA Augmentation")
    print(f"Target cells: {WORST_CELLS}")
    print(f"LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print("=" * 60)

    # ── Load data + generate augmentation ─────────────────────────────
    print("\nLoading worst-cell training data...")
    train_samples = load_worst_cell_samples("train")
    eval_samples = load_worst_cell_samples("repair_val")
    print(f"  Original train: {len(train_samples)}")
    print(f"  Eval: {len(eval_samples)}")

    print("\nGenerating counterfactual QA pairs...")
    aug_samples, aug_stats = generate_counterfactual_samples(train_samples)
    print(f"  Parseable open-ended: {aug_stats['parseable']}")
    print(f"  Skipped (yes/no): {aug_stats['skipped_yesno']}")
    print(f"  Skipped (no parse): {aug_stats['skipped_noparse']}")
    print(f"  Generated: {len(aug_samples)} augmented samples")

    aug_type_dist = Counter(s["aug_type"] for s in aug_samples)
    aug_rel_dist = Counter(s["source_relation"] for s in aug_samples)
    print(f"  By type: {dict(aug_type_dist)}")
    print(f"  By relation: {dict(aug_rel_dist)}")

    # Show examples
    print("\n  Example augmentations:")
    for s in aug_samples[:4]:
        print(f"    [{s['aug_type']}] \"{s['statement']}\" → {s['answer']}")

    # Mark originals
    for s in train_samples:
        s["is_augmented"] = False

    print(f"\n  Original pool: {len(train_samples)}")
    print(f"  Augmented pool: {len(aug_samples)}")
    print(f"  Aug ratio: {args.aug_ratio:.2f}")

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

    partition = MondrianPartition.load(B1_DIR / "partition.json")

    # ── Training setup (matched-compute: same budget as specialist) ──
    n_epoch = len(train_samples)  # 5946, same as specialist
    steps_per_epoch = n_epoch // GRAD_ACCUM
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    print(f"\n  Samples/epoch: {n_epoch} (matched-compute with specialist)")
    print(f"  Steps/epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Aug ratio target: {args.aug_ratio:.2f}")

    # ── Pre-training eval ─────────────────────────────────────────────
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
    epoch_sampling_log = []

    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 60}")

        # ── Explicit stratified sampling ──────────────────────────
        n_aug_target = round(n_epoch * args.aug_ratio)
        n_orig_target = n_epoch - n_aug_target

        # Sample from augmented pool
        if n_aug_target <= len(aug_samples):
            epoch_aug = random.sample(aug_samples, n_aug_target)
            sampling_mode = "without_replacement"
        else:
            epoch_aug = random.choices(aug_samples, k=n_aug_target)
            sampling_mode = "with_replacement"

        # Sample from original pool (always enough)
        epoch_orig = random.sample(train_samples, n_orig_target)

        # Combine and shuffle
        epoch_data = epoch_orig + epoch_aug
        random.shuffle(epoch_data)

        # Log epoch sampling composition
        aug_ratio_actual = len(epoch_aug) / len(epoch_data)
        epoch_stats = {
            "epoch": epoch,
            "n_epoch_samples": len(epoch_data),
            "n_orig_sampled": len(epoch_orig),
            "n_aug_sampled": len(epoch_aug),
            "aug_ratio_actual": aug_ratio_actual,
            "n_orig_pool_total": len(train_samples),
            "n_aug_pool_total": len(aug_samples),
            "sampling_mode": sampling_mode,
        }
        epoch_sampling_log.append(epoch_stats)
        print(f"  Sampling: {len(epoch_orig)} orig + {len(epoch_aug)} aug = {len(epoch_data)} total")
        print(f"  Aug ratio actual: {aug_ratio_actual:.4f}, mode: {sampling_mode}")

        optimizer.zero_grad()
        epoch_loss_orig = 0.0
        epoch_loss_aug = 0.0
        n_orig_seen = 0
        n_aug_seen = 0
        n_steps = 0

        for micro_step in range(len(epoch_data)):
            sample = epoch_data[micro_step]

            try:
                inputs, labels = tokenize_train_example(processor, sample, process_vision_info)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                labels = labels.to(model.device)

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss / GRAD_ACCUM
                loss.backward()

                if sample.get("is_augmented"):
                    epoch_loss_aug += outputs.loss.item()
                    n_aug_seen += 1
                else:
                    epoch_loss_orig += outputs.loss.item()
                    n_orig_seen += 1

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
                    avg_orig = epoch_loss_orig / max(n_orig_seen, 1)
                    avg_aug = epoch_loss_aug / max(n_aug_seen, 1)
                    print(f"  Step {n_steps}/{steps_per_epoch}, "
                          f"orig={avg_orig:.4f}, aug={avg_aug:.4f}")

        avg_orig = epoch_loss_orig / max(n_orig_seen, 1)
        avg_aug = epoch_loss_aug / max(n_aug_seen, 1)
        print(f"\n  Epoch {epoch} — orig_loss: {avg_orig:.4f}, "
              f"aug_loss: {avg_aug:.4f} "
              f"(n_orig={n_orig_seen}, n_aug={n_aug_seen})")

        # Eval (original QA only)
        print(f"  Evaluating (original QA only)...")
        eval_result = evaluate_loss(model, processor, process_vision_info, eval_samples, partition)
        w10 = eval_result["worst_10pct_cvar"]
        wc = eval_result["worst_cell_loss"]
        print(f"  worst_10pct_cvar: {w10:.4f}")
        print(f"  worst_cell_loss: {wc:.4f}")
        for cid, loss in sorted(eval_result['cell_losses'].items()):
            print(f"    {cid}: {loss:.4f}")

        epoch_history.append({
            "epoch": epoch,
            "train_loss_original": avg_orig,
            "train_loss_augmented": avg_aug,
            "eval": eval_result,
        })

        if w10 < best_metric:
            best_metric = w10
            best_epoch = epoch
            ckpt_path = run_dir / "checkpoint-best"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_path)
            processor.save_pretrained(ckpt_path)
            print(f"  ★ New best: W10%CVaR={w10:.4f}")

    elapsed = time.time() - train_start

    # ── Comparison ────────────────────────────────────────────────────
    SPECIALIST_W10 = 0.5409
    SPECIALIST_WC  = 0.5409
    GLOBAL_W10     = 0.4077
    GLOBAL_WC      = 0.5196

    best_eval = epoch_history[best_epoch - 1]["eval"]
    cf_w10 = best_eval["worst_10pct_cvar"]
    cf_wc = best_eval["worst_cell_loss"]

    specialist_cells = {
        "in_front_of|True|gqa": 0.5409, "inside|False|gqa": 0.4543,
        "under|False|gqa": 0.5094, "behind|True|gqa": 0.3166,
    }
    global_cells = {
        "in_front_of|True|gqa": 0.5196, "inside|False|gqa": 0.4165,
        "under|False|gqa": 0.4144, "behind|True|gqa": 0.2802,
    }

    print(f"\n{'=' * 60}")
    print("COMPARISON: Counterfactual QA vs Specialist vs Global FT")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25s} {'CF QA':>12s} {'Specialist':>12s} {'Global FT':>12s}")
    print(f"  {'-' * 65}")
    print(f"  {'W10% CVaR':<25s} {cf_w10:>12.4f} {SPECIALIST_W10:>12.4f} {GLOBAL_W10:>12.4f}")
    print(f"  {'Worst cell loss':<25s} {cf_wc:>12.4f} {SPECIALIST_WC:>12.4f} {GLOBAL_WC:>12.4f}")

    print(f"\n  Per-cell:")
    print(f"  {'Cell':<30s} {'CF QA':>12s} {'Specialist':>12s} {'Global FT':>12s}")
    for cid in WORST_CELLS:
        cf = best_eval["cell_losses"].get(cid, float("nan"))
        sp = specialist_cells.get(cid, float("nan"))
        gl = global_cells.get(cid, float("nan"))
        print(f"  {cid:<30s} {cf:>12.4f} {sp:>12.4f} {gl:>12.4f}")

    # Go/No-Go
    print(f"\n{'=' * 60}")
    print("GO/NO-GO ASSESSMENT")
    print(f"{'=' * 60}")

    if cf_wc < 0.48:
        signal = "POSITIVE"
        verdict = "Contrastive QA augmentation breaks the loss ceiling → data diversity IS a bottleneck"
    elif cf_wc < 0.51:
        signal = "WEAK POSITIVE"
        verdict = "Some improvement from contrastive data → partial signal, needs scaling"
    else:
        signal = "NEGATIVE"
        verdict = "Text-level contrastive augmentation insufficient → data-level direction closed at 2B scale"

    print(f"  Signal: {signal}")
    print(f"  Verdict: {verdict}")
    print(f"  vs Specialist: {cf_wc - SPECIALIST_WC:+.4f}")
    print(f"  vs Global FT:  {cf_wc - GLOBAL_WC:+.4f}")
    print(f"  Training time: {elapsed / 3600:.1f}h")

    # ── Save results ──────────────────────────────────────────────────
    output = {
        "pilot": "counterfactual_qa",
        "hypothesis": "data_level_contrastive_augmentation",
        "augmentation": "binary verification pairs via relation inversion",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "lora_r": LORA_R,
        "worst_cells": WORST_CELLS,
        "n_train_original": len(train_samples),
        "n_train_augmented": len(aug_samples),
        "n_epoch_samples": n_epoch,
        "aug_ratio_setting": args.aug_ratio,
        "n_eval": len(eval_samples),
        "augmentation_stats": aug_stats,
        "aug_type_dist": dict(aug_type_dist),
        "aug_rel_dist": dict(aug_rel_dist),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "elapsed_hours": elapsed / 3600,
        "pre_training_eval": pre_eval,
        "epoch_sampling": epoch_sampling_log,
        "epoch_history": epoch_history,
        "best_eval": best_eval,
        "comparison": {
            "specialist_w10_cvar": SPECIALIST_W10,
            "specialist_worst_cell": SPECIALIST_WC,
            "global_ft_w10_cvar": GLOBAL_W10,
            "global_ft_worst_cell": GLOBAL_WC,
            "cf_w10_cvar": cf_w10,
            "cf_worst_cell": cf_wc,
            "delta_vs_specialist": cf_wc - SPECIALIST_WC,
            "delta_vs_global": cf_wc - GLOBAL_WC,
        },
        "signal": signal,
        "verdict": verdict,
    }

    out_path = run_dir / "pilot_counterfactual_qa.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
