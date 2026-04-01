#!/usr/bin/env python3
"""
B2 Full: Targeted (CVaR) vs Global QLoRA fine-tuning on Qwen2-VL-2B.

B2.1: CVaR targeted repair — cell-weighted sampling (upweight worst cells)
B2.2: Global fine-tuning — uniform sampling baseline

Both use identical QLoRA config (matched-budget protocol).
Evaluates on repair_val every epoch; early stops on worst-10%-cell CVaR(loss).

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_b2_full.py --method cvar --seed 1
  CUDA_VISIBLE_DEVICES=1 python scripts/run_b2_full.py --method global --seed 1

  # Or run both in parallel:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_b2_full.py --method cvar --seed 1 &
  CUDA_VISIBLE_DEVICES=1 python scripts/run_b2_full.py --method global --seed 1 &
  wait
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
from datetime import datetime

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diagnosis.mondrian_partition import MondrianPartition

# ── Config (matched-budget) ──────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen2-VL-2B-Instruct"
CACHE_DIR  = "/LOCAL2/psqhe8/hf_cache"
SPLITS_DIR = Path("data/splits")
B1_DIR     = Path("results/sprint2/b1_diagnosis")
OUT_DIR    = Path("results/sprint2/b2_targeted_vs_global")

# QLoRA config (identical for all methods)
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

# Training config (identical for all methods)
LR             = 2e-4
WEIGHT_DECAY   = 0.01
MAX_GRAD_NORM  = 1.0
MAX_EPOCHS     = 3
MICRO_BS       = 1        # one sample at a time (VLM variable-length images)
GRAD_ACCUM     = 16       # effective batch size = 16
WARMUP_RATIO   = 0.03

# CVaR config
CVAR_ALPHA     = 0.1      # worst 10% cells
MAX_CELL_WEIGHT = 3.0

# Eval config
EVAL_EVERY_EPOCH = True
PATIENCE         = 1       # early stopping patience (with 3 epochs, patience>1 never triggers)

# Prompt templates (same as base_vlm.py and run_b2_quick.py)
BINARY_PROMPT = 'Look at the image. Is the following spatial statement true or false?\n\nStatement: "{caption}"\n\nAnswer with ONLY "true" or "false".'
OPEN_PROMPT   = 'Look at the image carefully. Answer the following spatial reasoning question with a short answer.\n\nQuestion: {question}\n\nAnswer:'
SPATIAL_PROMPT = 'Look at the image carefully. Answer the following spatial reasoning question.\n\nQuestion: {question}\n\nChoose the correct answer from: {choices}\n\nAnswer with ONLY the letter or the exact answer text, nothing else.'


# ── Data helpers ──────────────────────────────────────────────────────
def load_samples(split: str, max_n: int = None):
    samples = []
    with open(SPLITS_DIR / f"{split}.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
            if max_n and len(samples) >= max_n:
                break
    return samples


def build_prompt(sample: dict) -> str:
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


def build_answer(sample: dict) -> str:
    return sample["answer"].lower().strip()


def load_image(path: str) -> Image.Image:
    if path and os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    return Image.new("RGB", (224, 224), (128, 128, 128))


def tokenize_train_example(processor, sample, process_vision_info):
    """Build a training example: tokenize prompt+answer, return
    input_ids and labels with prompt tokens masked (-100).

    Label masking strategy: The processor expands <|image_pad|> into
    hundreds of tokens based on image resolution, but tokenizer.encode()
    treats it as 1 token. We compute the expansion offset from the full
    sequence and apply it to the text-only prompt length.
    Verified equivalent to double-processor approach on diverse samples.
    """
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

    # Compute prompt_len via offset: processor expands image tokens,
    # so actual_len > text_only_len. The expansion is identical for
    # full and prompt-only since both share the same image.
    full_actual_len = inputs["input_ids"].shape[1]
    full_text_len = len(processor.tokenizer.encode(text_full))
    prompt_text_len = len(processor.tokenizer.encode(text_prompt))
    image_expansion = full_actual_len - full_text_len
    prompt_len = prompt_text_len + image_expansion

    labels = inputs["input_ids"].clone()
    labels[0, :prompt_len] = -100

    return inputs, labels


# ── Cell weights ──────────────────────────────────────────────────────
def compute_cell_weights(partition, diagnostics):
    """Compute cell sampling weights: π_c ∝ max(loss_c / loss_avg, 1), clipped."""
    losses = {}
    for cid, d in diagnostics.items():
        losses[cid] = d.get("mean_loss_shrunk", d.get("mean_loss", 0.5))

    avg_loss = np.mean(list(losses.values())) if losses else 1e-6
    if avg_loss == 0:
        avg_loss = 1e-6

    cell_weights = {}
    for cid, loss in losses.items():
        w = max(loss / avg_loss, 1.0)
        support = diagnostics[cid].get("support", 0)
        clip = MAX_CELL_WEIGHT * 0.5 if support < 20 else MAX_CELL_WEIGHT
        cell_weights[cid] = min(w, clip)

    return cell_weights


def assign_sample_weights(samples, partition, cell_weights):
    """Map each sample to its cell and assign weight."""
    weights = []
    mapped = 0
    for s in samples:
        cid = partition.get_cell_by_features(s)
        w = cell_weights.get(cid, 1.0) if cid else 1.0
        weights.append(w)
        if cid:
            mapped += 1
    return weights, mapped


# ── Evaluation ────────────────────────────────────────────────────────
def evaluate_loss_cvar(model, processor, process_vision_info, eval_samples,
                       partition, alpha=0.1):
    """
    Evaluate model on eval samples. Compute per-sample loss and
    aggregate into worst-α-cell CVaR (loss-based, no generation needed).
    """
    model.eval()
    sample_losses = []
    sample_cells = []
    errors = 0

    for i, s in enumerate(eval_samples):
        try:
            inputs, labels = tokenize_train_example(processor, s, process_vision_info)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            labels = labels.to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, labels=labels)
            sample_losses.append(outputs.loss.item())
            cid = partition.get_cell_by_features(s)
            sample_cells.append(cid)
        except Exception as e:
            errors += 1
            sample_losses.append(10.0)  # pessimistic default
            sample_cells.append(None)

        if (i + 1) % 500 == 0:
            print(f"    eval: {i+1}/{len(eval_samples)}")

    # Per-cell mean loss
    cell_losses = {}
    for loss, cid in zip(sample_losses, sample_cells):
        if cid is None:
            continue
        cell_losses.setdefault(cid, []).append(loss)
    cell_mean_loss = {cid: np.mean(ls) for cid, ls in cell_losses.items()}

    # Worst-α-cell CVaR
    if cell_mean_loss:
        sorted_losses = sorted(cell_mean_loss.values(), reverse=True)
        k = max(1, int(np.ceil(len(sorted_losses) * alpha)))
        worst_cvar = float(np.mean(sorted_losses[:k]))
    else:
        worst_cvar = 10.0

    overall_loss = float(np.mean(sample_losses))

    return {
        "overall_loss": overall_loss,
        "worst_10pct_cvar": worst_cvar,
        "n_cells_evaluated": len(cell_mean_loss),
        "worst_cell_loss": float(max(cell_mean_loss.values())) if cell_mean_loss else 10.0,
        "best_cell_loss": float(min(cell_mean_loss.values())) if cell_mean_loss else 10.0,
        "eval_errors": errors,
    }


def evaluate_accuracy(model, processor, process_vision_info, eval_samples,
                      partition, alpha=0.1):
    """
    Full generation-based evaluation: accuracy + worst-cell CVaR(error).
    Slower than loss-based eval — use for final checkpoint only.
    """
    model.eval()
    results = []

    for i, s in enumerate(eval_samples):
        prompt = build_prompt(s)
        image = load_image(s["image_path"])
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]

        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            input_len = inputs["input_ids"].shape[1]
            response = processor.decode(
                gen_ids[0][input_len:], skip_special_tokens=True,
            ).strip().lower()

            answer = s["answer"].lower().strip()
            is_correct = (
                response == answer
                or (answer in response)
                or (response in answer and len(response) >= 2)
            )
        except Exception:
            response = ""
            is_correct = False

        cid = partition.get_cell_by_features(s)
        results.append({"correct": is_correct, "cell_id": cid, "response": response})

        if (i + 1) % 200 == 0:
            running_acc = np.mean([r["correct"] for r in results])
            print(f"    gen eval: {i+1}/{len(eval_samples)}, running acc={running_acc:.3f}")

    # Overall accuracy
    overall_acc = float(np.mean([r["correct"] for r in results]))

    # Per-cell error rate
    cell_errors = {}
    for r in results:
        cid = r["cell_id"]
        if cid is None:
            continue
        cell_errors.setdefault(cid, []).append(1.0 - float(r["correct"]))
    cell_mean_error = {cid: np.mean(errs) for cid, errs in cell_errors.items()}

    # Worst-α-cell CVaR(error)
    if cell_mean_error:
        sorted_errors = sorted(cell_mean_error.values(), reverse=True)
        k = max(1, int(np.ceil(len(sorted_errors) * alpha)))
        worst_cvar = float(np.mean(sorted_errors[:k]))
    else:
        worst_cvar = 1.0

    return {
        "overall_accuracy": overall_acc,
        "worst_10pct_cvar_error": worst_cvar,
        "n_cells_evaluated": len(cell_mean_error),
        "worst_cell_error": float(max(cell_mean_error.values())) if cell_mean_error else 1.0,
        "best_cell_error": float(min(cell_mean_error.values())) if cell_mean_error else 1.0,
        "n_evaluated": len(results),
    }


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["cvar", "global"], required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--eval_samples", type=int, default=0,
                        help="Max eval samples (0=all)")
    parser.add_argument("--skip_final_gen_eval", action="store_true",
                        help="Skip generation-based final eval (faster)")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="biasmap-cp",
                        help="wandb project name")
    args = parser.parse_args()

    torch.manual_seed(args.seed + 42)
    np.random.seed(args.seed + 42)

    run_name = f"b2_{args.method}_seed{args.seed}"
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── wandb init ──
    use_wandb = args.wandb and HAS_WANDB
    if args.wandb and not HAS_WANDB:
        print("⚠ wandb not installed, logging disabled")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "method": args.method,
                "seed": args.seed,
                "model_id": MODEL_ID,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "target_modules": TARGET_MODULES,
                "lr": args.lr,
                "weight_decay": WEIGHT_DECAY,
                "max_grad_norm": MAX_GRAD_NORM,
                "max_epochs": args.max_epochs,
                "micro_bs": MICRO_BS,
                "grad_accum": args.grad_accum,
                "effective_bs": MICRO_BS * args.grad_accum,
                "warmup_ratio": WARMUP_RATIO,
                "patience": args.patience,
                "cvar_alpha": CVAR_ALPHA if args.method == "cvar" else None,
                "max_cell_weight": MAX_CELL_WEIGHT if args.method == "cvar" else None,
            },
            tags=["b2", args.method, f"seed{args.seed}"],
        )

    print("=" * 60)
    print(f"B2 Full Training: {args.method} (seed={args.seed})")
    print(f"Output: {run_dir}")
    if use_wandb:
        print(f"wandb: {wandb.run.url}")
    print("=" * 60)

    # ── 1. Load model ──
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from transformers import BitsAndBytesConfig, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model, TaskType
    from qwen_vl_utils import process_vision_info

    print("\nLoading model with 4-bit quantization + LoRA...")
    t0 = time.time()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # ── 2. Load data ──
    print("\nLoading training data...")
    train_samples = load_samples("train")
    print(f"  Train: {len(train_samples)} samples")

    eval_max = args.eval_samples if args.eval_samples > 0 else None
    repair_val = load_samples("repair_val", eval_max)
    print(f"  Repair_val: {len(repair_val)} samples")

    # ── 3. Load partition + compute weights ──
    partition_path = B1_DIR / "partition.json"
    diag_path = B1_DIR / "Qwen2-VL-2B-Instruct" / "diagnostics.json"
    partition = MondrianPartition.load(partition_path)

    sample_weights = None
    if args.method == "cvar":
        print("\nComputing CVaR cell weights...")
        with open(diag_path) as f:
            diagnostics = json.load(f)
        cell_weights = compute_cell_weights(partition, diagnostics)
        sample_weights, mapped = assign_sample_weights(
            train_samples, partition, cell_weights,
        )
        w_arr = np.array(sample_weights)
        print(f"  Mapped: {mapped}/{len(train_samples)}")
        print(f"  Weights: min={w_arr.min():.3f}, max={w_arr.max():.3f}, "
              f"mean={w_arr.mean():.3f}, n_upweighted={np.sum(w_arr > 1.01)}")
    else:
        print("\nGlobal fine-tuning: uniform sampling")

    # Normalize weights for sampling
    if sample_weights:
        w = np.array(sample_weights)
        w = w / w.sum()
    else:
        w = None

    # ── 4. Training setup ──
    n_train = len(train_samples)
    steps_per_epoch = n_train // args.grad_accum
    total_steps = steps_per_epoch * args.max_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\nTraining config:")
    print(f"  Method:          {args.method}")
    print(f"  Epochs:          {args.max_epochs}")
    print(f"  Micro batch:     {MICRO_BS}")
    print(f"  Grad accum:      {args.grad_accum}")
    print(f"  Effective BS:    {MICRO_BS * args.grad_accum}")
    print(f"  Steps/epoch:     {steps_per_epoch}")
    print(f"  Total steps:     {total_steps}")
    print(f"  Warmup steps:    {warmup_steps}")
    print(f"  LR:              {args.lr}")
    print(f"  Patience:        {args.patience}")

    # ── 5. Training loop ──
    model.train()
    losses_log = []
    best_metric = float("inf")
    best_epoch = 0
    patience_counter = 0
    epoch_history = []
    total_errors = 0

    train_start = time.time()

    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.max_epochs}")
        print(f"{'='*60}")

        # Shuffle indices for this epoch (weighted or uniform)
        if w is not None:
            indices = np.random.choice(n_train, size=n_train, replace=True, p=w)
        else:
            indices = np.random.permutation(n_train)

        optimizer.zero_grad()
        accum_loss = 0.0
        epoch_loss = 0.0
        epoch_batches = 0
        log_interval = max(steps_per_epoch // 20, 1)  # ~20 log lines per epoch

        for micro_step in range(n_train):
            idx = indices[micro_step]
            sample = train_samples[idx]

            try:
                inputs, labels = tokenize_train_example(
                    processor, sample, process_vision_info,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                labels = labels.to(model.device)

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss / args.grad_accum
                loss.backward()
                accum_loss += loss.item() * args.grad_accum
            except Exception as e:
                total_errors += 1
                if total_errors <= 10:
                    print(f"  ⚠ micro_step {micro_step} error: {e}")
                elif total_errors == 11:
                    print("  ⚠ Suppressing further error messages...")
                # Don't zero_grad — preserve gradients from other micro-steps
                # in this accumulation window. Just skip this sample.
                continue

            if (micro_step + 1) % args.grad_accum == 0:
                actual_step = (micro_step + 1) // args.grad_accum
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                avg_l = accum_loss / args.grad_accum
                epoch_loss += avg_l
                epoch_batches += 1

                global_step = (epoch - 1) * steps_per_epoch + actual_step
                cur_lr = scheduler.get_last_lr()[0]

                losses_log.append({
                    "epoch": epoch,
                    "step": actual_step,
                    "global_step": global_step,
                    "loss": avg_l,
                    "lr": cur_lr,
                })

                if use_wandb:
                    wandb.log({
                        "train/loss": avg_l,
                        "train/lr": cur_lr,
                        "train/epoch": epoch,
                    }, step=global_step)

                if actual_step % log_interval == 0 or actual_step <= 3:
                    elapsed = time.time() - epoch_start
                    eta = elapsed / actual_step * (steps_per_epoch - actual_step)
                    print(f"  step {actual_step:5d}/{steps_per_epoch}: "
                          f"loss={avg_l:.4f}  lr={cur_lr:.2e}  "
                          f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")
                accum_loss = 0.0

        avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
        epoch_elapsed = time.time() - epoch_start
        print(f"\nEpoch {epoch} done: avg_loss={avg_epoch_loss:.4f}, "
              f"time={epoch_elapsed/60:.1f}m, errors={total_errors}")

        # ── Epoch eval ──
        if EVAL_EVERY_EPOCH:
            print(f"\nEvaluating on repair_val ({len(repair_val)} samples)...")
            eval_start = time.time()
            metrics = evaluate_loss_cvar(
                model, processor, process_vision_info, repair_val,
                partition, CVAR_ALPHA,
            )
            eval_elapsed = time.time() - eval_start
            model.train()

            current_metric = metrics["worst_10pct_cvar"]
            print(f"  Eval: overall_loss={metrics['overall_loss']:.4f}, "
                  f"worst_10pct_cvar={current_metric:.4f}, "
                  f"n_cells={metrics['n_cells_evaluated']}, "
                  f"time={eval_elapsed/60:.1f}m")

            epoch_history.append({
                "epoch": epoch,
                "train_loss": avg_epoch_loss,
                "eval_metrics": metrics,
                "time_train_min": epoch_elapsed / 60,
                "time_eval_min": eval_elapsed / 60,
                "timestamp": datetime.now().isoformat(),
            })

            if use_wandb:
                wandb.log({
                    "eval/overall_loss": metrics["overall_loss"],
                    "eval/worst_10pct_cvar": current_metric,
                    "eval/worst_cell_loss": metrics["worst_cell_loss"],
                    "eval/best_cell_loss": metrics["best_cell_loss"],
                    "eval/n_cells": metrics["n_cells_evaluated"],
                    "epoch/train_loss": avg_epoch_loss,
                    "epoch/time_train_min": epoch_elapsed / 60,
                    "epoch/time_eval_min": eval_elapsed / 60,
                }, step=epoch * steps_per_epoch)

            # Early stopping
            if current_metric < best_metric:
                best_metric = current_metric
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(model, processor, run_dir / "checkpoint-best")
                print(f"  ★ New best: worst_10pct_cvar={best_metric:.4f}")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{args.patience}")
                if patience_counter >= args.patience:
                    print(f"\n  Early stopping at epoch {epoch}")
                    break
        else:
            epoch_history.append({
                "epoch": epoch,
                "train_loss": avg_epoch_loss,
                "time_train_min": epoch_elapsed / 60,
                "timestamp": datetime.now().isoformat(),
            })

        # Save latest checkpoint
        save_checkpoint(model, processor, run_dir / "checkpoint-latest")

        # Save intermediate training log
        with open(run_dir / "training_log.json", "w") as f:
            json.dump({
                "method": args.method,
                "seed": args.seed,
                "epoch_history": epoch_history,
                "losses_log_last100": losses_log[-100:],  # keep log small
                "config": {
                    "lr": args.lr,
                    "grad_accum": args.grad_accum,
                    "max_epochs": args.max_epochs,
                    "lora_r": LORA_R,
                    "lora_alpha": LORA_ALPHA,
                },
            }, f, indent=2)

    total_time = time.time() - train_start

    # ── 6. Final generation-based eval (optional) ──
    gen_eval_results = None
    if not args.skip_final_gen_eval:
        # Load best checkpoint for final eval
        print(f"\nFinal generation-based eval (best checkpoint, epoch {best_epoch})...")
        # Already have the model loaded; if early stopped, load best
        best_ckpt = run_dir / "checkpoint-best"
        if best_ckpt.exists():
            from peft import PeftModel
            # Reload base + best LoRA
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID, cache_dir=CACHE_DIR,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, str(best_ckpt))
            print("  Loaded best checkpoint for final eval")

        gen_eval_results = evaluate_accuracy(
            model, processor, process_vision_info, repair_val,
            partition, CVAR_ALPHA,
        )
        print(f"  Accuracy: {gen_eval_results['overall_accuracy']:.3f}")
        print(f"  Worst-10%-cell CVaR(error): {gen_eval_results['worst_10pct_cvar_error']:.3f}")

        if use_wandb:
            wandb.log({
                "final/accuracy": gen_eval_results["overall_accuracy"],
                "final/worst_10pct_cvar_error": gen_eval_results["worst_10pct_cvar_error"],
                "final/worst_cell_error": gen_eval_results["worst_cell_error"],
                "final/best_cell_error": gen_eval_results["best_cell_error"],
            })

    # ── 7. Save final summary ──
    summary = {
        "method": args.method,
        "seed": args.seed,
        "train_samples": n_train,
        "eval_samples": len(repair_val),
        "epochs_completed": len(epoch_history),
        "best_epoch": best_epoch,
        "best_worst_10pct_cvar": float(best_metric),
        "final_train_loss": float(epoch_history[-1]["train_loss"]),
        "total_time_hours": total_time / 3600,
        "total_errors": total_errors,
        "epoch_history": epoch_history,
        "gen_eval": gen_eval_results,
        "config": {
            "model_id": MODEL_ID,
            "lr": args.lr,
            "grad_accum": args.grad_accum,
            "max_epochs": args.max_epochs,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "cvar_alpha": CVAR_ALPHA if args.method == "cvar" else None,
            "max_cell_weight": MAX_CELL_WEIGHT if args.method == "cvar" else None,
        },
    }

    # Save full losses log separately (can be large)
    with open(run_dir / "losses_log.json", "w") as f:
        json.dump(losses_log, f)

    with open(run_dir / "b2_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"B2 FULL TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Method:         {args.method}")
    print(f"  Seed:           {args.seed}")
    print(f"  Epochs:         {len(epoch_history)}/{args.max_epochs}")
    print(f"  Best epoch:     {best_epoch}")
    print(f"  Best CVaR:      {best_metric:.4f}")
    print(f"  Final loss:     {epoch_history[-1]['train_loss']:.4f}")
    print(f"  Total time:     {total_time/3600:.1f} hours")
    print(f"  Errors:         {total_errors}")
    if gen_eval_results:
        print(f"  Final accuracy: {gen_eval_results['overall_accuracy']:.3f}")
        print(f"  Final CVaR(e):  {gen_eval_results['worst_10pct_cvar_error']:.3f}")
    print(f"  Output:         {run_dir}")
    print(f"  Best ckpt:      {run_dir}/checkpoint-best")

    if use_wandb:
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["best_worst_10pct_cvar"] = float(best_metric)
        wandb.summary["total_time_hours"] = total_time / 3600
        if gen_eval_results:
            wandb.summary["final_accuracy"] = gen_eval_results["overall_accuracy"]
            wandb.summary["final_cvar_error"] = gen_eval_results["worst_10pct_cvar_error"]
        wandb.finish()


def save_checkpoint(model, processor, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    ckpt_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    print(f"  Checkpoint saved: {path} ({ckpt_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
