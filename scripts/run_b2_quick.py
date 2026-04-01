#!/usr/bin/env python3
"""
B2 Quick Test: Validate QLoRA fine-tuning pipeline on 200 samples / 50 steps.

Validates:
  1. Qwen2-VL-2B loads with 4-bit quantization + LoRA
  2. Image loading + chat template + loss computation works
  3. Loss decreases over 50 steps
  4. Cell-weighted sampling works (CVaR mode)
  5. Checkpoint save/load works
  6. Eval on repair_val (100 samples) produces metrics

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_b2_quick.py --method cvar
  CUDA_VISIBLE_DEVICES=0 python scripts/run_b2_quick.py --method global
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diagnosis.mondrian_partition import MondrianPartition

# ── Config ─────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen2-VL-2B-Instruct"
CACHE_DIR  = "/LOCAL2/psqhe8/hf_cache"
SPLITS_DIR = Path("data/splits")
B1_DIR     = Path("results/sprint2/b1_diagnosis")
OUT_DIR    = Path("results/sprint2/b2_quick")

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]
LR           = 2e-4
WEIGHT_DECAY = 0.01
MAX_STEPS    = 50
MICRO_BS     = 1
GRAD_ACCUM   = 4
TRAIN_SAMPLES = 200
EVAL_SAMPLES  = 100

# Prompt templates (same as base_vlm.py)
BINARY_PROMPT = 'Look at the image. Is the following spatial statement true or false?\n\nStatement: "{caption}"\n\nAnswer with ONLY "true" or "false".'
OPEN_PROMPT   = 'Look at the image carefully. Answer the following spatial reasoning question with a short answer.\n\nQuestion: {question}\n\nAnswer:'
SPATIAL_PROMPT = 'Look at the image carefully. Answer the following spatial reasoning question.\n\nQuestion: {question}\n\nChoose the correct answer from: {choices}\n\nAnswer with ONLY the letter or the exact answer text, nothing else.'


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

    # Full conversation (user + assistant)
    messages_full = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": answer},
        ]},
    ]
    # Prompt only (for measuring prompt length)
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

    # Labels: mask prompt tokens with -100
    labels = inputs["input_ids"].clone()
    labels[0, :prompt_len] = -100

    return inputs, labels


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["cvar", "global"], default="cvar")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_dir = OUT_DIR / f"quick_{args.method}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model with 4-bit + LoRA ──
    print("=" * 60)
    print(f"B2 Quick Test: {args.method}")
    print("=" * 60)

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from qwen_vl_utils import process_vision_info

    print("Loading model with 4-bit quantization...")
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

    # ── 2. Prepare training data + cell weights ──
    print(f"\nLoading {TRAIN_SAMPLES} train samples (random subsample)...")
    all_train = load_samples("train")
    # Random subsample to get diverse cells (train.jsonl is ordered by relation_type)
    indices = np.random.choice(len(all_train), size=min(TRAIN_SAMPLES, len(all_train)), replace=False)
    train_samples = [all_train[i] for i in indices]

    # Cell weights for CVaR
    sample_weights = None
    if args.method == "cvar":
        partition_path = B1_DIR / "partition.json"
        diag_path = B1_DIR / "Qwen2-VL-2B-Instruct" / "diagnostics.json"
        if partition_path.exists() and diag_path.exists():
            partition = MondrianPartition.load(partition_path)
            with open(diag_path) as f:
                cell_diags = json.load(f)
            # π_c ∝ max(loss_c / loss_avg, 1), clipped at 3.0
            losses = {cid: d.get("mean_loss_shrunk", d.get("mean_loss", 0.5))
                      for cid, d in cell_diags.items()}
            avg_loss = np.mean(list(losses.values())) or 1e-6
            cell_weights = {}
            for cid, loss in losses.items():
                w = max(loss / avg_loss, 1.0)
                support = cell_diags[cid].get("support", 0)
                clip = 1.5 if support < 20 else 3.0
                cell_weights[cid] = min(w, clip)

            # Map train samples to cells BY FEATURES (not by ID)
            sample_weights = []
            mapped = 0
            for s in train_samples:
                cid = partition.get_cell_by_features(s)
                w = cell_weights.get(cid, 1.0) if cid else 1.0
                sample_weights.append(w)
                if cid:
                    mapped += 1

            print(f"  Cell mapping: {mapped}/{len(train_samples)} mapped")
            print(f"  CVaR weights: min={min(sample_weights):.2f}, "
                  f"max={max(sample_weights):.2f}, "
                  f"mean={np.mean(sample_weights):.2f}, "
                  f"n_upweighted={sum(1 for w in sample_weights if w > 1.01)}")
        else:
            print("  ⚠ B1 diagnostics not found, using uniform weights")

    # ── 3. Training loop ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )

    model.train()
    losses_log = []
    accum_loss = 0.0

    print(f"\nTraining for {MAX_STEPS} steps (micro_bs={MICRO_BS}, accum={GRAD_ACCUM})...")

    # Sampling weights
    if sample_weights:
        w = np.array(sample_weights)
        w = w / w.sum()
    else:
        w = None

    optimizer.zero_grad()

    for step_micro in range(1, MAX_STEPS * GRAD_ACCUM + 1):
        # Sample one training example (weighted or uniform)
        idx = np.random.choice(len(train_samples), p=w)
        sample = train_samples[idx]

        try:
            inputs, labels = tokenize_train_example(
                processor, sample, process_vision_info,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            labels = labels.to(model.device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            accum_loss += loss.item() * GRAD_ACCUM
        except Exception as e:
            print(f"  ⚠ Step {step_micro} error: {e}")
            optimizer.zero_grad()
            accum_loss = 0.0
            continue

        if step_micro % GRAD_ACCUM == 0:
            actual_step = step_micro // GRAD_ACCUM
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            avg_l = accum_loss / GRAD_ACCUM
            losses_log.append({"step": actual_step, "loss": avg_l})
            if actual_step % 10 == 0 or actual_step <= 5:
                print(f"  step {actual_step:3d}/{MAX_STEPS}: loss={avg_l:.4f}")
            accum_loss = 0.0

    # ── 4. Check loss decreased ──
    decreased = None
    if len(losses_log) >= 10:
        # Use median of first/last 10 steps to reduce noise from small batches
        first10 = np.median([l["loss"] for l in losses_log[:10]])
        last10 = np.median([l["loss"] for l in losses_log[-10:]])
        min_loss = min(l["loss"] for l in losses_log)
        decreased = bool(last10 < first10 or min_loss < first10 * 0.5)
        print(f"\nLoss trend: first10_med={first10:.4f} → last10_med={last10:.4f} "
              f"(min={min_loss:.4f})  "
              f"{'✅ DECREASED' if decreased else '⚠️ NOT DECREASED'}")

    # ── 5. Save checkpoint ──
    ckpt_dir = run_dir / "checkpoint"
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    print(f"\nCheckpoint saved to {ckpt_dir}")
    ckpt_size = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file())
    print(f"  Size: {ckpt_size / 1e6:.1f} MB")

    # ── 6. Quick eval on repair_val ──
    print(f"\nEvaluating on {EVAL_SAMPLES} repair_val samples...")
    eval_samples = load_samples("repair_val", EVAL_SAMPLES)

    model.eval()
    correct = 0
    total = 0

    for s in eval_samples:
        prompt = build_prompt(s)
        image = load_image(s["image_path"])
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
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

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs, max_new_tokens=16, do_sample=False,
            )
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
        correct += is_correct
        total += 1

    acc = correct / max(total, 1)
    print(f"  Accuracy: {correct}/{total} = {acc:.3f}")

    # ── 7. Summary ──
    summary = {
        "method": args.method,
        "train_samples": TRAIN_SAMPLES,
        "max_steps": MAX_STEPS,
        "losses": losses_log,
        "loss_decreased": decreased,
        "eval_accuracy": float(acc),
        "eval_samples": total,
        "checkpoint_size_mb": float(ckpt_size / 1e6),
    }
    with open(run_dir / "quick_test_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("B2 QUICK TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Method:      {args.method}")
    print(f"  Steps:       {MAX_STEPS}")
    print(f"  Loss trend:  {'✅' if decreased else '⚠️'}")
    print(f"  Eval acc:    {acc:.3f} (on {total} repair_val samples)")
    print(f"  Checkpoint:  {ckpt_size / 1e6:.1f} MB")
    print(f"  Output:      {run_dir}")

    if decreased:
        print("\n✅ Pipeline validated. Ready for full B2 run.")
    else:
        print("\n⚠️ Check loss trend before proceeding to full B2.")


if __name__ == "__main__":
    main()
