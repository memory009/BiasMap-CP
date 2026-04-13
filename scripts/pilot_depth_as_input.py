#!/usr/bin/env python3
"""
PROXY Pilot: Depth-as-Input Fine-Tuning (NOT the true Depth Auxiliary)

This is a PROXY pilot, not the formal Pilot 2. It tests:
  "Does providing a depth map as a SECOND IMAGE INPUT help the model
   answer spatial questions about worst cells?"

This is NOT the same as true depth pseudo-label auxiliary training:
  - Here: depth map is fed as extra visual input (dual-image)
  - True auxiliary: model learns to infer depth FROM the original image
    via auxiliary depth-ordering loss (single-image, extra supervision)

Design:
  - Phase 0: Pre-compute depth maps using Depth Anything V2
  - Phase 1-3: Train specialist LoRA with dual-image input
    [original scene, depth map] + depth-aware prompt
  - Compare against Specialist LoRA (no depth): worst_cell = 0.5409

Go/No-Go:
  POSITIVE:      worst_cell < 0.48
  WEAK POSITIVE: worst_cell 0.48-0.51
  NEGATIVE:      worst_cell >= 0.51

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/pilot_depth_as_input.py --epochs 3 --seed 1
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
from collections import defaultdict, Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.diagnosis.mondrian_partition import MondrianPartition

# ── Config ────────────────────────────────────────────────────────────
MODEL_ID     = "Qwen/Qwen3-VL-2B-Instruct"
DEPTH_MODEL  = "depth-anything/Depth-Anything-V2-Small-hf"
CACHE_DIR    = "/LOCAL2/psqhe8/hf_cache"
SPLITS_DIR   = Path("data/splits")
B1_DIR       = Path("results/sprint2/b1_diagnosis")
OUT_DIR      = Path("results/sprint2/pilots")
DEPTH_CACHE  = OUT_DIR / "depth_maps"

WORST_CELLS = [
    "in_front_of|True|gqa",
    "inside|False|gqa",
    "under|False|gqa",
    "behind|True|gqa",
]

# LoRA config (same as specialist)
LORA_R       = 4
LORA_ALPHA   = 8
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

# Training config
LR             = 2e-4
WEIGHT_DECAY   = 0.01
MAX_GRAD_NORM  = 1.0
MICRO_BS       = 1
GRAD_ACCUM     = 8
WARMUP_RATIO   = 0.03
MIN_CELL_SUPPORT = 20

# ── Prompt templates (depth-augmented) ───────────────────────────────
DEPTH_PREFIX = (
    "You are given two images. The first image shows the original scene. "
    "The second image is a depth map where brighter regions are farther "
    "from the camera and darker regions are closer. Use the depth information "
    "to help answer the spatial reasoning question.\n\n"
)

BINARY_PROMPT = 'Is the following spatial statement true or false?\n\nStatement: "{caption}"\n\nAnswer with ONLY "true" or "false".'
OPEN_PROMPT   = 'Answer the following spatial reasoning question with a short answer.\n\nQuestion: {question}\n\nAnswer:'
SPATIAL_PROMPT = 'Answer the following spatial reasoning question.\n\nQuestion: {question}\n\nChoose the correct answer from: {choices}\n\nAnswer with ONLY the letter or the exact answer text, nothing else.'


def build_prompt(sample):
    choices = sample.get("choices")
    answer = sample["answer"].lower().strip()
    if choices and len(choices) == 2 and set(c.lower() for c in choices) == {"true", "false"}:
        q = sample["question"]
        stmt = q.split('"')[1] if '"' in q else q
        base = BINARY_PROMPT.format(caption=stmt)
    elif choices and len(choices) >= 2:
        base = SPATIAL_PROMPT.format(question=sample["question"],
                                     choices=" / ".join(choices))
    elif answer in ("yes", "no"):
        base = SPATIAL_PROMPT.format(question=sample["question"],
                                     choices="yes / no")
    else:
        base = OPEN_PROMPT.format(question=sample["question"])
    return DEPTH_PREFIX + base


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


# ── Phase 0: Depth map generation ─────────────────────────────────────

def generate_depth_maps(samples, device="cuda"):
    """Pre-compute depth maps for all unique images in samples."""
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor

    DEPTH_CACHE.mkdir(parents=True, exist_ok=True)

    # Collect unique image paths
    unique_images = {}
    for s in samples:
        img_path = s["image_path"]
        img_id = Path(img_path).stem
        if img_id not in unique_images:
            unique_images[img_id] = img_path

    # Check cache
    to_process = {}
    for img_id, img_path in unique_images.items():
        cache_path = DEPTH_CACHE / f"{img_id}.png"
        if not cache_path.exists():
            to_process[img_id] = img_path

    if not to_process:
        print(f"  All {len(unique_images)} depth maps already cached")
        return

    print(f"  Need to generate {len(to_process)}/{len(unique_images)} depth maps")

    # Load depth model
    print(f"  Loading {DEPTH_MODEL}...")
    depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL, cache_dir=CACHE_DIR)
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        DEPTH_MODEL, cache_dir=CACHE_DIR
    ).to(device)
    depth_model.eval()

    # Generate depth maps
    with torch.no_grad():
        for i, (img_id, img_path) in enumerate(to_process.items()):
            image = load_image(img_path)
            inputs = depth_processor(images=image, return_tensors="pt").to(device)
            outputs = depth_model(**inputs)
            depth = outputs.predicted_depth

            # Interpolate to original size
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=image.size[::-1],  # (H, W)
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            # Normalize to 0-255 (brighter = farther)
            depth_np = depth.cpu().numpy()
            depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
            depth_uint8 = (depth_norm * 255).astype(np.uint8)

            # Save as RGB image (grayscale → 3-channel)
            depth_img = Image.fromarray(depth_uint8, mode="L").convert("RGB")
            depth_img.save(DEPTH_CACHE / f"{img_id}.png")

            if (i + 1) % 200 == 0:
                print(f"    depth: {i+1}/{len(to_process)}")

    print(f"  Depth map generation complete")

    # Free depth model
    del depth_model, depth_processor
    torch.cuda.empty_cache()


def get_depth_map(sample):
    """Load cached depth map for a sample."""
    img_id = Path(sample["image_path"]).stem
    depth_path = DEPTH_CACHE / f"{img_id}.png"
    if depth_path.exists():
        return Image.open(depth_path).convert("RGB")
    # Fallback: gray placeholder
    return Image.new("RGB", (224, 224), (128, 128, 128))


# ── Training / Eval ───────────────────────────────────────────────────

def tokenize_train_example(processor, sample, process_vision_info):
    prompt = build_prompt(sample)
    answer = build_answer(sample)
    image = load_image(sample["image_path"])
    depth_map = get_depth_map(sample)

    messages_full = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "image", "image": depth_map},
            {"type": "text", "text": prompt},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": answer},
        ]},
    ]
    messages_prompt = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "image", "image": depth_map},
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

    run_dir = OUT_DIR / "pilot_depth_as_input"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PROXY Pilot: Depth-as-Input Fine-Tuning")
    print(f"Target cells: {WORST_CELLS}")
    print(f"Depth model: {DEPTH_MODEL}")
    print(f"LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print("=" * 60)

    # ── Phase 0: Depth maps ──────────────────────────────────────────
    print("\nPhase 0: Loading worst-cell data for depth map generation...")
    train_samples = load_worst_cell_samples("train")
    eval_samples = load_worst_cell_samples("repair_val")
    all_samples = train_samples + eval_samples
    print(f"  Train: {len(train_samples)}, Eval: {len(eval_samples)}")

    print("\nGenerating depth maps...")
    generate_depth_maps(all_samples, device="cuda")

    # ── Load VLM ─────────────────────────────────────────────────────
    from transformers import AutoProcessor, BitsAndBytesConfig, get_cosine_schedule_with_warmup
    from transformers import Qwen3VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, TaskType
    from qwen_vl_utils import process_vision_info

    print("\nLoading Qwen3-VL model...")
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

    # ── Data stats ────────────────────────────────────────────────────
    print(f"\n  Train: {len(train_samples)} (worst cells only)")
    print(f"  Eval: {len(eval_samples)} (worst cells only)")
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

    # ── Pre-training eval ─────────────────────────────────────────────
    print("\nPre-training eval on worst cells (with depth maps)...")
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
            ckpt_path = run_dir / "checkpoint-best"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_path)
            processor.save_pretrained(ckpt_path)
            print(f"  ★ New best: W10%CVaR={w10:.4f}")

    elapsed = time.time() - train_start

    # ── Comparison ────────────────────────────────────────────────────
    SPECIALIST_W10_CVAR = 0.5409
    SPECIALIST_WORST_CELL = 0.5409
    GLOBAL_W10_CVAR = 0.4077
    GLOBAL_WORST_CELL = 0.5196

    best_eval = epoch_history[best_epoch - 1]["eval"]
    depth_w10 = best_eval["worst_10pct_cvar"]
    depth_wc = best_eval["worst_cell_loss"]

    specialist_cell_losses = {
        "in_front_of|True|gqa": 0.5409,
        "inside|False|gqa": 0.4543,
        "under|False|gqa": 0.5094,
        "behind|True|gqa": 0.3166,
    }
    global_cell_losses = {
        "in_front_of|True|gqa": 0.5196,
        "inside|False|gqa": 0.4165,
        "under|False|gqa": 0.4144,
        "behind|True|gqa": 0.2802,
    }

    print(f"\n{'=' * 60}")
    print("COMPARISON: Depth-Augmented vs Specialist vs Global FT")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25s} {'Depth+Spec':>12s} {'Specialist':>12s} {'Global FT':>12s}")
    print(f"  {'-' * 65}")
    print(f"  {'W10% CVaR':<25s} {depth_w10:>12.4f} {SPECIALIST_W10_CVAR:>12.4f} {GLOBAL_W10_CVAR:>12.4f}")
    print(f"  {'Worst cell loss':<25s} {depth_wc:>12.4f} {SPECIALIST_WORST_CELL:>12.4f} {GLOBAL_WORST_CELL:>12.4f}")

    print(f"\n  Per-cell comparison:")
    print(f"  {'Cell':<30s} {'Depth+Spec':>12s} {'Specialist':>12s} {'Global FT':>12s}")
    for cid in WORST_CELLS:
        dp = best_eval["cell_losses"].get(cid, float("nan"))
        sp = specialist_cell_losses.get(cid, float("nan"))
        gl = global_cell_losses.get(cid, float("nan"))
        print(f"  {cid:<30s} {dp:>12.4f} {sp:>12.4f} {gl:>12.4f}")

    # Go/No-Go
    print(f"\n{'=' * 60}")
    print("GO/NO-GO ASSESSMENT")
    print(f"{'=' * 60}")

    if depth_wc < 0.48:
        signal = "POSITIVE"
        verdict = "Depth augmentation significantly reduces worst-cell loss → depth IS the missing information"
    elif depth_wc < 0.51:
        signal = "WEAK POSITIVE"
        verdict = "Some improvement from depth → partial signal, needs further investigation"
    else:
        signal = "NEGATIVE"
        verdict = "Depth augmentation can't improve worst cells → problem may not be solvable with depth alone"

    # Also check improvement over specialist (isolates depth effect)
    depth_vs_specialist = depth_wc - SPECIALIST_WORST_CELL
    print(f"  Signal: {signal}")
    print(f"  Verdict: {verdict}")
    print(f"  Depth effect (vs specialist): {depth_vs_specialist:+.4f}")
    print(f"  Training time: {elapsed / 3600:.1f}h")

    # ── Save results ──────────────────────────────────────────────────
    output = {
        "pilot": "depth_as_input_proxy",
        "hypothesis": "depth_as_extra_visual_modality",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "depth_model": DEPTH_MODEL,
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
            "specialist_w10_cvar": SPECIALIST_W10_CVAR,
            "specialist_worst_cell": SPECIALIST_WORST_CELL,
            "global_ft_w10_cvar": GLOBAL_W10_CVAR,
            "global_ft_worst_cell": GLOBAL_WORST_CELL,
            "depth_w10_cvar": depth_w10,
            "depth_worst_cell": depth_wc,
            "delta_vs_specialist": depth_vs_specialist,
            "delta_vs_global": depth_wc - GLOBAL_WORST_CELL,
        },
        "signal": signal,
        "verdict": verdict,
    }

    out_path = run_dir / "pilot_depth_as_input.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
