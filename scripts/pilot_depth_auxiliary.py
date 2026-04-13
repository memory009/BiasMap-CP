#!/usr/bin/env python3
"""
Phase 2 Pilot 2 (TRUE): Depth Pseudo-Label Auxiliary Training

Tests the DEPTH SUPERVISION hypothesis:
  "Worst cells fail because the 2B model lacks depth understanding.
   An auxiliary depth-ordering task — using pseudo-labels from Depth
   Anything V2 — can teach the model to infer depth from the ORIGINAL
   image alone, improving spatial reasoning on worst cells."

Key difference from proxy pilot (depth_as_input):
  - Proxy: feeds depth map as second image → model reads depth directly
  - THIS:  single image only, auxiliary QA about depth ordering →
           model must LEARN to extract depth from original image

Auxiliary task design (minimum viable, no bboxes needed):
  For each worst-cell image, pre-compute depth map, then generate
  region-based depth-ordering QA pairs:
    1. Left-half vs right-half: "Which side is closer to the camera?"
    2. Top-half vs bottom-half: "Which part is closer to the camera?"
    3. Quadrant depth ranking: "Which quadrant is closest?"
  These teach the model coarse depth understanding from the original image.

Approximation assumption:
  GQA lacks subject/object bboxes. We use region-level depth ordering
  as a proxy for object-level depth comparison. This is coarser than
  ideal but still teaches the model to correlate visual features with
  relative depth — the skill needed for in_front_of/behind/under.

Go/No-Go (vs Specialist LoRA 0.5409, Global FT 0.5196):
  POSITIVE:      worst_cell < 0.48
  WEAK POSITIVE: worst_cell 0.48-0.51
  NEGATIVE:      worst_cell >= 0.51

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/pilot_depth_auxiliary.py
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
DEPTH_NP_CACHE = OUT_DIR / "depth_maps_npy"  # raw numpy arrays for aux generation

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
AUX_RATIO      = 1.0  # 1 aux sample per original sample

# ── Prompt templates ──────────────────────────────────────────────────
# Original QA prompts (same as specialist LoRA — single image, no depth map)
BINARY_PROMPT = 'Look at the image. Is the following spatial statement true or false?\n\nStatement: "{caption}"\n\nAnswer with ONLY "true" or "false".'
OPEN_PROMPT   = 'Look at the image carefully. Answer the following spatial reasoning question with a short answer.\n\nQuestion: {question}\n\nAnswer:'
SPATIAL_PROMPT = 'Look at the image carefully. Answer the following spatial reasoning question.\n\nQuestion: {question}\n\nChoose the correct answer from: {choices}\n\nAnswer with ONLY the letter or the exact answer text, nothing else.'

# Auxiliary depth prompts (single image, depth-ordering questions)
AUX_LR_PROMPT = 'Look at this image carefully and estimate the depth layout. Which side of the image contains objects that are generally closer to the camera?\n\nAnswer with ONLY "left" or "right".'
AUX_TB_PROMPT = 'Look at this image carefully and estimate the depth layout. Which part of the image contains objects that are generally closer to the camera?\n\nAnswer with ONLY "upper" or "lower".'
AUX_QUAD_PROMPT = 'Look at this image carefully and estimate which region contains the objects closest to the camera.\n\nChoose from: upper-left / upper-right / lower-left / lower-right\n\nAnswer with ONLY one of the four options.'


def build_prompt(sample):
    """Build prompt for original QA samples."""
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


# ── Depth map generation + auxiliary sample creation ──────────────────

def generate_depth_maps(samples, device="cuda"):
    """Pre-compute depth maps (PNG + NPY) for all unique images."""
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor

    DEPTH_CACHE.mkdir(parents=True, exist_ok=True)
    DEPTH_NP_CACHE.mkdir(parents=True, exist_ok=True)

    unique_images = {}
    for s in samples:
        img_path = s["image_path"]
        img_id = Path(img_path).stem
        if img_id not in unique_images:
            unique_images[img_id] = img_path

    to_process = {}
    for img_id, img_path in unique_images.items():
        npy_path = DEPTH_NP_CACHE / f"{img_id}.npy"
        if not npy_path.exists():
            to_process[img_id] = img_path

    if not to_process:
        print(f"  All {len(unique_images)} depth maps already cached")
        return

    print(f"  Need to generate {len(to_process)}/{len(unique_images)} depth maps")
    print(f"  Loading {DEPTH_MODEL}...")
    depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL, cache_dir=CACHE_DIR)
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        DEPTH_MODEL, cache_dir=CACHE_DIR
    ).to(device)
    depth_model.eval()

    with torch.no_grad():
        for i, (img_id, img_path) in enumerate(to_process.items()):
            image = load_image(img_path)
            inputs = depth_processor(images=image, return_tensors="pt").to(device)
            outputs = depth_model(**inputs)
            depth = outputs.predicted_depth

            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth_np = depth.cpu().numpy()

            # Save raw numpy (for auxiliary generation)
            np.save(DEPTH_NP_CACHE / f"{img_id}.npy", depth_np)

            # Save normalized PNG (for proxy pilot compatibility)
            depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
            depth_uint8 = (depth_norm * 255).astype(np.uint8)
            depth_img = Image.fromarray(depth_uint8, mode="L").convert("RGB")
            depth_img.save(DEPTH_CACHE / f"{img_id}.png")

            if (i + 1) % 200 == 0:
                print(f"    depth: {i+1}/{len(to_process)}")

    print(f"  Depth map generation complete")
    del depth_model, depth_processor
    torch.cuda.empty_cache()


def load_depth_np(sample):
    """Load raw depth numpy array for a sample."""
    img_id = Path(sample["image_path"]).stem
    npy_path = DEPTH_NP_CACHE / f"{img_id}.npy"
    if npy_path.exists():
        return np.load(npy_path)
    return None


def generate_auxiliary_samples(original_samples, rng):
    """
    Generate depth-ordering auxiliary QA pairs from pre-computed depth maps.

    For each original sample, create 1 auxiliary sample (randomly chosen type).
    Auxiliary types:
      - left/right depth comparison
      - upper/lower depth comparison
      - quadrant closest to camera

    Depth Anything V2 outputs predicted_depth where HIGHER = FARTHER.
    So LOWER mean depth = closer to camera.
    """
    aux_samples = []
    skipped = 0

    for s in original_samples:
        depth_np = load_depth_np(s)
        if depth_np is None:
            skipped += 1
            continue

        h, w = depth_np.shape

        # Compute regional mean depths (lower = closer)
        left_d  = depth_np[:, :w // 2].mean()
        right_d = depth_np[:, w // 2:].mean()
        top_d   = depth_np[:h // 2, :].mean()
        bot_d   = depth_np[h // 2:, :].mean()

        # Quadrant depths
        quad_depths = {
            "upper-left":  depth_np[:h // 2, :w // 2].mean(),
            "upper-right": depth_np[:h // 2, w // 2:].mean(),
            "lower-left":  depth_np[h // 2:, :w // 2].mean(),
            "lower-right": depth_np[h // 2:, w // 2:].mean(),
        }

        candidates = []

        # Skip if depth is nearly flat (< 5% relative range) — no meaningful signal
        depth_range = depth_np.max() - depth_np.min()
        if depth_range < 0.05 * depth_np.mean():
            skipped += 1
            continue

        # Left-right comparison (only if meaningful difference)
        if abs(left_d - right_d) > 0.02 * depth_range:
            lr_answer = "left" if left_d < right_d else "right"
            candidates.append({
                "image_path": s["image_path"],
                "question": AUX_LR_PROMPT,
                "answer": lr_answer,
                "is_auxiliary": True,
                "aux_type": "depth_lr",
                "dataset": s["dataset"],
                "relation_type": s["relation_type"],
                "depth_ambiguity": s.get("depth_ambiguity", False),
            })

        # Top-bottom comparison
        if abs(top_d - bot_d) > 0.02 * depth_range:
            tb_answer = "upper" if top_d < bot_d else "lower"
            candidates.append({
                "image_path": s["image_path"],
                "question": AUX_TB_PROMPT,
                "answer": tb_answer,
                "is_auxiliary": True,
                "aux_type": "depth_tb",
                "dataset": s["dataset"],
                "relation_type": s["relation_type"],
                "depth_ambiguity": s.get("depth_ambiguity", False),
            })

        # Quadrant ranking
        closest_quad = min(quad_depths, key=quad_depths.get)
        candidates.append({
            "image_path": s["image_path"],
            "question": AUX_QUAD_PROMPT,
            "answer": closest_quad,
            "is_auxiliary": True,
            "aux_type": "depth_quad",
            "dataset": s["dataset"],
            "relation_type": s["relation_type"],
            "depth_ambiguity": s.get("depth_ambiguity", False),
        })

        # Pick 1 random auxiliary from candidates
        if candidates:
            aux_samples.append(candidates[rng.integers(len(candidates))])

    if skipped:
        print(f"  Skipped {skipped} samples (flat depth or missing map)")
    return aux_samples


# ── Tokenization (single image, standard QA) ─────────────────────────

def tokenize_train_example(processor, sample, process_vision_info):
    """Tokenize a training example — works for both original and auxiliary QA."""
    if sample.get("is_auxiliary"):
        prompt = sample["question"]  # aux prompts are pre-formatted
        answer = sample["answer"]
    else:
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
    """Evaluate per-cell loss — ONLY on original QA, NOT auxiliary."""
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
    parser.add_argument("--aux_ratio", type=float, default=AUX_RATIO,
                        help="Auxiliary samples per original sample (default 1.0)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed + 42)
    torch.manual_seed(args.seed + 42)
    np.random.seed(args.seed + 42)

    run_dir = OUT_DIR / "pilot_depth_auxiliary"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 2 Pilot 2 (TRUE): Depth Pseudo-Label Auxiliary")
    print(f"Target cells: {WORST_CELLS}")
    print(f"Depth model: {DEPTH_MODEL}")
    print(f"LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"Aux ratio: {args.aux_ratio}")
    print("=" * 60)
    print("\nNOTE: Single-image input. Auxiliary teaches depth ordering")
    print("      from the original image — no depth map at inference.\n")

    # ── Phase 0: Depth maps + auxiliary generation ────────────────────
    print("Phase 0a: Loading worst-cell data...")
    train_samples = load_worst_cell_samples("train")
    eval_samples = load_worst_cell_samples("repair_val")
    all_samples = train_samples + eval_samples
    print(f"  Train: {len(train_samples)}, Eval: {len(eval_samples)}")

    print("\nPhase 0b: Generating depth maps...")
    generate_depth_maps(all_samples, device="cuda")

    print("\nPhase 0c: Generating auxiliary depth-ordering samples...")
    aux_train = generate_auxiliary_samples(train_samples, rng)
    print(f"  Generated {len(aux_train)} auxiliary samples from {len(train_samples)} originals")

    aux_type_dist = Counter(s["aux_type"] for s in aux_train)
    for t, n in sorted(aux_type_dist.items()):
        print(f"    {t}: {n}")

    # Combine and mark original samples
    for s in train_samples:
        s["is_auxiliary"] = False

    # Subsample auxiliary if ratio < 1.0
    n_aux_target = int(len(train_samples) * args.aux_ratio)
    if len(aux_train) > n_aux_target:
        aux_indices = rng.choice(len(aux_train), n_aux_target, replace=False)
        aux_train = [aux_train[i] for i in aux_indices]
        print(f"  Subsampled to {len(aux_train)} auxiliary (ratio={args.aux_ratio})")

    combined_train = train_samples + aux_train
    print(f"  Combined training set: {len(combined_train)} "
          f"({len(train_samples)} original + {len(aux_train)} auxiliary)")

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

    train_dist = Counter(get_cell_id(s) for s in train_samples)
    for c, n in sorted(train_dist.items()):
        print(f"    {c}: {n}")

    partition = MondrianPartition.load(B1_DIR / "partition.json")

    # ── Training setup ────────────────────────────────────────────────
    n_combined = len(combined_train)
    steps_per_epoch = n_combined // GRAD_ACCUM
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    print(f"\n  Combined samples: {n_combined}")
    print(f"  Steps/epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")

    # ── Pre-training eval (original QA only) ──────────────────────────
    print("\nPre-training eval on worst cells (original QA, no depth map)...")
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

        indices = rng.permutation(n_combined)
        optimizer.zero_grad()
        epoch_loss_orig = 0.0
        epoch_loss_aux = 0.0
        n_orig = 0
        n_aux = 0
        n_steps = 0

        for micro_step in range(n_combined):
            idx = int(indices[micro_step])
            sample = combined_train[idx]

            try:
                inputs, labels = tokenize_train_example(processor, sample, process_vision_info)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                labels = labels.to(model.device)

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss / GRAD_ACCUM
                loss.backward()

                if sample.get("is_auxiliary"):
                    epoch_loss_aux += outputs.loss.item()
                    n_aux += 1
                else:
                    epoch_loss_orig += outputs.loss.item()
                    n_orig += 1

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
                    avg_orig = epoch_loss_orig / max(n_orig, 1)
                    avg_aux = epoch_loss_aux / max(n_aux, 1)
                    print(f"  Step {n_steps}/{steps_per_epoch}, "
                          f"orig_loss={avg_orig:.4f}, aux_loss={avg_aux:.4f}")

        avg_orig_loss = epoch_loss_orig / max(n_orig, 1)
        avg_aux_loss = epoch_loss_aux / max(n_aux, 1)
        print(f"\n  Epoch {epoch} — orig_loss: {avg_orig_loss:.4f}, "
              f"aux_loss: {avg_aux_loss:.4f} "
              f"(n_orig={n_orig}, n_aux={n_aux})")

        # Eval (original QA only — no auxiliary at eval time!)
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
            "train_loss_original": avg_orig_loss,
            "train_loss_auxiliary": avg_aux_loss,
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
    aux_w10 = best_eval["worst_10pct_cvar"]
    aux_wc = best_eval["worst_cell_loss"]

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
    print("COMPARISON: Depth Auxiliary vs Specialist vs Global FT")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25s} {'Depth Aux':>12s} {'Specialist':>12s} {'Global FT':>12s}")
    print(f"  {'-' * 65}")
    print(f"  {'W10% CVaR':<25s} {aux_w10:>12.4f} {SPECIALIST_W10_CVAR:>12.4f} {GLOBAL_W10_CVAR:>12.4f}")
    print(f"  {'Worst cell loss':<25s} {aux_wc:>12.4f} {SPECIALIST_WORST_CELL:>12.4f} {GLOBAL_WORST_CELL:>12.4f}")

    print(f"\n  Per-cell comparison:")
    print(f"  {'Cell':<30s} {'Depth Aux':>12s} {'Specialist':>12s} {'Global FT':>12s}")
    for cid in WORST_CELLS:
        da = best_eval["cell_losses"].get(cid, float("nan"))
        sp = specialist_cell_losses.get(cid, float("nan"))
        gl = global_cell_losses.get(cid, float("nan"))
        print(f"  {cid:<30s} {da:>12.4f} {sp:>12.4f} {gl:>12.4f}")

    # Go/No-Go
    print(f"\n{'=' * 60}")
    print("GO/NO-GO ASSESSMENT")
    print(f"{'=' * 60}")

    if aux_wc < 0.48:
        signal = "POSITIVE"
        verdict = "Depth auxiliary supervision breaks the loss ceiling → depth IS the missing info, learnable from original images"
    elif aux_wc < 0.51:
        signal = "WEAK POSITIVE"
        verdict = "Some improvement from depth supervision → partial signal, consider scaling up"
    else:
        signal = "NEGATIVE"
        verdict = "Depth auxiliary can't improve worst cells → coarse region-level depth ordering insufficient, may need object-level depth"

    depth_vs_specialist = aux_wc - SPECIALIST_WORST_CELL
    depth_vs_global = aux_wc - GLOBAL_WORST_CELL

    print(f"  Signal: {signal}")
    print(f"  Verdict: {verdict}")
    print(f"  vs Specialist: {depth_vs_specialist:+.4f}")
    print(f"  vs Global FT:  {depth_vs_global:+.4f}")
    print(f"  Training time: {elapsed / 3600:.1f}h")

    # ── Save results ──────────────────────────────────────────────────
    output = {
        "pilot": "depth_pseudo_label_auxiliary",
        "pilot_type": "TRUE auxiliary (not proxy)",
        "hypothesis": "depth_supervision_from_original_image",
        "approximation": "region-level depth ordering (no bboxes); left/right, upper/lower, quadrant",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "depth_model": DEPTH_MODEL,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "worst_cells": WORST_CELLS,
        "n_train_original": len(train_samples),
        "n_train_auxiliary": len(aux_train),
        "n_train_combined": len(combined_train),
        "n_eval": len(eval_samples),
        "aux_type_distribution": dict(aux_type_dist),
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
            "depth_aux_w10_cvar": aux_w10,
            "depth_aux_worst_cell": aux_wc,
            "delta_vs_specialist": depth_vs_specialist,
            "delta_vs_global": depth_vs_global,
        },
        "signal": signal,
        "verdict": verdict,
    }

    out_path = run_dir / "pilot_depth_auxiliary.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
