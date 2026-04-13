#!/usr/bin/env python3
"""
B2-v3: Group Construction Ablation — compare different group units for CVaR repair.

Extends B2-v2 with 2 new methods that use the SAME CVaR repair recipe but
different group construction strategies:

Methods (from B2-v2, unchanged):
  global         — Global FT matched budget (control)
  cvar_cell      — Cell-Level CVaR DRO on Mondrian cells (B2-v2 main)
  jtt_cell       — JTT-Cell (two-stage: ERM warm-start → cell-targeted upweight)
  cell_only      — Uniform upweight of worst-k cells

New methods (B2-v3):
  cluster_cvar   — KMeans clustering of Mondrian cells → group-level CVaR
  lossgroup_cvar — Loss-bin grouping of Mondrian cells → group-level CVaR

Research question:
  Is Mondrian cell the optimal repair unit, or do coarser groups perform better?
  - cluster_cvar > cvar_cell → Mondrian good for diagnosis, not repair target
  - lossgroup_cvar ≈ cluster_cvar → loss-based grouping sufficient, CP not needed

Usage:
  # Quick single run
  CUDA_VISIBLE_DEVICES=0 python scripts/run_b2v3.py --method cluster_cvar --seed 1

  # Full B2-v3 sweep
  for method in global cvar_cell cluster_cvar lossgroup_cvar; do
    for seed in 1 2 3; do
      CUDA_VISIBLE_DEVICES=0 python scripts/run_b2v3.py --method $method --seed $seed &
    done
  done
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
from collections import defaultdict

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.diagnosis.mondrian_partition import MondrianPartition
from src.diagnosis.group_construction import (
    GroupPartition, build_cluster_groups, build_lossgroup_groups,
)

# ── Paths ─────────────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
CACHE_DIR  = "/LOCAL2/psqhe8/hf_cache"
SPLITS_DIR = Path("data/splits")
B1_DIR     = Path("results/sprint2/b1_diagnosis")
OUT_DIR    = Path("results/sprint2/b2v3")

# ── QLoRA config (identical across all methods) ──────────────────────
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

# ── Training config (matched budget) ─────────────────────────────────
LR             = 2e-4
WEIGHT_DECAY   = 0.01
MAX_GRAD_NORM  = 1.0
MAX_EPOCHS     = 3
MICRO_BS       = 1
GRAD_ACCUM     = 16
WARMUP_RATIO   = 0.03
PATIENCE       = 2          # increased from 1 (reviewer feedback)

# ── CVaR Cell config ─────────────────────────────────────────────────
CVAR_ALPHA         = 0.1    # worst 10% of cells (~8 of 74)
RESCORE_INTERVAL   = 0.25   # rescore cells every 0.25 epoch
WARMUP_EPOCHS      = 0.5    # uniform weights (multiplier=1.0) for first 0.5 epoch, then enable CVaR + rescore
MULTIPLIER_CLIP    = 5.0    # max gradient multiplier per sample
MIN_CELL_SUPPORT   = 20     # ignore cells with fewer repair_val samples

# ── JTT-Cell config ──────────────────────────────────────────────────
JTT_WARMUP_EPOCHS  = 1      # stage 1: full ERM epoch
JTT_WORST_CELLS_K  = 12     # select from worst k cells
JTT_HARD_FRAC      = 0.2    # top 20% loss within worst cells
JTT_UPWEIGHT       = 5.0    # upweight factor for hard samples

# ── Cell-only config ─────────────────────────────────────────────────
CELL_ONLY_K        = 8      # upweight worst 8 cells
CELL_ONLY_WEIGHT   = 3.0    # uniform upweight factor

# ── Eval config ──────────────────────────────────────────────────────
EVAL_EVERY_EPOCH = True

# ── Prompt templates (same as B2) ────────────────────────────────────
BINARY_PROMPT = 'Look at the image. Is the following spatial statement true or false?\n\nStatement: "{caption}"\n\nAnswer with ONLY "true" or "false".'
OPEN_PROMPT   = 'Look at the image carefully. Answer the following spatial reasoning question with a short answer.\n\nQuestion: {question}\n\nAnswer:'
SPATIAL_PROMPT = 'Look at the image carefully. Answer the following spatial reasoning question.\n\nQuestion: {question}\n\nChoose the correct answer from: {choices}\n\nAnswer with ONLY the letter or the exact answer text, nothing else.'


# ═══════════════════════════════════════════════════════════════════════
# Model loading dispatch
# ═══════════════════════════════════════════════════════════════════════
def load_vlm_model(model_id, cache_dir, quantization_config, **kwargs):
    """Load the correct VLM class based on model_id."""
    if "Qwen2-VL" in model_id:
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, cache_dir=cache_dir, quantization_config=quantization_config,
            **kwargs,
        )
    elif "Qwen3-VL" in model_id:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, cache_dir=cache_dir, quantization_config=quantization_config,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported model_id: {model_id}. "
            f"Expected model_id to contain 'Qwen2-VL' or 'Qwen3-VL'."
        )


def safe_model_tag(model_id):
    """Extract a filesystem-safe short tag from model_id.

    Returns '' for the default model (backward compat), otherwise e.g. 'qwen3vl2b'.
    """
    if model_id == DEFAULT_MODEL_ID:
        return ""
    # e.g. "Qwen/Qwen3-VL-4B-Instruct" -> "qwen3-vl-4b-instruct"
    name = model_id.split("/")[-1].lower()
    # Remove "instruct" suffix and dashes for compactness
    name = name.replace("-instruct", "").replace("-", "")
    return name


# ═══════════════════════════════════════════════════════════════════════
# Data helpers (unchanged from B2)
# ═══════════════════════════════════════════════════════════════════════
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
    """Build a training example: tokenize prompt+answer, mask prompt tokens."""
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


def save_checkpoint(model, processor, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    ckpt_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    print(f"  Checkpoint saved: {path} ({ckpt_size/1e6:.1f} MB)")


# ═══════════════════════════════════════════════════════════════════════
# Cell loss estimation
# ═══════════════════════════════════════════════════════════════════════
def estimate_cell_losses(model, processor, process_vision_info,
                         samples, partition, min_support=MIN_CELL_SUPPORT):
    """Compute per-cell mean CE loss on ALL provided samples.

    Uses full sample set (no subsampling) to ensure stable cell loss
    estimates. Cells with fewer than min_support samples are excluded
    to prevent noisy η estimation.

    Returns dict: cell_id -> mean_loss (only cells with >= min_support samples).
    """
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

        if (i + 1) % 1000 == 0:
            print(f"    cell estimation: {i+1}/{len(samples)}")

    cell_mean = {cid: float(np.mean(ls)) for cid, ls in cell_losses.items()
                 if len(ls) >= min_support}
    n_excluded = sum(1 for ls in cell_losses.values() if len(ls) < min_support)
    if n_excluded > 0:
        print(f"    Excluded {n_excluded} cells with <{min_support} samples")
    model.train()
    return cell_mean


def estimate_cell_losses_detailed(model, processor, process_vision_info,
                                  samples, partition, min_support=MIN_CELL_SUPPORT):
    """Like estimate_cell_losses but also returns per-cell loss lists.

    Used by cluster_cvar to compute loss_std for clustering features.
    Returns (cell_mean_losses, cell_loss_lists) where cell_loss_lists
    is {cell_id: [loss_values]} for cells with >= min_support samples.
    """
    model.eval()
    cell_loss_lists = defaultdict(list)
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
            cell_loss_lists[cid].append(outputs.loss.item())
        except Exception:
            cell_loss_lists[cid].append(10.0)
        if (i + 1) % 1000 == 0:
            print(f"    cell estimation (detailed): {i+1}/{len(samples)}")

    filtered = {cid: ls for cid, ls in cell_loss_lists.items()
                if len(ls) >= min_support}
    cell_mean = {cid: float(np.mean(ls)) for cid, ls in filtered.items()}
    n_excluded = len(cell_loss_lists) - len(filtered)
    if n_excluded > 0:
        print(f"    Excluded {n_excluded} cells with <{min_support} samples")
    model.train()
    return cell_mean, filtered


# ═══════════════════════════════════════════════════════════════════════
# Method: Cell-Level Dual CVaR DRO
# ═══════════════════════════════════════════════════════════════════════
class CellCVaRWeighter:
    """
    Rockafellar-Uryasev CVaR at the cell level:
      CVaR_α = η + (1/(α·|C|)) Σ_c [L_c - η]_+

    Each sample gets a gradient multiplier based on whether its cell is
    in the worst-α tail (binary: clip or 1.0).

    Key design (from Rockafellar-Uryasev + reviewer feedback):
    - η is the exact empirical quantile, NOT learned by SGD
    - All tail cells get the SAME multiplier (true CVaR, not rank-interpolated)
    - Cell losses are rescored periodically on full repair_val
    - First warmup_epochs use uniform weights (multiplier=1.0 for all)
    """

    def __init__(self, partition, alpha=CVAR_ALPHA,
                 multiplier_clip=MULTIPLIER_CLIP,
                 min_support=MIN_CELL_SUPPORT):
        self.partition = partition
        self.alpha = alpha
        self.multiplier_clip = multiplier_clip
        self.min_support = min_support
        self.cell_losses = {}       # cell_id -> mean_loss
        self.eta = 0.0              # CVaR threshold (quantile)
        self.sample_multipliers = {}  # precomputed per-cell multiplier
        self.is_warm = False
        self.cvar_fallback_reason = None

    def update(self, cell_losses: dict):
        """Rescore: update cell losses and recompute binary tail multipliers.

        True Rockafellar-Uryasev CVaR gradient: ∇CVaR_α = E[∇L | L > η].
        All cells in the worst-α tail get the SAME multiplier (= clip),
        all other cells get 1.0. This is the exact CVaR formulation —
        no rank interpolation, no excess scaling.
        """
        self.cell_losses = cell_losses
        if not cell_losses:
            print("  [CVaR] WARNING: no eligible cells (all below min_support). Keeping stale weights.")
            self.cvar_fallback_reason = "no_eligible_cells"
            return

        # Compute tail size k, then check if CVaR is feasible
        sorted_cells = sorted(cell_losses.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_cells)
        k = max(1, int(np.ceil(n * self.alpha)))

        # CVaR requires at least 1 tail and 1 non-tail unit
        if n < 2 or k >= n:
            reason = f"n={n},k={k},alpha={self.alpha}"
            print(f"  [CVaR] WARNING: cannot form tail/non-tail split ({reason}). "
                  f"Falling back to uniform weights.")
            self.sample_multipliers = {cid: 1.0 for cid in cell_losses}
            self.is_warm = True
            self.cvar_fallback_reason = reason
            return
        self.cvar_fallback_reason = None

        # η = loss of the k-th worst cell (CVaR threshold)
        self.eta = sorted_cells[k - 1][1] if k <= n else sorted_cells[-1][1]

        # Binary CVaR multipliers: tail cells get uniform clip, rest get 1.0
        # This matches ∇CVaR_α = E[∇L | L > η] (equal weight for all tail cells)
        self.sample_multipliers = {}
        for rank, (cid, loss) in enumerate(sorted_cells):
            if rank < k:
                self.sample_multipliers[cid] = self.multiplier_clip
            else:
                self.sample_multipliers[cid] = 1.0

        self.is_warm = True
        n_active = sum(1 for m in self.sample_multipliers.values() if m > 1.01)
        print(f"  [CVaR] η={self.eta:.4f}, active cells={n_active}/{n}, "
              f"k={k}, multiplier_tail={self.multiplier_clip:.1f}, multiplier_rest=1.0")

    def get_multiplier(self, sample: dict, global_step: int = None,
                       warmup_step_count: int = 0) -> float:
        """Get gradient multiplier for a training sample.

        During warm-start (global_step <= warmup_step_count), returns 1.0
        for all samples — the model learns basics with uniform weights first.
        After warm-start, returns clip for tail cells and 1.0 for the rest.
        """
        if not self.is_warm:
            return 1.0
        if global_step is not None and global_step <= warmup_step_count:
            return 1.0
        cid = self.partition.get_cell_by_features(sample)
        return self.sample_multipliers.get(cid, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# Method: JTT-Cell
# ═══════════════════════════════════════════════════════════════════════
def identify_jtt_hard_samples(model, processor, process_vision_info,
                              train_samples, partition, cell_losses,
                              worst_k=JTT_WORST_CELLS_K,
                              hard_frac=JTT_HARD_FRAC):
    """
    Stage 1 → Stage 2 transition:
    1. Identify worst-k cells by current cell loss
    2. Within those cells, find top hard_frac samples by CE loss
    Returns set of sample indices to upweight.
    """
    # Find worst-k cells
    sorted_cells = sorted(cell_losses.items(), key=lambda x: x[1], reverse=True)
    worst_cells = set(cid for cid, _ in sorted_cells[:worst_k])
    print(f"  [JTT] Worst {worst_k} cells: {list(worst_cells)[:5]}...")

    # Collect samples in worst cells + their losses
    candidate_indices = []
    for i, s in enumerate(train_samples):
        cid = partition.get_cell_by_features(s)
        if cid in worst_cells:
            candidate_indices.append(i)

    print(f"  [JTT] Candidates from worst cells: {len(candidate_indices)}")

    # Score candidates by CE loss
    model.eval()
    scored = []
    for idx in candidate_indices:
        s = train_samples[idx]
        try:
            inputs, labels = tokenize_train_example(processor, s, process_vision_info)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            labels = labels.to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, labels=labels)
            scored.append((idx, outputs.loss.item()))
        except Exception:
            scored.append((idx, 10.0))

        if len(scored) % 1000 == 0:
            print(f"    scoring: {len(scored)}/{len(candidate_indices)}")

    model.train()

    # Select top hard_frac
    scored.sort(key=lambda x: x[1], reverse=True)
    n_hard = max(1, int(len(scored) * hard_frac))
    hard_set = set(idx for idx, _ in scored[:n_hard])
    print(f"  [JTT] Hard samples selected: {len(hard_set)} "
          f"(loss range: {scored[0][1]:.3f} - {scored[n_hard-1][1]:.3f})")

    return hard_set


# ═══════════════════════════════════════════════════════════════════════
# Method: Cell-only upweight
# ═══════════════════════════════════════════════════════════════════════
def compute_cell_only_weights(train_samples, partition, cell_losses,
                              worst_k=CELL_ONLY_K, upweight=CELL_ONLY_WEIGHT):
    """Uniform upweight for samples in worst-k cells. No CVaR objective."""
    sorted_cells = sorted(cell_losses.items(), key=lambda x: x[1], reverse=True)
    worst_cells = set(cid for cid, _ in sorted_cells[:worst_k])

    weights = []
    n_up = 0
    for s in train_samples:
        cid = partition.get_cell_by_features(s)
        if cid in worst_cells:
            weights.append(upweight)
            n_up += 1
        else:
            weights.append(1.0)

    print(f"  [CellOnly] Worst {worst_k} cells, upweight={upweight}, "
          f"n_upweighted={n_up}/{len(train_samples)}")
    return weights


# ═══════════════════════════════════════════════════════════════════════
# Evaluation (same as B2, with minor cleanup)
# ═══════════════════════════════════════════════════════════════════════
def evaluate_loss_cvar(model, processor, process_vision_info, eval_samples,
                       partition, alpha=0.1, min_support=MIN_CELL_SUPPORT):
    """Per-cell loss → worst-α-cell CVaR. No generation needed.

    Reports BOTH filtered (min_support) and unfiltered CVaR so we can
    compare apples-to-apples with training-side cell selection.
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
            sample_cells.append(partition.get_cell_by_features(s))
        except Exception:
            errors += 1
            sample_losses.append(10.0)
            sample_cells.append(None)

        if (i + 1) % 500 == 0:
            print(f"    eval: {i+1}/{len(eval_samples)}")

    # Per-cell mean loss (all cells)
    cell_losses_map = defaultdict(list)
    for loss, cid in zip(sample_losses, sample_cells):
        if cid is not None:
            cell_losses_map[cid].append(loss)
    cell_mean_loss_all = {cid: np.mean(ls) for cid, ls in cell_losses_map.items()}

    # Filtered: only cells with >= min_support (matches training-side filter)
    cell_mean_loss_filtered = {cid: np.mean(ls) for cid, ls in cell_losses_map.items()
                               if len(ls) >= min_support}

    def compute_cvar(cell_mean_loss, alpha):
        if cell_mean_loss:
            sorted_losses = sorted(cell_mean_loss.values(), reverse=True)
            k = max(1, int(np.ceil(len(sorted_losses) * alpha)))
            worst_cvar = float(np.mean(sorted_losses[:k]))
            worst_cells = sorted(cell_mean_loss.items(), key=lambda x: x[1], reverse=True)[:k]
        else:
            worst_cvar = 10.0
            worst_cells = []
        return worst_cvar, worst_cells

    cvar_all, worst_cells_all = compute_cvar(cell_mean_loss_all, alpha)
    cvar_filtered, worst_cells_filtered = compute_cvar(cell_mean_loss_filtered, alpha)

    overall_loss = float(np.mean(sample_losses))
    n_excluded = len(cell_mean_loss_all) - len(cell_mean_loss_filtered)

    return {
        "overall_loss": overall_loss,
        # Primary metric: filtered (consistent with training-side cell selection)
        "worst_10pct_cvar": cvar_filtered,
        # Secondary: unfiltered (for backwards compatibility / comparison)
        "worst_10pct_cvar_unfiltered": cvar_all,
        "n_cells_evaluated": len(cell_mean_loss_filtered),
        "n_cells_total": len(cell_mean_loss_all),
        "n_cells_excluded": n_excluded,
        "worst_cell_loss": float(max(cell_mean_loss_filtered.values())) if cell_mean_loss_filtered else 10.0,
        "best_cell_loss": float(min(cell_mean_loss_filtered.values())) if cell_mean_loss_filtered else 10.0,
        "eval_errors": errors,
        "worst_cells": [(cid, float(l)) for cid, l in worst_cells_filtered],
        "cell_losses": {cid: float(l) for cid, l in cell_mean_loss_all.items()},
    }


def evaluate_accuracy(model, processor, process_vision_info, eval_samples,
                      partition, alpha=0.1):
    """Full generation-based eval: accuracy + worst-cell CVaR(error)."""
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

    overall_acc = float(np.mean([r["correct"] for r in results]))

    cell_errors = defaultdict(list)
    for r in results:
        if r["cell_id"] is not None:
            cell_errors[r["cell_id"]].append(1.0 - float(r["correct"]))

    # Filtered (matches training-side min_support)
    cell_mean_error_filtered = {cid: np.mean(errs) for cid, errs in cell_errors.items()
                                if len(errs) >= MIN_CELL_SUPPORT}
    # Unfiltered (all cells)
    cell_mean_error_all = {cid: np.mean(errs) for cid, errs in cell_errors.items()}

    def cvar_from(cell_mean, alpha):
        if cell_mean:
            sorted_vals = sorted(cell_mean.values(), reverse=True)
            k = max(1, int(np.ceil(len(sorted_vals) * alpha)))
            return float(np.mean(sorted_vals[:k]))
        return 1.0

    return {
        "overall_accuracy": overall_acc,
        "worst_10pct_cvar_error": cvar_from(cell_mean_error_filtered, alpha),
        "worst_10pct_cvar_error_unfiltered": cvar_from(cell_mean_error_all, alpha),
        "n_cells_evaluated": len(cell_mean_error_filtered),
        "n_cells_total": len(cell_mean_error_all),
        "worst_cell_error": float(max(cell_mean_error_filtered.values())) if cell_mean_error_filtered else 1.0,
        "best_cell_error": float(min(cell_mean_error_filtered.values())) if cell_mean_error_filtered else 1.0,
        "n_evaluated": len(results),
    }


# ═══════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="B2-v3: Group Construction Ablation")
    parser.add_argument("--method", choices=[
        "cvar_cell", "jtt_cell", "cell_only", "global",
        "cluster_cvar", "lossgroup_cvar",
    ], required=True)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                        help="HuggingFace model ID (default: Qwen/Qwen2-VL-2B-Instruct)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--eval_samples", type=int, default=0)
    parser.add_argument("--skip_final_gen_eval", action="store_true")
    # CVaR hyperparams
    parser.add_argument("--cvar_alpha", type=float, default=CVAR_ALPHA)
    parser.add_argument("--multiplier_clip", type=float, default=MULTIPLIER_CLIP)
    parser.add_argument("--warmup_epochs", type=float, default=WARMUP_EPOCHS)
    parser.add_argument("--rescore_interval", type=float, default=RESCORE_INTERVAL)
    # JTT hyperparams
    parser.add_argument("--jtt_worst_k", type=int, default=JTT_WORST_CELLS_K)
    parser.add_argument("--jtt_hard_frac", type=float, default=JTT_HARD_FRAC)
    parser.add_argument("--jtt_upweight", type=float, default=JTT_UPWEIGHT)
    # Cell-only hyperparams
    parser.add_argument("--cell_only_k", type=int, default=CELL_ONLY_K)
    parser.add_argument("--cell_only_weight", type=float, default=CELL_ONLY_WEIGHT)
    # B2-v3: group construction
    parser.add_argument("--n_groups", type=int, default=4,
                        help="Number of groups for cluster_cvar / lossgroup_cvar")
    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="biasmap-cp")
    args = parser.parse_args()

    torch.manual_seed(args.seed + 42)
    np.random.seed(args.seed + 42)

    model_tag = safe_model_tag(args.model_id)
    run_name = f"b2v3_{args.method}_seed{args.seed}"
    if model_tag:
        run_name += f"_{model_tag}"
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── wandb ──
    use_wandb = args.wandb and HAS_WANDB
    if use_wandb:
        wandb.init(
            project=args.wandb_project, name=run_name,
            config=vars(args),
            tags=["b2v3", args.method, f"seed{args.seed}"],
        )

    print("=" * 60)
    print(f"B2-v3: {args.method} (seed={args.seed})")
    print(f"Output: {run_dir}")
    print("=" * 60)

    # Helper: which methods use CVaR-style training?
    is_cvar_method = args.method in ("cvar_cell", "cluster_cvar", "lossgroup_cvar")

    # ── 1. Load model ──
    from transformers import AutoProcessor
    from transformers import BitsAndBytesConfig, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model, TaskType
    from qwen_vl_utils import process_vision_info

    print(f"\nLoading model: {args.model_id}")
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
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # ── 2. Load data ──
    print("\nLoading data...")
    train_samples = load_samples("train")
    eval_max = args.eval_samples if args.eval_samples > 0 else None
    repair_val = load_samples("repair_val", eval_max)
    print(f"  Train: {len(train_samples)}, Repair_val: {len(repair_val)}")

    # ── 3. Load partition ──
    partition = MondrianPartition.load(B1_DIR / "partition.json")

    # Pre-map train samples to cells for fast lookup
    train_cell_map = {}  # index -> cell_id
    cell_train_indices = defaultdict(list)  # cell_id -> [indices]
    for i, s in enumerate(train_samples):
        cid = partition.get_cell_by_features(s)
        train_cell_map[i] = cid
        if cid:
            cell_train_indices[cid].append(i)
    print(f"  Mapped {sum(1 for v in train_cell_map.values() if v)}/{len(train_samples)} "
          f"train samples to {len(cell_train_indices)} cells")

    # ── 4. Method-specific setup ──
    n_train = len(train_samples)
    steps_per_epoch = n_train // args.grad_accum
    total_steps = steps_per_epoch * args.max_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    # For JTT: stage 1 uses first epoch, stage 2 uses remaining epochs
    # Total gradient updates are matched: JTT gets max_epochs total
    jtt_hard_set = None
    jtt_stage = 1 if args.method == "jtt_cell" else None

    # CVaR weighter (used by cvar_cell, cluster_cvar, lossgroup_cvar)
    # Partition will be set after group construction for new methods
    cvar_weighter = None
    train_partition = partition   # partition used for CVaR targeting
    eval_partition = partition    # partition used for eval (always Mondrian)
    group_partition = None        # only set for cluster/lossgroup methods

    # Cell-only weights (computed after initial cell loss estimation)
    cell_only_sample_weights = None

    # ── 5. Optimizer + scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    print(f"\nTraining config:")
    print(f"  Method:          {args.method}")
    print(f"  Epochs:          {args.max_epochs}")
    print(f"  Steps/epoch:     {steps_per_epoch}")
    print(f"  Total steps:     {total_steps}")
    print(f"  Patience:        {args.patience}")

    # ── 6. Initial cell loss estimation (for methods that need it) ──
    needs_cell_losses = args.method in (
        "cvar_cell", "cell_only", "jtt_cell", "cluster_cvar", "lossgroup_cvar",
    )
    init_cell_losses = None
    init_cell_loss_lists = None

    if needs_cell_losses:
        if args.method == "cluster_cvar":
            # cluster_cvar needs per-cell loss lists for std computation
            print("\nEstimating initial cell losses (detailed) on repair_val...")
            init_cell_losses, init_cell_loss_lists = estimate_cell_losses_detailed(
                model, processor, process_vision_info, repair_val, partition,
            )
        else:
            print("\nEstimating initial cell losses on repair_val...")
            init_cell_losses = estimate_cell_losses(
                model, processor, process_vision_info, repair_val, partition,
            )
        print(f"  Estimated {len(init_cell_losses)} eligible cells")

    # ── 6b. Build group partition for cluster/lossgroup methods ──
    if args.method == "cluster_cvar":
        print(f"\nBuilding cluster groups (n_groups={args.n_groups})...")
        group_partition = build_cluster_groups(
            partition, init_cell_losses,
            n_groups=args.n_groups,
            cell_loss_lists=init_cell_loss_lists,
        )
        train_partition = group_partition
        print(group_partition.summary())
        # Log group composition
        group_comp = group_partition.get_group_composition()
        print(f"  Group composition:")
        for gid in sorted(group_comp.keys()):
            print(f"    {gid}: {group_comp[gid]}")

    elif args.method == "lossgroup_cvar":
        print(f"\nBuilding loss-bin groups (n_groups={args.n_groups})...")
        group_partition = build_lossgroup_groups(
            partition, init_cell_losses, n_groups=args.n_groups,
        )
        train_partition = group_partition
        print(group_partition.summary())
        group_comp = group_partition.get_group_composition()
        print(f"  Group composition:")
        for gid in sorted(group_comp.keys()):
            print(f"    {gid}: {group_comp[gid]}")

    # ── 6c. Initialize CVaR weighter ──
    if is_cvar_method:
        cvar_weighter = CellCVaRWeighter(
            train_partition, alpha=args.cvar_alpha,
            multiplier_clip=args.multiplier_clip,
        )
        if is_cvar_method and init_cell_losses is not None:
            # For cluster/lossgroup, compute group-level losses for initial CVaR
            if args.method in ("cluster_cvar", "lossgroup_cvar"):
                group_losses = {}
                for gid, members in group_partition.groups.items():
                    member_losses = [init_cell_losses[c] for c in members
                                     if c in init_cell_losses]
                    if member_losses:
                        group_losses[gid] = float(np.mean(member_losses))
                print(f"  Initial group-level losses: {group_losses}")
                cvar_weighter.update(group_losses)
            else:
                print("  Initializing CVaR weighter with zero-shot cell losses...")
                cvar_weighter.update(init_cell_losses)

    if args.method == "cell_only":
        cell_only_sample_weights = compute_cell_only_weights(
            train_samples, partition, init_cell_losses,
            worst_k=args.cell_only_k, upweight=args.cell_only_weight,
        )

    # ── 7. Training loop ──
    model.train()
    best_metric = float("inf")
    best_epoch = 0
    patience_counter = 0
    epoch_history = []
    total_errors = 0
    losses_log = []
    rescore_log = []

    train_start = time.time()
    rescore_step_interval = max(1, int(steps_per_epoch * args.rescore_interval))
    warmup_step_count = int(steps_per_epoch * args.warmup_epochs)
    warmup_rescore_done = not is_cvar_method  # all CVaR methods need warmup rescore

    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.max_epochs}")
        if args.method == "jtt_cell" and jtt_stage is not None:
            print(f"JTT Stage: {jtt_stage}")
        print(f"{'='*60}")

        # -- JTT transition: after stage 1, identify hard samples --
        if args.method == "jtt_cell" and epoch == JTT_WARMUP_EPOCHS + 1 and jtt_stage == 1:
            print("\n[JTT] Transitioning to Stage 2...")
            # Rescore cells with current model
            stage1_cell_losses = estimate_cell_losses(
                model, processor, process_vision_info, repair_val, partition,
            )
            jtt_hard_set = identify_jtt_hard_samples(
                model, processor, process_vision_info,
                train_samples, partition, stage1_cell_losses,
                worst_k=args.jtt_worst_k, hard_frac=args.jtt_hard_frac,
            )
            jtt_stage = 2
            model.train()

        # -- Build sampling weights for this epoch --
        if args.method == "global":
            sample_probs = None  # uniform
        elif args.method == "cell_only":
            w = np.array(cell_only_sample_weights)
            sample_probs = w / w.sum()
        elif args.method == "jtt_cell":
            if jtt_stage == 1 or jtt_hard_set is None:
                sample_probs = None  # uniform during stage 1
            else:
                w = np.ones(n_train)
                for idx in jtt_hard_set:
                    w[idx] = args.jtt_upweight
                sample_probs = w / w.sum()
        elif is_cvar_method:
            sample_probs = None  # CVaR uses per-sample multiplier, not sampling

        # Shuffle / sample indices
        if sample_probs is not None:
            indices = np.random.choice(n_train, size=n_train, replace=True, p=sample_probs)
        else:
            indices = np.random.permutation(n_train)

        optimizer.zero_grad()
        accum_loss = 0.0
        epoch_loss = 0.0
        epoch_batches = 0
        log_interval = max(steps_per_epoch // 20, 1)

        for micro_step in range(n_train):
            idx = int(indices[micro_step])
            sample = train_samples[idx]
            global_step = (epoch - 1) * steps_per_epoch + (micro_step + 1) // args.grad_accum

            # -- CVaR: immediate rescore at warm-start end --
            # Ensures the model transitions from zero-shot cell rankings
            # to fresh rankings based on the warm-start-trained model.
            did_rescore_this_step = False
            if (is_cvar_method
                and cvar_weighter is not None
                and not warmup_rescore_done
                and global_step > warmup_step_count):
                warmup_rescore_done = True
                did_rescore_this_step = True
                print(f"\n  [CVaR] Warm-start ended at step {global_step}. "
                      f"Rescoring with warm-started model...")
                rescore_t0 = time.time()
                current_cell_losses = estimate_cell_losses(
                    model, processor, process_vision_info, repair_val, train_partition,
                )
                cvar_weighter.update(current_cell_losses)
                tail_cells = [cid for cid, m in cvar_weighter.sample_multipliers.items()
                              if m > 1.01]
                rescore_log.append({
                    "global_step": global_step,
                    "eta": cvar_weighter.eta,
                    "n_cells": len(current_cell_losses),
                    "tail_cell_ids": tail_cells,
                    "cell_losses": {str(k): v for k, v in current_cell_losses.items()},
                    "time_s": time.time() - rescore_t0,
                    "trigger": "warmup_end",
                    "partition_type": "group" if group_partition else "mondrian",
                    "cvar_fallback": cvar_weighter.cvar_fallback_reason,
                })
                model.train()
                print(f"  CVaR multipliers now active (tail={cvar_weighter.multiplier_clip:.1f}).")

            # -- CVaR: periodic rescore --
            if (is_cvar_method
                and cvar_weighter is not None
                and (micro_step + 1) % (rescore_step_interval * args.grad_accum) == 0
                and global_step > warmup_step_count
                and warmup_rescore_done
                and not did_rescore_this_step):
                print(f"\n  [CVaR] Periodic rescore at step {global_step}...")
                rescore_t0 = time.time()
                current_cell_losses = estimate_cell_losses(
                    model, processor, process_vision_info, repair_val, train_partition,
                )
                cvar_weighter.update(current_cell_losses)
                tail_cells = [cid for cid, m in cvar_weighter.sample_multipliers.items()
                              if m > 1.01]
                rescore_log.append({
                    "global_step": global_step,
                    "eta": cvar_weighter.eta,
                    "n_cells": len(current_cell_losses),
                    "tail_cell_ids": tail_cells,
                    "cell_losses": {str(k): v for k, v in current_cell_losses.items()},
                    "time_s": time.time() - rescore_t0,
                    "trigger": "periodic",
                    "partition_type": "group" if group_partition else "mondrian",
                    "cvar_fallback": cvar_weighter.cvar_fallback_reason,
                })
                model.train()

            try:
                inputs, labels = tokenize_train_example(
                    processor, sample, process_vision_info,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                labels = labels.to(model.device)

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss / args.grad_accum

                # -- CVaR: apply gradient multiplier --
                # During warm-start (first warmup_epochs), multiplier=1.0 (uniform)
                if is_cvar_method and cvar_weighter is not None:
                    multiplier = cvar_weighter.get_multiplier(
                        sample, global_step=global_step,
                        warmup_step_count=warmup_step_count,
                    )
                    loss = loss * multiplier

                loss.backward()
                accum_loss += outputs.loss.item()  # log unscaled loss

            except Exception as e:
                total_errors += 1
                if total_errors <= 10:
                    print(f"  ⚠ micro_step {micro_step} error: {e}")
                elif total_errors == 11:
                    print("  ⚠ Suppressing further error messages...")
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

                cur_global_step = (epoch - 1) * steps_per_epoch + actual_step
                cur_lr = scheduler.get_last_lr()[0]

                losses_log.append({
                    "epoch": epoch, "step": actual_step,
                    "global_step": cur_global_step,
                    "loss": avg_l, "lr": cur_lr,
                })

                if use_wandb:
                    log_dict = {
                        "train/loss": avg_l, "train/lr": cur_lr,
                        "train/epoch": epoch,
                    }
                    if cvar_weighter and cvar_weighter.is_warm:
                        log_dict["train/cvar_eta"] = cvar_weighter.eta
                    wandb.log(log_dict, step=cur_global_step)

                if actual_step % log_interval == 0 or actual_step <= 3:
                    elapsed = time.time() - epoch_start
                    eta_time = elapsed / actual_step * (steps_per_epoch - actual_step)
                    extra = ""
                    if cvar_weighter and cvar_weighter.is_warm:
                        extra = f"  η={cvar_weighter.eta:.3f}"
                    print(f"  step {actual_step:5d}/{steps_per_epoch}: "
                          f"loss={avg_l:.4f}  lr={cur_lr:.2e}  "
                          f"elapsed={elapsed/60:.1f}m  eta={eta_time/60:.1f}m{extra}")

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
                eval_partition, alpha=args.cvar_alpha,
            )
            eval_elapsed = time.time() - eval_start
            model.train()

            current_metric = metrics["worst_10pct_cvar"]
            print(f"  Eval: overall_loss={metrics['overall_loss']:.4f}, "
                  f"worst_10pct_cvar={current_metric:.4f}, "
                  f"n_cells={metrics['n_cells_evaluated']}, "
                  f"time={eval_elapsed/60:.1f}m")

            epoch_entry = {
                "epoch": epoch,
                "train_loss": avg_epoch_loss,
                "eval_metrics": {k: v for k, v in metrics.items()
                                 if k not in ("worst_cells", "cell_losses")},
                # Kill condition 1: need worst-8 cell IDs for overlap analysis
                "worst_cells_top8": metrics["worst_cells"][:8],
                # Full cell losses for post-hoc analysis (kill conditions 1,3,5)
                "all_cell_losses": metrics["cell_losses"],
                "time_train_min": epoch_elapsed / 60,
                "time_eval_min": eval_elapsed / 60,
                "timestamp": datetime.now().isoformat(),
            }
            if args.method == "jtt_cell":
                epoch_entry["jtt_stage"] = jtt_stage
            epoch_history.append(epoch_entry)

            if use_wandb:
                wandb.log({
                    "eval/overall_loss": metrics["overall_loss"],
                    "eval/worst_10pct_cvar": current_metric,
                    "eval/worst_cell_loss": metrics["worst_cell_loss"],
                    "eval/best_cell_loss": metrics["best_cell_loss"],
                    "epoch/train_loss": avg_epoch_loss,
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
                "epoch": epoch, "train_loss": avg_epoch_loss,
                "time_train_min": epoch_elapsed / 60,
                "timestamp": datetime.now().isoformat(),
            })

        save_checkpoint(model, processor, run_dir / "checkpoint-latest")

        # Save intermediate log
        with open(run_dir / "training_log.json", "w") as f:
            json.dump({
                "method": args.method, "seed": args.seed,
                "epoch_history": epoch_history,
                "rescore_log": rescore_log,
                "config": vars(args),
            }, f, indent=2)

    total_time = time.time() - train_start

    # ── 8. Final generation-based eval ──
    gen_eval_results = None
    if not args.skip_final_gen_eval:
        print(f"\nFinal generation-based eval (best checkpoint, epoch {best_epoch})...")
        best_ckpt = run_dir / "checkpoint-best"
        if best_ckpt.exists():
            from peft import PeftModel
            base_model = load_vlm_model(
                args.model_id, cache_dir=CACHE_DIR, quantization_config=bnb_config,
                device_map="auto", trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, str(best_ckpt))
            print("  Loaded best checkpoint")

        gen_eval_results = evaluate_accuracy(
            model, processor, process_vision_info, repair_val,
            eval_partition, alpha=args.cvar_alpha,
        )
        print(f"  Accuracy: {gen_eval_results['overall_accuracy']:.3f}")
        print(f"  Worst-10%-cell CVaR(error): {gen_eval_results['worst_10pct_cvar_error']:.3f}")

        if use_wandb:
            wandb.log({
                "final/accuracy": gen_eval_results["overall_accuracy"],
                "final/worst_10pct_cvar_error": gen_eval_results["worst_10pct_cvar_error"],
            })

    # ── 9. Save summary ──
    summary = {
        "method": args.method,
        "seed": args.seed,
        "train_samples": n_train,
        "eval_samples": len(repair_val),
        "epochs_completed": len(epoch_history),
        "best_epoch": best_epoch,
        "best_worst_10pct_cvar": float(best_metric),
        "final_train_loss": float(epoch_history[-1]["train_loss"]) if epoch_history else None,
        "total_time_hours": total_time / 3600,
        "total_errors": total_errors,
        "epoch_history": epoch_history,
        "rescore_log": rescore_log,
        "gen_eval": gen_eval_results,
        "config": vars(args),
        # B2 baseline for easy comparison
        "b2_global_baseline": {
            "worst_10pct_cvar": 0.485,
            "accuracy": 0.823,
            "note": "B2 global FT seed1 best epoch2",
        },
    }

    with open(run_dir / "losses_log.json", "w") as f:
        json.dump(losses_log, f)
    # B2-v3: save group composition info
    if group_partition is not None:
        summary["group_construction"] = {
            "type": args.method,
            "n_groups": args.n_groups,
            "composition": {gid: members for gid, members
                            in group_partition.get_group_composition().items()},
            "group_info": group_partition.group_info,
        }

    with open(run_dir / "b2v3_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"B2-v3 COMPLETE: {args.method}")
    print(f"{'='*60}")
    print(f"  Seed:           {args.seed}")
    print(f"  Epochs:         {len(epoch_history)}/{args.max_epochs}")
    print(f"  Best epoch:     {best_epoch}")
    print(f"  Best CVaR:      {best_metric:.4f}  (B2 global baseline: 0.485)")
    delta = (0.485 - best_metric) / 0.485 * 100
    print(f"  Δ vs baseline:  {delta:+.1f}% {'✓ BETTER' if delta > 0 else '✗ WORSE'}")
    if gen_eval_results:
        print(f"  Final accuracy: {gen_eval_results['overall_accuracy']:.3f}  (B2 baseline: 0.823)")
        print(f"  Final CVaR(e):  {gen_eval_results['worst_10pct_cvar_error']:.3f}  (B2 baseline: 0.518)")
    print(f"  Total time:     {total_time/3600:.1f} hours")
    print(f"  Output:         {run_dir}")

    # Decision gate check
    print(f"\n{'='*60}")
    print("DECISION GATE CHECK")
    print(f"{'='*60}")
    target = 0.465  # must beat 0.485 by >=0.02 absolute
    if best_metric <= target:
        print(f"  ✓ PASS: {best_metric:.4f} <= {target} (target)")
    else:
        print(f"  ✗ FAIL: {best_metric:.4f} > {target} (target)")
    if gen_eval_results and gen_eval_results["overall_accuracy"] >= 0.808:
        print(f"  ✓ Accuracy guard: {gen_eval_results['overall_accuracy']:.3f} >= 0.808")
    elif gen_eval_results:
        print(f"  ✗ Accuracy guard: {gen_eval_results['overall_accuracy']:.3f} < 0.808")

    if use_wandb:
        wandb.summary.update({
            "best_epoch": best_epoch,
            "best_worst_10pct_cvar": float(best_metric),
            "delta_vs_baseline_pct": delta,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
