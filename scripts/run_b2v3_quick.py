#!/usr/bin/env python3
"""
B2-v3 Quick Test: Validate group construction methods on 200 samples / 50 steps.

Validates per method:
  1. Model loads with 4-bit + LoRA                              (all)
  2. Loss decreases over 50 steps                               (all)
  3. Cell loss estimation works                                  (cvar, cluster, lossgroup)
  4. CVaR warm-start → rescore → tail activation                 (cvar, cluster, lossgroup)
  5. Periodic rescore fires and updates multipliers              (cvar, cluster, lossgroup)
  6. Group construction: correct #groups, cell membership         (cluster, lossgroup)
  7. Active tail groups tracked across rescores                   (cluster, lossgroup)
  8. Checkpoint save/load works                                  (all)
  9. Eval on 100 repair_val samples produces metrics             (all)

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_b2v3_quick.py --method cluster_cvar
  CUDA_VISIBLE_DEVICES=0 python scripts/run_b2v3_quick.py --method lossgroup_cvar
  CUDA_VISIBLE_DEVICES=0 python scripts/run_b2v3_quick.py --method cvar_cell
  CUDA_VISIBLE_DEVICES=0 python scripts/run_b2v3_quick.py --method global

  # Run all 4 sequentially (~20-30 min total):
  for m in global cvar_cell cluster_cvar lossgroup_cvar; do
    CUDA_VISIBLE_DEVICES=0 python scripts/run_b2v3_quick.py --method $m
  done
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.diagnosis.mondrian_partition import MondrianPartition
from src.diagnosis.group_construction import (
    GroupPartition, build_cluster_groups, build_lossgroup_groups,
)

# ── Quick test config ─────────────────────────────────────────────────
MODEL_ID     = "Qwen/Qwen2-VL-2B-Instruct"
CACHE_DIR    = "/LOCAL2/psqhe8/hf_cache"
SPLITS_DIR   = Path("data/splits")
B1_DIR       = Path("results/sprint2/b1_diagnosis")
OUT_DIR      = Path("results/sprint2/b2v3_quick")

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]
LR           = 2e-4
WEIGHT_DECAY = 0.01

MAX_STEPS       = 50
MICRO_BS        = 1
GRAD_ACCUM      = 4       # smaller accum for quick test
TRAIN_SAMPLES   = 200
EVAL_SAMPLES    = 100
RESCORE_SAMPLES = 80      # small subset for cell loss estimation in quick test

# CVaR quick test params
CVAR_ALPHA       = 0.1
MULTIPLIER_CLIP  = 5.0
WARMUP_STEPS     = 10     # first 10 optimizer steps are warmup (multiplier=1.0)
RESCORE_EVERY_N  = 15     # rescore every 15 steps (ensure ≥2 rescores after warmup)

# Prompt templates
BINARY_PROMPT = 'Look at the image. Is the following spatial statement true or false?\n\nStatement: "{caption}"\n\nAnswer with ONLY "true" or "false".'
OPEN_PROMPT   = 'Look at the image carefully. Answer the following spatial reasoning question with a short answer.\n\nQuestion: {question}\n\nAnswer:'
SPATIAL_PROMPT = 'Look at the image carefully. Answer the following spatial reasoning question.\n\nQuestion: {question}\n\nChoose the correct answer from: {choices}\n\nAnswer with ONLY the letter or the exact answer text, nothing else.'


# ═══════════════════════════════════════════════════════════════════════
# Data helpers (shared with B2-v2 quick)
# ═══════════════════════════════════════════════════════════════════════
def load_samples(split, max_n=None):
    samples = []
    with open(SPLITS_DIR / f"{split}.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
            if max_n and len(samples) >= max_n:
                break
    return samples


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


# ═══════════════════════════════════════════════════════════════════════
# Cell loss estimation (quick version — small subset)
# ═══════════════════════════════════════════════════════════════════════
def estimate_cell_losses_quick(model, processor, process_vision_info,
                               samples, partition, max_samples=RESCORE_SAMPLES):
    """Quick cell loss estimation on a small subset."""
    model.eval()
    if len(samples) > max_samples:
        idx = np.random.choice(len(samples), max_samples, replace=False)
        subset = [samples[i] for i in idx]
    else:
        subset = samples

    cell_losses = defaultdict(list)
    for s in subset:
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

    cell_mean = {cid: float(np.mean(ls)) for cid, ls in cell_losses.items()}
    model.train()
    return cell_mean


def estimate_cell_losses_quick_detailed(model, processor, process_vision_info,
                                        samples, partition,
                                        max_samples=RESCORE_SAMPLES):
    """Quick cell loss estimation that also returns per-cell loss lists.

    Used by cluster_cvar to compute loss_std for clustering features.
    """
    model.eval()
    if len(samples) > max_samples:
        idx = np.random.choice(len(samples), max_samples, replace=False)
        subset = [samples[i] for i in idx]
    else:
        subset = samples

    cell_loss_lists = defaultdict(list)
    for s in subset:
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

    cell_mean = {cid: float(np.mean(ls)) for cid, ls in cell_loss_lists.items()}
    model.train()
    return cell_mean, dict(cell_loss_lists)


# ═══════════════════════════════════════════════════════════════════════
# CVaR Cell weighter (same logic as full version, with warmup tracking)
# ═══════════════════════════════════════════════════════════════════════
class CellCVaRWeighter:
    def __init__(self, partition, alpha=CVAR_ALPHA, clip=MULTIPLIER_CLIP):
        self.partition = partition
        self.alpha = alpha
        self.clip = clip
        self.cell_losses = {}
        self.eta = 0.0
        self.multipliers = {}
        self.is_warm = False
        self.update_count = 0

    def update(self, cell_losses):
        self.cell_losses = cell_losses
        if not cell_losses:
            return

        sorted_cells = sorted(cell_losses.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_cells)
        k = max(1, int(np.ceil(n * self.alpha)))

        self.eta = sorted_cells[k - 1][1] if k <= n else sorted_cells[-1][1]

        self.multipliers = {}
        for rank, (cid, loss) in enumerate(sorted_cells):
            if rank < k:
                self.multipliers[cid] = self.clip
            else:
                self.multipliers[cid] = 1.0

        self.is_warm = True
        self.update_count += 1
        n_active = sum(1 for m in self.multipliers.values() if m > 1.01)
        tail_ids = [cid for cid, m in self.multipliers.items() if m > 1.01]
        print(f"    [CVaR rescore #{self.update_count}] eta={self.eta:.4f}, "
              f"active={n_active}/{n}, k={k}, tail={tail_ids}")

    def get_multiplier(self, sample, step=None, warmup_steps=0):
        """Get gradient multiplier. Returns 1.0 during warmup phase."""
        if not self.is_warm:
            return 1.0
        if step is not None and step <= warmup_steps:
            return 1.0
        cid = self.partition.get_cell_by_features(sample)
        return self.multipliers.get(cid, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="B2-v3 Quick Test")
    parser.add_argument("--method", choices=[
        "cvar_cell", "global", "cluster_cvar", "lossgroup_cvar",
        "jtt_cell", "cell_only",
    ], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_groups", type=int, default=4,
                        help="Number of groups for cluster/lossgroup methods")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    is_cvar_method = args.method in ("cvar_cell", "cluster_cvar", "lossgroup_cvar")

    run_dir = OUT_DIR / f"quick_{args.method}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checks = {}  # track pass/fail for each validation

    print("=" * 60)
    print(f"B2-v3 QUICK TEST: {args.method}")
    print("=" * 60)

    # ── 1. Load model ──
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from qwen_vl_utils import process_vision_info

    print("\n[CHECK 1] Loading model with 4-bit + LoRA...")
    t0 = time.time()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    checks["model_loaded"] = True

    # ── 2. Load data ──
    print(f"\n[CHECK 2] Loading {TRAIN_SAMPLES} train + {EVAL_SAMPLES} eval samples...")
    all_train = load_samples("train")
    indices = np.random.choice(len(all_train), size=min(TRAIN_SAMPLES, len(all_train)),
                               replace=False)
    train_samples = [all_train[i] for i in indices]
    eval_samples = load_samples("repair_val", EVAL_SAMPLES)
    print(f"  Train: {len(train_samples)}, Eval: {len(eval_samples)}")
    checks["data_load"] = True

    # ── 3. Load Mondrian partition ──
    partition = MondrianPartition.load(B1_DIR / "partition.json")
    train_cell_map = {}
    for i, s in enumerate(train_samples):
        train_cell_map[i] = partition.get_cell_by_features(s)
    n_mapped = sum(1 for v in train_cell_map.values() if v)
    print(f"  Partition: {len(partition.cells)} cells, "
          f"{n_mapped}/{len(train_samples)} train mapped")

    # ── 4. Cell loss estimation ──
    cell_losses = None
    cell_loss_lists = None
    train_partition = partition    # default: CVaR targets Mondrian cells
    eval_partition = partition     # always Mondrian for eval
    group_partition = None

    if args.method != "global":
        print(f"\n[CHECK 3] Cell loss estimation ({RESCORE_SAMPLES} samples)...")
        if args.method == "cluster_cvar":
            cell_losses, cell_loss_lists = estimate_cell_losses_quick_detailed(
                model, processor, process_vision_info,
                eval_samples, partition, RESCORE_SAMPLES,
            )
        else:
            cell_losses = estimate_cell_losses_quick(
                model, processor, process_vision_info,
                eval_samples, partition, RESCORE_SAMPLES,
            )
        n_cells = len(cell_losses)
        print(f"  Estimated {n_cells} cells")
        if n_cells > 0:
            sorted_cl = sorted(cell_losses.values())
            print(f"  Loss range: [{sorted_cl[0]:.3f}, {sorted_cl[-1]:.3f}], "
                  f"median={np.median(sorted_cl):.3f}")
        checks["cell_estimation"] = n_cells > 0
    else:
        checks["cell_estimation"] = "N/A (global)"

    # ── 5. Group construction (cluster_cvar / lossgroup_cvar) ──
    checks["group_partition_built"] = "N/A"
    checks["n_groups"] = "N/A"
    checks["group_coverage"] = "N/A"

    if args.method == "cluster_cvar" and cell_losses:
        print(f"\n[CHECK 4a] Building cluster groups (n_groups={args.n_groups})...")
        group_partition = build_cluster_groups(
            partition, cell_losses,
            n_groups=args.n_groups,
            cell_loss_lists=cell_loss_lists,
        )
        train_partition = group_partition
        print(group_partition.summary())
        n_groups_actual = len(group_partition.groups)
        n_cells_covered = len(group_partition.cell_to_group)
        print(f"\n  Group composition:")
        for gid in sorted(group_partition.groups.keys()):
            members = group_partition.groups[gid]
            info = group_partition.group_info.get(gid, {})
            print(f"    {gid}: {len(members)} cells = {members}")
            if info:
                print(f"      mean_loss={info.get('mean_loss', '?'):.4f}, "
                      f"loss_range={info.get('loss_range', '?')}")
        checks["group_partition_built"] = True
        checks["n_groups"] = n_groups_actual
        checks["group_coverage"] = f"{n_cells_covered}/{len(cell_losses)} cells"

    elif args.method == "lossgroup_cvar" and cell_losses:
        print(f"\n[CHECK 4a] Building loss-bin groups (n_groups={args.n_groups})...")
        group_partition = build_lossgroup_groups(
            partition, cell_losses, n_groups=args.n_groups,
        )
        train_partition = group_partition
        print(group_partition.summary())
        n_groups_actual = len(group_partition.groups)
        n_cells_covered = len(group_partition.cell_to_group)
        print(f"\n  Group composition:")
        for gid in sorted(group_partition.groups.keys()):
            members = group_partition.groups[gid]
            info = group_partition.group_info.get(gid, {})
            print(f"    {gid}: {len(members)} cells = {members}")
            if info:
                print(f"      mean_loss={info.get('mean_loss', '?'):.4f}, "
                      f"loss_range={info.get('loss_range', '?')}")
        checks["group_partition_built"] = True
        checks["n_groups"] = n_groups_actual
        checks["group_coverage"] = f"{n_cells_covered}/{len(cell_losses)} cells"

    # ── 6. CVaR weighter init ──
    cvar_weighter = None
    warmup_rescore_done = not is_cvar_method
    rescore_count = 0
    active_tail_history = []  # track tail groups across rescores

    if is_cvar_method and cell_losses:
        print(f"\n[CHECK 4b] CVaR weighter initialization on train_partition...")
        cvar_weighter = CellCVaRWeighter(train_partition, CVAR_ALPHA, MULTIPLIER_CLIP)

        # Compute initial losses at the train_partition level
        if group_partition is not None:
            group_losses = {}
            for gid, members in group_partition.groups.items():
                member_losses = [cell_losses[c] for c in members if c in cell_losses]
                if member_losses:
                    group_losses[gid] = float(np.mean(member_losses))
            print(f"  Initial group-level losses: {group_losses}")
            cvar_weighter.update(group_losses)
        else:
            cvar_weighter.update(cell_losses)

        mults = list(cvar_weighter.multipliers.values())
        has_diversity = max(mults) > 1.5 if mults else False
        print(f"  Multiplier range: [{min(mults):.2f}, {max(mults):.2f}]")
        print(f"  eta = {cvar_weighter.eta:.4f}")
        print(f"  Has meaningful diversity (>1.5x): {has_diversity}")
        checks["cvar_initialized"] = cvar_weighter.is_warm
        checks["cvar_multiplier_diversity"] = has_diversity

        # Record initial tail
        tail = [cid for cid, m in cvar_weighter.multipliers.items() if m > 1.01]
        active_tail_history.append({"step": 0, "trigger": "init", "tail": tail})

    elif args.method == "cell_only" and cell_losses:
        # Minimal cell_only support (from b2v2_quick)
        sorted_cells = sorted(cell_losses.items(), key=lambda x: x[1], reverse=True)
        worst_cells = set(cid for cid, _ in sorted_cells[:5])
        n_up = sum(1 for i in range(len(train_samples))
                   if train_cell_map[i] in worst_cells)
        print(f"\n[CHECK 4] Cell-only: {n_up} samples in worst 5 cells upweighted")
        checks["cell_only_weights"] = n_up > 0

    # ── 7. Training loop ──
    print(f"\n[CHECK 5] Training for {MAX_STEPS} steps "
          f"(warmup={WARMUP_STEPS}, rescore_every={RESCORE_EVERY_N})...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()

    losses_log = []
    accum_loss = 0.0
    optimizer.zero_grad()
    total_errors = 0
    sample_probs = None  # uniform for all CVaR methods + global

    # cell_only / jtt_cell would set sample_probs here; omitted for quick simplicity

    for step_micro in range(1, MAX_STEPS * GRAD_ACCUM + 1):
        actual_step = step_micro // GRAD_ACCUM if step_micro % GRAD_ACCUM == 0 else None

        # -- CVaR: warmup-end rescore --
        if (is_cvar_method and cvar_weighter is not None
            and not warmup_rescore_done
            and actual_step is not None and actual_step > WARMUP_STEPS):
            warmup_rescore_done = True
            print(f"\n  [CVaR] Warm-start ended at step {actual_step}. "
                  f"Rescoring on train_partition...")
            new_losses = estimate_cell_losses_quick(
                model, processor, process_vision_info,
                eval_samples, train_partition, RESCORE_SAMPLES,
            )
            cvar_weighter.update(new_losses)
            rescore_count += 1
            tail = [cid for cid, m in cvar_weighter.multipliers.items() if m > 1.01]
            active_tail_history.append({
                "step": actual_step, "trigger": "warmup_end", "tail": tail,
            })
            print(f"  Warmup rescore done. Active tail: {tail}")
            model.train()

        # -- CVaR: periodic rescore --
        if (is_cvar_method and cvar_weighter is not None
            and warmup_rescore_done
            and actual_step is not None and actual_step > WARMUP_STEPS
            and actual_step % RESCORE_EVERY_N == 0):
            print(f"\n  [CVaR] Periodic rescore at step {actual_step}...")
            new_losses = estimate_cell_losses_quick(
                model, processor, process_vision_info,
                eval_samples, train_partition, RESCORE_SAMPLES,
            )
            cvar_weighter.update(new_losses)
            rescore_count += 1
            tail = [cid for cid, m in cvar_weighter.multipliers.items() if m > 1.01]
            active_tail_history.append({
                "step": actual_step, "trigger": "periodic", "tail": tail,
            })
            print(f"  Active tail: {tail}")
            model.train()

        # -- Sample training example --
        if sample_probs is not None:
            idx = np.random.choice(len(train_samples), p=sample_probs)
        else:
            idx = np.random.randint(len(train_samples))
        sample = train_samples[idx]

        try:
            inputs, labels = tokenize_train_example(processor, sample, process_vision_info)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            labels = labels.to(model.device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / GRAD_ACCUM

            # CVaR multiplier (with warmup awareness)
            if is_cvar_method and cvar_weighter is not None:
                mult = cvar_weighter.get_multiplier(
                    sample,
                    step=actual_step if actual_step else (step_micro // GRAD_ACCUM),
                    warmup_steps=WARMUP_STEPS,
                )
                loss = loss * mult

            loss.backward()
            accum_loss += outputs.loss.item()

        except Exception as e:
            total_errors += 1
            if total_errors <= 5:
                print(f"  error at micro_step {step_micro}: {e}")
            continue

        if step_micro % GRAD_ACCUM == 0:
            step = step_micro // GRAD_ACCUM
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            avg_l = accum_loss / GRAD_ACCUM
            losses_log.append({"step": step, "loss": avg_l})
            if step % 10 == 0 or step <= 5:
                extra = ""
                if cvar_weighter and cvar_weighter.is_warm:
                    extra = f"  eta={cvar_weighter.eta:.3f}"
                    if step <= WARMUP_STEPS:
                        extra += " [warmup: mult=1.0]"
                print(f"  step {step:3d}/{MAX_STEPS}: loss={avg_l:.4f}{extra}")
            accum_loss = 0.0

    # ── 8. Check loss trend ──
    decreased = None
    if len(losses_log) >= 10:
        first10 = np.median([l["loss"] for l in losses_log[:10]])
        last10 = np.median([l["loss"] for l in losses_log[-10:]])
        min_loss = min(l["loss"] for l in losses_log)
        decreased = bool(last10 < first10 or min_loss < first10 * 0.5)
        print(f"\n[CHECK 6] Loss: first10_med={first10:.4f} -> last10_med={last10:.4f} "
              f"(min={min_loss:.4f})  {'DECREASED' if decreased else 'NOT DECREASED'}")
        checks["loss_decreased"] = decreased

    # ── 9. CVaR rescore checks ──
    if is_cvar_method:
        checks["warmup_rescore_fired"] = warmup_rescore_done
        checks["periodic_rescore_count"] = rescore_count
        # Were tail groups ever activated?
        any_tail_seen = any(len(h["tail"]) > 0 for h in active_tail_history)
        checks["active_tail_groups_seen"] = any_tail_seen
        print(f"\n[CHECK 7] CVaR rescores: {rescore_count} total")
        print(f"  Warmup rescore fired: {warmup_rescore_done}")
        print(f"  Active tail groups ever seen: {any_tail_seen}")
        print(f"  Tail history:")
        for h in active_tail_history:
            print(f"    step={h['step']:3d} ({h['trigger']:12s}): tail={h['tail']}")

    # ── 10. Checkpoint save ──
    print(f"\n[CHECK 8] Saving checkpoint...")
    ckpt_dir = run_dir / "checkpoint"
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    ckpt_size = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file())
    print(f"  Size: {ckpt_size / 1e6:.1f} MB")
    checks["checkpoint_saved"] = ckpt_dir.exists()

    # ── 11. Quick eval ──
    print(f"\n[CHECK 9] Eval on {EVAL_SAMPLES} repair_val samples...")
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
        correct += is_correct
        total += 1

    acc = correct / max(total, 1)
    print(f"  Accuracy: {correct}/{total} = {acc:.3f}")
    checks["eval_ran"] = total > 0

    # ── 12. Summary ──
    summary = {
        "method": args.method,
        "seed": args.seed,
        "n_groups_requested": args.n_groups,
        "train_samples": TRAIN_SAMPLES,
        "eval_samples": total,
        "max_steps": MAX_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "rescore_every_n": RESCORE_EVERY_N,
        "losses": losses_log,
        "loss_decreased": decreased,
        "eval_accuracy": float(acc),
        "checkpoint_size_mb": float(ckpt_size / 1e6),
        "total_errors": total_errors,
        "checks": checks,
    }

    if is_cvar_method and cvar_weighter:
        summary["cvar_final_eta"] = cvar_weighter.eta
        summary["cvar_rescore_count"] = rescore_count
        summary["cvar_multiplier_range"] = [
            min(cvar_weighter.multipliers.values()) if cvar_weighter.multipliers else 1.0,
            max(cvar_weighter.multipliers.values()) if cvar_weighter.multipliers else 1.0,
        ]
        summary["active_tail_history"] = active_tail_history

    if group_partition is not None:
        summary["group_construction"] = {
            "type": args.method,
            "n_groups_actual": len(group_partition.groups),
            "composition": {
                gid: members
                for gid, members in group_partition.get_group_composition().items()
            },
            "group_info": {
                gid: {k: v for k, v in info.items() if k != "members"}
                for gid, info in group_partition.group_info.items()
            },
        }

    with open(run_dir / "quick_test_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Final report ──
    print(f"\n{'='*60}")
    print(f"B2-v3 QUICK TEST SUMMARY: {args.method}")
    print(f"{'='*60}")

    # Determine pass/fail for boolean checks only
    all_pass = True
    for check_name, val in checks.items():
        if isinstance(val, bool):
            status = "PASS" if val else "FAIL"
            if not val:
                all_pass = False
        elif isinstance(val, int):
            status = f"= {val}"
        elif isinstance(val, str):
            status = val
        else:
            status = str(val)
        print(f"  [{status:>6s}]  {check_name}")

    print(f"\n  Loss trend:  {'OK' if decreased else 'WARN'}")
    print(f"  Eval acc:    {acc:.3f}")
    print(f"  Errors:      {total_errors}")
    print(f"  Output:      {run_dir}")

    if all_pass:
        print(f"\n  ALL CHECKS PASSED. Ready for full B2-v3 run:")
        cmd = f"CUDA_VISIBLE_DEVICES=0 python scripts/run_b2v3.py --method {args.method} --seed 1"
        if args.method in ("cluster_cvar", "lossgroup_cvar"):
            cmd += f" --n_groups {args.n_groups}"
        print(f"   {cmd}")
    else:
        failed = [k for k, v in checks.items() if isinstance(v, bool) and not v]
        print(f"\n  FAILED CHECKS: {failed}")
        print(f"   Fix issues before running full B2-v3.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
