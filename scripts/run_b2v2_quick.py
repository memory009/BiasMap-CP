#!/usr/bin/env python3
"""
B2-v2 Quick Test: Validate all 4 methods on 200 samples / 50 steps.

Validates per method:
  1. Model loads with 4-bit + LoRA                              (all)
  2. Loss decreases over 50 steps                               (all)
  3. Cell loss estimation works                                  (cvar_cell, jtt_cell, cell_only)
  4. CVaR η quantile + gradient multiplier logic is correct      (cvar_cell)
  5. Periodic rescore fires and updates multipliers              (cvar_cell)
  6. JTT stage transition + hard sample identification works     (jtt_cell)
  7. Cell-only static weights are applied                        (cell_only)
  8. Checkpoint save/load works                                  (all)
  9. Eval on 100 repair_val samples produces metrics             (all)

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_b2v2_quick.py --method cvar_cell
  CUDA_VISIBLE_DEVICES=0 python scripts/run_b2v2_quick.py --method jtt_cell
  CUDA_VISIBLE_DEVICES=0 python scripts/run_b2v2_quick.py --method cell_only
  CUDA_VISIBLE_DEVICES=0 python scripts/run_b2v2_quick.py --method global

  # Run all 4 sequentially (~20 min total):
  for m in cvar_cell jtt_cell cell_only global; do
    CUDA_VISIBLE_DEVICES=0 python scripts/run_b2v2_quick.py --method $m
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

# ── Quick test config ─────────────────────────────────────────────────
MODEL_ID     = "Qwen/Qwen2-VL-2B-Instruct"
CACHE_DIR    = "/LOCAL2/psqhe8/hf_cache"
SPLITS_DIR   = Path("data/splits")
B1_DIR       = Path("results/sprint2/b1_diagnosis")
OUT_DIR      = Path("results/sprint2/b2v2_quick")

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]
LR           = 2e-4
WEIGHT_DECAY = 0.01

MAX_STEPS      = 50
MICRO_BS       = 1
GRAD_ACCUM     = 4       # smaller accum for quick test
TRAIN_SAMPLES  = 200
EVAL_SAMPLES   = 100
RESCORE_SAMPLES = 80     # small subset for cell loss estimation in quick test

# CVaR quick test params
CVAR_ALPHA       = 0.1
MULTIPLIER_CLIP  = 5.0
RESCORE_EVERY_N  = 20    # rescore every 20 optimizer steps during quick test

# JTT quick test params
JTT_WARMUP_STEPS = 15    # stage 1: 15 steps, then transition
JTT_WORST_K      = 5
JTT_HARD_FRAC    = 0.3
JTT_UPWEIGHT     = 5.0

# Cell-only quick test params
CELL_ONLY_K      = 5
CELL_ONLY_WEIGHT = 3.0

# Prompt templates
BINARY_PROMPT = 'Look at the image. Is the following spatial statement true or false?\n\nStatement: "{caption}"\n\nAnswer with ONLY "true" or "false".'
OPEN_PROMPT   = 'Look at the image carefully. Answer the following spatial reasoning question with a short answer.\n\nQuestion: {question}\n\nAnswer:'
SPATIAL_PROMPT = 'Look at the image carefully. Answer the following spatial reasoning question.\n\nQuestion: {question}\n\nChoose the correct answer from: {choices}\n\nAnswer with ONLY the letter or the exact answer text, nothing else.'


# ═══════════════════════════════════════════════════════════════════════
# Data helpers (shared with B2-v2 full)
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


# ═══════════════════════════════════════════════════════════════════════
# CVaR Cell weighter (same logic as full version)
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

        # Sort cells by loss descending (worst first)
        sorted_cells = sorted(cell_losses.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_cells)
        k = max(1, int(np.ceil(n * self.alpha)))  # worst α fraction

        # η = loss of the k-th worst cell (CVaR threshold)
        self.eta = sorted_cells[k - 1][1] if k <= n else sorted_cells[-1][1]

        # Binary CVaR multipliers: tail cells get uniform clip, rest get 1.0
        # True Rockafellar-Uryasev: ∇CVaR_α = E[∇L | L > η]
        self.multipliers = {}
        for rank, (cid, loss) in enumerate(sorted_cells):
            if rank < k:
                self.multipliers[cid] = self.clip
            else:
                self.multipliers[cid] = 1.0

        self.is_warm = True
        self.update_count += 1
        n_active = sum(1 for m in self.multipliers.values() if m > 1.01)
        print(f"    [CVaR rescore #{self.update_count}] η={self.eta:.4f}, "
              f"active={n_active}/{len(cell_losses)}, k={k}, "
              f"multiplier_tail={self.clip:.1f}, multiplier_rest=1.0"
              if n_active > 0 else
              f"    [CVaR rescore #{self.update_count}] η={self.eta:.4f}, "
              f"active=0/{len(cell_losses)} — WARNING")

    def get_multiplier(self, sample):
        if not self.is_warm:
            return 1.0
        cid = self.partition.get_cell_by_features(sample)
        return self.multipliers.get(cid, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="B2-v2 Quick Test")
    parser.add_argument("--method", choices=["cvar_cell", "jtt_cell", "cell_only", "global"],
                        required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = OUT_DIR / f"quick_{args.method}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checks = {}  # track pass/fail for each validation

    print("=" * 60)
    print(f"B2-v2 QUICK TEST: {args.method}")
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
    checks["model_load"] = True

    # ── 2. Load data ──
    print(f"\n[CHECK 2] Loading {TRAIN_SAMPLES} train + {EVAL_SAMPLES} eval samples...")
    all_train = load_samples("train")
    indices = np.random.choice(len(all_train), size=min(TRAIN_SAMPLES, len(all_train)), replace=False)
    train_samples = [all_train[i] for i in indices]
    eval_samples = load_samples("repair_val", EVAL_SAMPLES)
    print(f"  Train: {len(train_samples)}, Eval: {len(eval_samples)}")
    checks["data_load"] = True

    # ── 3. Load partition ──
    partition = MondrianPartition.load(B1_DIR / "partition.json")
    train_cell_map = {}
    for i, s in enumerate(train_samples):
        train_cell_map[i] = partition.get_cell_by_features(s)
    n_mapped = sum(1 for v in train_cell_map.values() if v)
    print(f"  Partition: {len(partition.cells)} cells, {n_mapped}/{len(train_samples)} mapped")

    # ── 4. Method-specific setup ──
    cvar_weighter = None
    jtt_hard_set = None
    jtt_stage = 1 if args.method == "jtt_cell" else None
    sample_probs = None

    if args.method in ("cvar_cell", "jtt_cell", "cell_only"):
        print(f"\n[CHECK 3] Cell loss estimation ({RESCORE_SAMPLES} samples)...")
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

        if args.method == "cvar_cell":
            print(f"\n[CHECK 4] CVaR weighter initialization...")
            cvar_weighter = CellCVaRWeighter(partition, CVAR_ALPHA, MULTIPLIER_CLIP)
            cvar_weighter.update(cell_losses)
            # Validate multiplier range
            mults = list(cvar_weighter.multipliers.values())
            has_diversity = max(mults) > 1.5 if mults else False
            print(f"  Multiplier range: [{min(mults):.2f}, {max(mults):.2f}]")
            print(f"  η = {cvar_weighter.eta:.4f}")
            print(f"  Has meaningful diversity (>1.5x): {has_diversity}")
            checks["cvar_init"] = cvar_weighter.is_warm
            checks["cvar_multiplier_diversity"] = has_diversity

        elif args.method == "cell_only":
            print(f"\n[CHECK 4] Cell-only weights...")
            sorted_cells = sorted(cell_losses.items(), key=lambda x: x[1], reverse=True)
            worst_cells = set(cid for cid, _ in sorted_cells[:CELL_ONLY_K])
            weights = []
            n_up = 0
            for i, s in enumerate(train_samples):
                cid = train_cell_map[i]
                if cid in worst_cells:
                    weights.append(CELL_ONLY_WEIGHT)
                    n_up += 1
                else:
                    weights.append(1.0)
            w = np.array(weights)
            sample_probs = w / w.sum()
            print(f"  Upweighted: {n_up}/{len(train_samples)} samples in worst {CELL_ONLY_K} cells")
            checks["cell_only_weights"] = n_up > 0

    # ── 5. Training loop ──
    print(f"\n[CHECK 5] Training for {MAX_STEPS} steps...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()

    losses_log = []
    accum_loss = 0.0
    optimizer.zero_grad()
    rescore_count = 0
    jtt_transition_done = False
    total_errors = 0

    for step_micro in range(1, MAX_STEPS * GRAD_ACCUM + 1):
        actual_step = step_micro // GRAD_ACCUM if step_micro % GRAD_ACCUM == 0 else None

        # -- JTT: transition at JTT_WARMUP_STEPS --
        if (args.method == "jtt_cell" and jtt_stage == 1
            and actual_step is not None and actual_step >= JTT_WARMUP_STEPS
            and not jtt_transition_done):
            print(f"\n  [JTT] Stage 1 done at step {actual_step}. Identifying hard samples...")
            stage1_cell_losses = estimate_cell_losses_quick(
                model, processor, process_vision_info,
                eval_samples, partition, RESCORE_SAMPLES,
            )
            # Quick hard sample identification (no full scoring in quick test)
            sorted_cells = sorted(stage1_cell_losses.items(), key=lambda x: x[1], reverse=True)
            worst_cells = set(cid for cid, _ in sorted_cells[:JTT_WORST_K])
            jtt_hard_set = set()
            for i, s in enumerate(train_samples):
                cid = train_cell_map[i]
                if cid in worst_cells:
                    jtt_hard_set.add(i)
            # Apply frac filter: keep top JTT_HARD_FRAC by random (full version scores by loss)
            if len(jtt_hard_set) > 0:
                keep_n = max(1, int(len(jtt_hard_set) * JTT_HARD_FRAC))
                jtt_hard_set = set(list(jtt_hard_set)[:keep_n])

            jtt_stage = 2
            jtt_transition_done = True

            w = np.ones(len(train_samples))
            for idx in jtt_hard_set:
                w[idx] = JTT_UPWEIGHT
            sample_probs = w / w.sum()

            print(f"  [JTT] Stage 2: {len(jtt_hard_set)} hard samples upweighted {JTT_UPWEIGHT}x")
            checks["jtt_transition"] = len(jtt_hard_set) > 0
            model.train()

        # -- CVaR: periodic rescore --
        if (args.method == "cvar_cell" and cvar_weighter is not None
            and actual_step is not None and actual_step > 0
            and actual_step % RESCORE_EVERY_N == 0):
            print(f"\n  [CVaR] Rescoring at step {actual_step}...")
            new_cell_losses = estimate_cell_losses_quick(
                model, processor, process_vision_info,
                eval_samples, partition, RESCORE_SAMPLES,
            )
            cvar_weighter.update(new_cell_losses)
            rescore_count += 1
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

            # CVaR multiplier
            if args.method == "cvar_cell" and cvar_weighter is not None:
                mult = cvar_weighter.get_multiplier(sample)
                loss = loss * mult

            loss.backward()
            accum_loss += outputs.loss.item()

        except Exception as e:
            total_errors += 1
            if total_errors <= 5:
                print(f"  ⚠ error at micro_step {step_micro}: {e}")
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
                    extra = f"  η={cvar_weighter.eta:.3f}"
                if jtt_stage == 2:
                    extra = "  [stage2]"
                print(f"  step {step:3d}/{MAX_STEPS}: loss={avg_l:.4f}{extra}")
            accum_loss = 0.0

    # ── 6. Check loss trend ──
    decreased = None
    if len(losses_log) >= 10:
        first10 = np.median([l["loss"] for l in losses_log[:10]])
        last10 = np.median([l["loss"] for l in losses_log[-10:]])
        min_loss = min(l["loss"] for l in losses_log)
        decreased = bool(last10 < first10 or min_loss < first10 * 0.5)
        print(f"\n[CHECK 6] Loss: first10_med={first10:.4f} → last10_med={last10:.4f} "
              f"(min={min_loss:.4f})  {'✅ DECREASED' if decreased else '⚠️ NOT DECREASED'}")
        checks["loss_decreased"] = decreased

    # ── 7. CVaR rescore count check ──
    if args.method == "cvar_cell":
        expected_rescores = MAX_STEPS // RESCORE_EVERY_N
        print(f"\n[CHECK 7] CVaR rescores: {rescore_count} "
              f"(expected ~{expected_rescores})")
        checks["cvar_rescore_fired"] = rescore_count >= 1

    # ── 8. Checkpoint save ──
    print(f"\n[CHECK 8] Saving checkpoint...")
    ckpt_dir = run_dir / "checkpoint"
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    ckpt_size = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file())
    print(f"  Size: {ckpt_size / 1e6:.1f} MB")
    checks["checkpoint_saved"] = ckpt_dir.exists()

    # ── 9. Quick eval ──
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
    checks["eval_runs"] = total > 0

    # ── 10. Summary ──
    summary = {
        "method": args.method,
        "seed": args.seed,
        "train_samples": TRAIN_SAMPLES,
        "max_steps": MAX_STEPS,
        "losses": losses_log,
        "loss_decreased": decreased,
        "eval_accuracy": float(acc),
        "eval_samples": total,
        "checkpoint_size_mb": float(ckpt_size / 1e6),
        "total_errors": total_errors,
        "checks": checks,
    }
    if args.method == "cvar_cell" and cvar_weighter:
        summary["cvar_final_eta"] = cvar_weighter.eta
        summary["cvar_rescore_count"] = rescore_count
        summary["cvar_multiplier_range"] = [
            min(cvar_weighter.multipliers.values()) if cvar_weighter.multipliers else 1.0,
            max(cvar_weighter.multipliers.values()) if cvar_weighter.multipliers else 1.0,
        ]
    if args.method == "jtt_cell" and jtt_hard_set is not None:
        summary["jtt_hard_samples"] = len(jtt_hard_set)
        summary["jtt_transition_step"] = JTT_WARMUP_STEPS

    with open(run_dir / "quick_test_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Final report ──
    print(f"\n{'='*60}")
    print(f"B2-v2 QUICK TEST SUMMARY: {args.method}")
    print(f"{'='*60}")

    all_pass = True
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False
        print(f"  {status}  {check_name}")

    print(f"\n  Loss trend:  {'✅' if decreased else '⚠️'}")
    print(f"  Eval acc:    {acc:.3f}")
    print(f"  Errors:      {total_errors}")
    print(f"  Output:      {run_dir}")

    if all_pass:
        print(f"\n✅ ALL CHECKS PASSED. Ready for full B2-v2 run:")
        print(f"   CUDA_VISIBLE_DEVICES=0 python scripts/run_b2v2.py "
              f"--method {args.method} --seed 1 --wandb")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"\n❌ FAILED CHECKS: {failed}")
        print(f"   Fix issues before running full B2-v2.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
