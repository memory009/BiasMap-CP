#!/usr/bin/env python3
"""
B4: Recalibration Guardrail — Post-FT Conformal Prediction recalibration.

Validates that fine-tuned models can recover >=90% marginal coverage via
fresh-split Mondrian CP recalibration.

Pipeline:
  Phase A: Inference on recal split (4,501 samples) → nc_scores + predictions
  Phase B: Inference on repair_val split (4,506 samples) → nc_scores + predictions
  Phase C: Mondrian CP calibration on recal nc_scores
  Phase D: Coverage evaluation on repair_val using recal-calibrated thresholds
  Phase E: Gate check & report

Gate (all must pass):
  - Marginal coverage >= 0.90 on repair_val
  - All cells: coverage >= 0.75 (worst-cell shortfall <= 15%)
  - Mean set size < 1.8 (reasonable for binary QA)

Usage:
  # Single checkpoint
  CUDA_VISIBLE_DEVICES=0 python scripts/run_b4_recalibration.py \\
      --method cvar_cell --seed 1

  # Skip inference if outputs already exist
  CUDA_VISIBLE_DEVICES=0 python scripts/run_b4_recalibration.py \\
      --method cvar_cell --seed 1 --skip_inference

  # Full sweep (sequential)
  for method in global lossgroup_cvar cvar_cell; do
    for seed in 1 2 3; do
      CUDA_VISIBLE_DEVICES=0 python scripts/run_b4_recalibration.py \\
          --method $method --seed $seed
    done
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
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.diagnosis.mondrian_partition import MondrianPartition
from src.evaluation.conformal import SplitCP, MondrianCP
from src.evaluation.metrics import compute_cvar

# ── Paths ─────────────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
CACHE_DIR  = "/LOCAL2/psqhe8/hf_cache"
SPLITS_DIR = Path("data/splits")
B1_DIR     = Path("results/sprint2/b1_diagnosis")
B2_DIR     = Path("results/sprint2/b2v3")
OUT_DIR    = Path("results/sprint3/b4_recalibration")

# ── CP config ────────────────────────────────────────────────────────
ALPHA = 0.1              # target coverage = 1 - alpha = 90%
MIN_CELL_SIZE = 30       # Mondrian CP minimum cell size for own threshold
MIN_CELL_SUPPORT = 20    # minimum cell support for reporting

# ── Prompt templates (same as B2-v3) ────────────────────────────────
BINARY_PROMPT = 'Look at the image. Is the following spatial statement true or false?\n\nStatement: "{caption}"\n\nAnswer with ONLY "true" or "false".'
OPEN_PROMPT   = 'Look at the image carefully. Answer the following spatial reasoning question with a short answer.\n\nQuestion: {question}\n\nAnswer:'
SPATIAL_PROMPT = 'Look at the image carefully. Answer the following spatial reasoning question.\n\nQuestion: {question}\n\nChoose the correct answer from: {choices}\n\nAnswer with ONLY the letter or the exact answer text, nothing else.'


# ═══════════════════════════════════════════════════════════════════════
# Helpers (copied from run_b2v3.py for standalone use)
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
        raise ValueError(f"Unsupported model_id: {model_id}")


def load_samples(split: str):
    samples = []
    with open(SPLITS_DIR / f"{split}.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
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


def load_image(path: str) -> Image.Image:
    if path and os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    return Image.new("RGB", (224, 224), (128, 128, 128))


# ═══════════════════════════════════════════════════════════════════════
# Phase A/B: Inference
# ═══════════════════════════════════════════════════════════════════════
def run_inference(model, processor, process_vision_info, samples, partition,
                  output_path, split_name="recal"):
    """Run generative inference and compute nc_scores via logit extraction.

    For each sample:
      1. Generate answer text (greedy)
      2. Forward pass to get logits for answer tokens
      3. nc_score = 1 - p(correct_answer_token)

    Returns list of result dicts.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    done_ids = set()
    existing = []
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.add(rec["sample_id"])
                    existing.append(rec)
                except Exception:
                    pass
        if done_ids:
            print(f"  Resuming {split_name}: {len(done_ids)} already done")

    remaining = [s for s in samples if s["id"] not in done_ids]
    if not remaining:
        print(f"  {split_name}: all {len(samples)} samples already done")
        return existing

    print(f"\n  Running inference on {split_name}: {len(remaining)} samples")
    model.eval()
    results = list(existing)
    t0 = time.time()

    with open(output_path, "a") as f_out:
        for i, s in enumerate(remaining):
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

                # Generate answer
                with torch.no_grad():
                    gen_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
                input_len = inputs["input_ids"].shape[1]
                response = processor.decode(
                    gen_ids[0][input_len:], skip_special_tokens=True,
                ).strip().lower()

                # Check correctness
                answer = s["answer"].lower().strip()
                is_correct = (
                    response == answer
                    or (answer in response)
                    or (response in answer and len(response) >= 2)
                )

                # Compute nc_score
                # For closed-form QA (explicit choices): logit-based softmax
                # For open-ended QA (no choices): correctness-based (0/1)
                raw_choices = s.get("choices")
                if raw_choices and len(raw_choices) >= 2:
                    # Logit-based nc_score over known choice set
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits[0, -1, :]

                    choice_tokens = {}
                    for c in raw_choices:
                        tids = processor.tokenizer.encode(c.lower(), add_special_tokens=False)
                        if tids:
                            choice_tokens[c.lower()] = tids[0]

                    if choice_tokens:
                        choice_logits = {c: float(logits[tid]) for c, tid in choice_tokens.items()}
                        max_logit = max(choice_logits.values())
                        exp_logits = {c: np.exp(v - max_logit) for c, v in choice_logits.items()}
                        total = sum(exp_logits.values())
                        probs = {c: exp_logits[c] / total for c in exp_logits}
                        p_correct = probs.get(answer, 0.5)
                        nc_score = 1.0 - p_correct
                    else:
                        nc_score = 0.0 if is_correct else 1.0
                        probs = {answer: 1.0 if is_correct else 0.0}
                else:
                    # Open-ended: use binary correctness as nc_score
                    # (consistent with Sprint 1 approach)
                    nc_score = 0.0 if is_correct else 1.0
                    probs = {answer: 1.0 if is_correct else 0.0}

            except Exception as e:
                response = ""
                is_correct = False
                nc_score = 1.0
                probs = {}
                if (i + 1) <= 5:
                    print(f"    Error on {s['id']}: {e}")

            cell_id = partition.get_cell_by_features(s)
            rec = {
                "sample_id": s["id"],
                "correct": is_correct,
                "response": response,
                "nc_score": float(nc_score),
                "probabilities": {k: float(v) for k, v in probs.items()},
                "cell_id": cell_id,
                "answer": s["answer"].lower().strip(),
            }
            f_out.write(json.dumps(rec) + "\n")
            results.append(rec)

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                running_acc = np.mean([r["correct"] for r in results])
                rate = (i + 1) / elapsed * 60
                eta = (len(remaining) - i - 1) / (rate / 60) if rate > 0 else 0
                print(f"    {split_name}: {i+1}/{len(remaining)} | "
                      f"acc={running_acc:.3f} | {rate:.0f} samples/min | "
                      f"ETA {eta/60:.1f}min")

    elapsed = time.time() - t0
    acc = np.mean([r["correct"] for r in results])
    print(f"  {split_name} done: {len(results)} samples, "
          f"acc={acc:.3f}, time={elapsed/60:.1f}min")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase C/D/E: CP Calibration + Evaluation + Gate
# ═══════════════════════════════════════════════════════════════════════
def run_cp_analysis(recal_results, repair_val_results, partition, alpha, run_dir):
    """Calibrate Mondrian CP on recal, evaluate coverage on repair_val."""
    run_dir = Path(run_dir)

    # --- Phase C: Calibrate on recal ---
    recal_nc = [r["nc_score"] for r in recal_results]
    recal_cells = [r["cell_id"] or "unknown" for r in recal_results]

    # Global CP
    global_cp = SplitCP(alpha)
    global_threshold = global_cp.calibrate(recal_nc)
    print(f"\n  Global CP threshold: {global_threshold:.4f}")

    # Mondrian CP
    mondrian_cp = MondrianCP(alpha, min_cell_size=MIN_CELL_SIZE)
    cell_thresholds = mondrian_cp.calibrate(recal_nc, recal_cells)
    print(f"  Mondrian CP: {len(cell_thresholds)} cell thresholds")

    # Save thresholds
    thresholds_data = {
        "alpha": alpha,
        "global_threshold": global_threshold,
        "cell_thresholds": cell_thresholds,
        "n_recal_samples": len(recal_results),
        "recal_accuracy": float(np.mean([r["correct"] for r in recal_results])),
    }
    with open(run_dir / "mondrian_thresholds.json", "w") as f:
        json.dump(thresholds_data, f, indent=2)

    # --- Phase D: Evaluate on repair_val ---
    test_nc = [r["nc_score"] for r in repair_val_results]
    test_cells = [r["cell_id"] or "unknown" for r in repair_val_results]
    test_probs = [r["probabilities"] for r in repair_val_results]

    # Global coverage
    marginal_coverage = global_cp.coverage(test_nc)
    global_mean_set_size = global_cp.mean_set_size(test_probs)
    print(f"\n  Global coverage on repair_val: {marginal_coverage:.4f}")
    print(f"  Global mean set size: {global_mean_set_size:.3f}")

    # Per-cell coverage
    per_cell_coverage = mondrian_cp.per_cell_coverage(test_nc, test_cells)
    per_cell_set_size = mondrian_cp.per_cell_set_size(test_probs, test_cells)

    # Per-cell error rate (for comparison with B2)
    cell_errors = defaultdict(list)
    for r in repair_val_results:
        if r["cell_id"]:
            cell_errors[r["cell_id"]].append(1.0 - float(r["correct"]))
    cell_mean_error = {cid: float(np.mean(errs)) for cid, errs in cell_errors.items()
                       if len(errs) >= MIN_CELL_SUPPORT}

    # Coverage shortfall per cell
    target_coverage = 1.0 - alpha
    coverage_shortfalls = {}
    for cell_id, cov in per_cell_coverage.items():
        shortfall = max(0.0, target_coverage - cov)
        coverage_shortfalls[cell_id] = {
            "coverage": float(cov),
            "shortfall": float(shortfall),
            "set_size": float(per_cell_set_size.get(cell_id, 0)),
            "error_rate": float(cell_mean_error.get(cell_id, -1)),
        }

    # Aggregate metrics
    all_coverages = list(per_cell_coverage.values())
    worst_cell_coverage = min(all_coverages) if all_coverages else 0.0
    worst_cell_id = min(per_cell_coverage, key=per_cell_coverage.get) if per_cell_coverage else "N/A"
    mean_cell_coverage = float(np.mean(all_coverages)) if all_coverages else 0.0
    cvar_coverage_gap = compute_cvar(
        [max(0, target_coverage - c) for c in all_coverages], alpha=0.1
    ) if all_coverages else 1.0

    # Mondrian coverage (weighted by cell size)
    mondrian_marginal = float(np.mean([r["nc_score"] <= mondrian_cp.cell_thresholds.get(
        r["cell_id"] or "unknown", mondrian_cp.global_threshold or 1.0
    ) for r in repair_val_results]))

    # --- Phase E: Gate check ---
    gate_marginal = marginal_coverage >= 0.90
    gate_worst_cell = worst_cell_coverage >= 0.75
    gate_set_size = global_mean_set_size < 1.8
    gate_passed = gate_marginal and gate_worst_cell and gate_set_size

    repair_val_acc = float(np.mean([r["correct"] for r in repair_val_results]))

    # W10%CVaR_err (same as B2 metric)
    if cell_mean_error:
        sorted_errs = sorted(cell_mean_error.values(), reverse=True)
        k = max(1, int(np.ceil(len(sorted_errs) * alpha)))
        w10_cvar_err = float(np.mean(sorted_errs[:k]))
    else:
        w10_cvar_err = 1.0

    results = {
        "timestamp": datetime.now().isoformat(),
        "alpha": alpha,
        # Accuracy metrics (for comparison with B2)
        "repair_val_accuracy": repair_val_acc,
        "w10_cvar_err": w10_cvar_err,
        "cell_mean_errors": cell_mean_error,
        # CP coverage metrics
        "global_threshold": global_threshold,
        "marginal_coverage": float(marginal_coverage),
        "mondrian_marginal_coverage": float(mondrian_marginal),
        "global_mean_set_size": float(global_mean_set_size),
        "mean_cell_coverage": mean_cell_coverage,
        "worst_cell_coverage": float(worst_cell_coverage),
        "worst_cell_id": worst_cell_id,
        "cvar_coverage_gap": float(cvar_coverage_gap),
        # Per-cell detail
        "per_cell_detail": coverage_shortfalls,
        "n_cells_evaluated": len(per_cell_coverage),
        # Gate
        "gate_marginal_coverage": gate_marginal,
        "gate_worst_cell": gate_worst_cell,
        "gate_set_size": gate_set_size,
        "gate_passed": gate_passed,
        # Sample counts
        "n_recal": len(recal_results),
        "n_repair_val": len(repair_val_results),
        "recal_accuracy": float(np.mean([r["correct"] for r in recal_results])),
    }

    with open(run_dir / "b4_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("B4 RECALIBRATION RESULTS")
    print(f"{'='*60}")
    print(f"  Recal split:       {len(recal_results)} samples, acc={results['recal_accuracy']:.3f}")
    print(f"  Repair_val split:  {len(repair_val_results)} samples, acc={repair_val_acc:.3f}")
    print(f"  W10%CVaR_err:      {w10_cvar_err:.4f}")
    print(f"\n  --- CP Coverage (target: {1-alpha:.0%}) ---")
    print(f"  Global marginal:   {marginal_coverage:.4f}  {'PASS' if gate_marginal else 'FAIL'}")
    print(f"  Mondrian marginal: {mondrian_marginal:.4f}")
    print(f"  Mean cell:         {mean_cell_coverage:.4f}")
    print(f"  Worst cell:        {worst_cell_coverage:.4f} ({worst_cell_id})  "
          f"{'PASS' if gate_worst_cell else 'FAIL'}")
    print(f"  CVaR coverage gap: {cvar_coverage_gap:.4f}")
    print(f"\n  --- Set Size ---")
    print(f"  Global mean:       {global_mean_set_size:.3f}  "
          f"{'PASS' if gate_set_size else 'FAIL'}")

    # Per-cell table (worst 10)
    sorted_cells = sorted(coverage_shortfalls.items(),
                          key=lambda x: x[1]["coverage"])
    print(f"\n  --- Worst 10 Cells by Coverage ---")
    print(f"  {'Cell':<35s} {'Coverage':>8s} {'Shortfall':>10s} {'SetSize':>8s} {'ErrRate':>8s}")
    for cid, info in sorted_cells[:10]:
        print(f"  {cid:<35s} {info['coverage']:8.4f} {info['shortfall']:10.4f} "
              f"{info['set_size']:8.3f} {info['error_rate']:8.3f}")

    print(f"\n{'='*60}")
    print(f"  GATE: {'PASSED' if gate_passed else 'FAILED'}")
    if not gate_marginal:
        print(f"    FAIL: marginal coverage {marginal_coverage:.4f} < 0.90")
    if not gate_worst_cell:
        print(f"    FAIL: worst cell coverage {worst_cell_coverage:.4f} < 0.75")
    if not gate_set_size:
        print(f"    FAIL: mean set size {global_mean_set_size:.3f} >= 1.8")
    print(f"{'='*60}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="B4: Recalibration Guardrail")
    parser.add_argument("--method", choices=["global", "lossgroup_cvar", "cvar_cell"],
                        required=True)
    parser.add_argument("--seed", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip inference, use existing output files for CP analysis")
    args = parser.parse_args()

    # Model tag for directory naming
    model_tag = args.model_id.split("/")[-1].lower().replace("-instruct", "").replace("-", "")
    run_name = f"{args.method}_seed{args.seed}"
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint path
    ckpt_dir = B2_DIR / f"b2v3_{args.method}_seed{args.seed}_{model_tag}" / "checkpoint-best"
    if not ckpt_dir.exists():
        print(f"ERROR: checkpoint not found: {ckpt_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"B4 Recalibration: {args.method} seed{args.seed}")
    print(f"{'='*60}")
    print(f"  Checkpoint: {ckpt_dir}")
    print(f"  Output:     {run_dir}")
    print(f"  Alpha:      {args.alpha}")

    # Load partition
    partition_path = B1_DIR / "partition.json"
    if not partition_path.exists():
        print(f"ERROR: B1 partition not found: {partition_path}")
        sys.exit(1)
    partition = MondrianPartition.load(partition_path)
    print(f"  Partition:  {len(partition.cells)} cells")

    # Load splits
    recal_samples = load_samples("recal")
    repair_val_samples = load_samples("repair_val")
    print(f"  Recal:      {len(recal_samples)} samples")
    print(f"  Repair_val: {len(repair_val_samples)} samples")

    recal_output_path = run_dir / "recal_outputs.jsonl"
    repair_val_output_path = run_dir / "repair_val_outputs.jsonl"

    if args.skip_inference:
        # Load existing outputs
        print("\n  Skipping inference, loading existing outputs...")
        recal_results = []
        with open(recal_output_path) as f:
            for line in f:
                recal_results.append(json.loads(line))
        repair_val_results = []
        with open(repair_val_output_path) as f:
            for line in f:
                repair_val_results.append(json.loads(line))
        print(f"  Loaded: {len(recal_results)} recal, {len(repair_val_results)} repair_val")
    else:
        # Load model
        print("\n  Loading model...")
        from transformers import BitsAndBytesConfig
        from peft import PeftModel

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        base_model = load_vlm_model(
            args.model_id, cache_dir=CACHE_DIR,
            quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(ckpt_dir))
        model.eval()

        processor = None
        process_vision_info = None
        if "Qwen2-VL" in args.model_id:
            from transformers import Qwen2VLProcessor
            from qwen_vl_utils import process_vision_info as pvi
            processor = Qwen2VLProcessor.from_pretrained(
                args.model_id, cache_dir=CACHE_DIR, trust_remote_code=True,
            )
            process_vision_info = pvi
        elif "Qwen3-VL" in args.model_id:
            from transformers import Qwen3VLProcessor
            from qwen_vl_utils import process_vision_info as pvi
            processor = Qwen3VLProcessor.from_pretrained(
                args.model_id, cache_dir=CACHE_DIR, trust_remote_code=True,
            )
            process_vision_info = pvi
        else:
            raise ValueError(f"Unsupported model: {args.model_id}")

        print(f"  Model loaded: {args.model_id} + LoRA from {ckpt_dir.name}")

        # Phase A: Inference on recal
        print(f"\n{'='*60}")
        print("Phase A: Inference on recal split")
        print(f"{'='*60}")
        recal_results = run_inference(
            model, processor, process_vision_info,
            recal_samples, partition,
            recal_output_path, split_name="recal",
        )

        # Phase B: Inference on repair_val
        print(f"\n{'='*60}")
        print("Phase B: Inference on repair_val split")
        print(f"{'='*60}")
        repair_val_results = run_inference(
            model, processor, process_vision_info,
            repair_val_samples, partition,
            repair_val_output_path, split_name="repair_val",
        )

    # Phase C/D/E: CP analysis + gate
    print(f"\n{'='*60}")
    print("Phase C/D/E: CP Calibration + Evaluation + Gate Check")
    print(f"{'='*60}")
    results = run_cp_analysis(
        recal_results, repair_val_results, partition, args.alpha, run_dir,
    )

    # Save run config
    config = {
        "method": args.method,
        "seed": args.seed,
        "model_id": args.model_id,
        "checkpoint": str(ckpt_dir),
        "alpha": args.alpha,
        "timestamp": datetime.now().isoformat(),
    }
    results["config"] = config
    with open(run_dir / "b4_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
