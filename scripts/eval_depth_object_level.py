#!/usr/bin/env python3
"""
Post-hoc generation-based evaluation for depth_object_level pilot checkpoint.

Loads the best LoRA checkpoint and runs generation-based accuracy eval
(same as evaluate_accuracy in run_b2v3.py) to produce error-rate metrics
that are directly comparable to B2v3 baselines.

No training needed — inference only.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/eval_depth_object_level.py
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.diagnosis.mondrian_partition import MondrianPartition

# -- Config -------------------------------------------------------------------
CACHE_DIR    = "/LOCAL2/psqhe8/hf_cache"
SPLITS_DIR   = Path("data/splits")
B1_DIR       = Path("results/sprint2/b1_diagnosis")
OUT_DIR      = Path("results/sprint2/pilots")

WORST_CELLS = [
    "in_front_of|True|gqa",
    "inside|False|gqa",
    "under|False|gqa",
    "behind|True|gqa",
]

MIN_CELL_SUPPORT = 20

# Prompt templates (same as pilot)
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


def evaluate_accuracy(model, processor, process_vision_info, eval_samples,
                      partition, alpha=0.1):
    """Generation-based eval: accuracy + worst-cell CVaR(error).
    Same logic as run_b2v3.py evaluate_accuracy."""
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
        except Exception as e:
            response = ""
            is_correct = False
            if i < 5:
                print(f"  Error at sample {i}: {e}")

        cid = partition.get_cell_by_features(s)
        results.append({"correct": is_correct, "cell_id": cid, "response": response,
                        "answer": s["answer"].lower().strip()})

        if (i + 1) % 100 == 0:
            running_acc = np.mean([r["correct"] for r in results])
            print(f"  gen eval: {i+1}/{len(eval_samples)}, running acc={running_acc:.3f}")

    overall_acc = float(np.mean([r["correct"] for r in results]))

    cell_errors = defaultdict(list)
    for r in results:
        if r["cell_id"] is not None:
            cell_errors[r["cell_id"]].append(1.0 - float(r["correct"]))

    cell_mean_error = {cid: float(np.mean(errs)) for cid, errs in cell_errors.items()
                       if len(errs) >= MIN_CELL_SUPPORT}

    def cvar_from(cell_mean, alpha):
        if cell_mean:
            sorted_vals = sorted(cell_mean.values(), reverse=True)
            k = max(1, int(np.ceil(len(sorted_vals) * alpha)))
            return float(np.mean(sorted_vals[:k]))
        return 1.0

    worst_cell_err = float(max(cell_mean_error.values())) if cell_mean_error else 1.0

    return {
        "overall_accuracy": overall_acc,
        "worst_10pct_cvar_error": cvar_from(cell_mean_error, alpha),
        "worst_cell_error": worst_cell_err,
        "n_cells_evaluated": len(cell_mean_error),
        "n_evaluated": len(results),
        "cell_errors": cell_mean_error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint dir (auto-detected if not set)")
    args = parser.parse_args()

    # Auto-detect checkpoint path
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        model_tag = args.model_id.split("/")[-1].lower().replace("-instruct", "").replace("-", "")
        ckpt_path = OUT_DIR / f"pilot_depth_object_level_seed{args.seed}_{model_tag}" / "checkpoint-best"

    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Model: {args.model_id}")

    # Load eval samples
    eval_samples = load_worst_cell_samples("repair_val")
    print(f"Eval samples: {len(eval_samples)}")

    # Load model
    from transformers import AutoProcessor, BitsAndBytesConfig
    from transformers import Qwen3VLForConditionalGeneration
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=CACHE_DIR)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, cache_dir=CACHE_DIR, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, str(ckpt_path))
    model.eval()
    print("Model loaded with LoRA checkpoint.")

    partition = MondrianPartition.load(B1_DIR / "partition.json")

    # Run generation-based eval
    print("\nRunning generation-based accuracy evaluation...")
    results = evaluate_accuracy(model, processor, process_vision_info,
                                eval_samples, partition)

    print(f"\n{'=' * 60}")
    print("GENERATION-BASED EVAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  Overall accuracy:     {results['overall_accuracy']:.4f}")
    print(f"  Worst cell error:     {results['worst_cell_error']:.4f}")
    print(f"  W10% CVaR(error):     {results['worst_10pct_cvar_error']:.4f}")
    print(f"  Cells evaluated:      {results['n_cells_evaluated']}")
    print(f"  Samples evaluated:    {results['n_evaluated']}")

    print(f"\n  Per-cell errors:")
    for cid, err in sorted(results["cell_errors"].items()):
        print(f"    {cid}: {err:.4f}")

    # B2v3 8B baselines (error rate, from generation-based eval)
    print(f"\n{'=' * 60}")
    print("COMPARISON WITH B2V3 BASELINES (error rate)")
    print(f"{'=' * 60}")
    baselines = {
        "Global FT 8B s1":  {"worst_cell_error": 0.3683, "W10%CVaR_error": 0.2800},
        "CVaR-cell 8B s1":  {"worst_cell_error": 0.3492, "W10%CVaR_error": 0.2577},
        "Depth obj-lvl 8B": {"worst_cell_error": results["worst_cell_error"],
                              "W10%CVaR_error": results["worst_10pct_cvar_error"]},
    }
    print(f"  {'Method':<22s} {'worst_cell_err':>15s} {'W10%CVaR_err':>15s}")
    print(f"  {'-' * 52}")
    for name, vals in baselines.items():
        print(f"  {name:<22s} {vals['worst_cell_error']:>15.4f} {vals['W10%CVaR_error']:>15.4f}")

    # Save results
    out_file = ckpt_path.parent / "eval_accuracy_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
