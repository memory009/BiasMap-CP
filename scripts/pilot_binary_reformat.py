#!/usr/bin/env python3
"""
Phase 2 Pilot: Binary Reformulation for Worst Cells

Key diagnostic finding: same spatial relations have 5-16x performance gap
between VSR (binary true/false) and GQA (open-ended). This pilot tests
whether reformulating GQA open-ended questions as binary verification
questions improves worst-cell accuracy.

Strategy:
  1. standard     — Original open-ended question (baseline)
  2. binary_reformat — Convert "What is behind X?" → "Is Y behind X? true/false"
     For each sample, generate the answer first with standard, then verify
     with a binary question.

This tests a core hypothesis: the model has spatial CAPABILITY but fails
at the open-ended EXPRESSION format. If binary verification corrects errors,
it supports data-level repair (reformatting training data) and inference-time
repair (verify-then-answer pipeline).

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/pilot_binary_reformat.py
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
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.diagnosis.mondrian_partition import MondrianPartition

# ── Config (same as pilot_inference_repair.py) ────────────────────────
MODEL_ID   = "Qwen/Qwen3-VL-2B-Instruct"
CACHE_DIR  = "/LOCAL2/psqhe8/hf_cache"
CHECKPOINT = Path("results/sprint2/b2v3/b2v3_global_seed1_qwen3vl2b/checkpoint-best")
SPLITS_DIR = Path("data/splits")
B1_DIR     = Path("results/sprint2/b1_diagnosis")
OUT_DIR    = Path("results/sprint2/pilots")

WORST_CELLS = [
    "in_front_of|True|gqa",
    "inside|False|gqa",
    "under|False|gqa",
    "behind|True|gqa",
]

# Prompts
OPEN_PROMPT = (
    "Look at the image carefully. Answer the following spatial reasoning "
    "question with a short answer.\n\nQuestion: {question}\n\nAnswer:"
)

BINARY_VERIFY_PROMPT = (
    'Look at the image. Is the following spatial statement true or false?\n\n'
    'Statement: "{statement}"\n\n'
    'Answer with ONLY "true" or "false".'
)


def load_image(path: str) -> Image.Image:
    if path and os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    return Image.new("RGB", (224, 224), (128, 128, 128))


def load_worst_cell_samples(split="repair_val", max_n=None):
    samples = []
    with open(SPLITS_DIR / f"{split}.jsonl") as f:
        for line in f:
            s = json.loads(line)
            da = str(s.get("depth_ambiguity", False))
            cid = f"{s['relation_type']}|{da}|{s['dataset']}"
            if cid in WORST_CELLS:
                s["_cell_id"] = cid
                samples.append(s)
                if max_n and len(samples) >= max_n:
                    break
    return samples


def check_correct(response: str, answer: str) -> bool:
    response = response.strip().lower()
    answer = answer.strip().lower()
    return (
        response == answer
        or answer in response
        or (response in answer and len(response) >= 2)
    )


def run_inference(model, processor, process_vision_info, sample, prompt_text,
                  max_new_tokens=16):
    image = load_image(sample["image_path"])
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt_text},
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
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(gen_ids[0][input_len:], skip_special_tokens=True)
    return response.strip()


def make_binary_statement(question: str, answer: str, relation: str) -> str:
    """Convert open-ended GQA question + answer into a binary spatial statement.

    Examples:
      Q: "Who is in front of the tree?" A: "woman"
      → "The woman is in front of the tree."

      Q: "What is the bus driver sitting inside of?" A: "bus"
      → "The bus driver is sitting inside of the bus."
    """
    q = question.lower().strip().rstrip("?").strip()
    a = answer.lower().strip()

    # Try common GQA patterns
    # "What/Who is [relation] [object]?" → "The [answer] is [relation] [object]"
    # "What is [preposition] the [object]?" → "The [answer] is [preposition] the [object]"

    # Simple replacement: replace the WH-word with the answer
    for wh in ["what", "who", "which kind of animal", "which animal",
               "what type of vehicle", "what kind of", "what type of",
               "which", "what"]:
        if q.startswith(wh):
            remainder = q[len(wh):].strip()
            if remainder.startswith("is ") or remainder.startswith("are "):
                # "What is behind the tree" → "The [answer] is behind the tree"
                return f"The {a} {remainder}."
            elif remainder.startswith("does ") or remainder.startswith("do "):
                # Harder, skip
                pass

    # Fallback: just create "The [answer] is [relation] ..."
    # Try to extract the reference object from the question
    return f"The {a} is {relation} mentioned in the image."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split", default="repair_val")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=" * 60)
    print(f"Phase 2 Pilot: Binary Reformulation")
    print(f"=" * 60)

    # Load model
    print("\nLoading model + LoRA checkpoint...")
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, str(CHECKPOINT))
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    print(f"Model loaded on {model.device}")

    partition = MondrianPartition.load(B1_DIR / "partition.json")
    all_samples = load_worst_cell_samples(args.split, max_n=None)

    if args.max_samples:
        cell_counts = Counter()
        limited = []
        for s in all_samples:
            if cell_counts[s["_cell_id"]] < args.max_samples:
                limited.append(s)
                cell_counts[s["_cell_id"]] += 1
        all_samples = limited

    print(f"Samples: {len(all_samples)}")
    cell_dist = Counter(s["_cell_id"] for s in all_samples)
    for c, n in sorted(cell_dist.items()):
        print(f"  {c}: {n}")

    # ── Run both strategies ───────────────────────────────────────────
    results_standard = defaultdict(list)
    results_binary = defaultdict(list)
    detail_log = []

    t0 = time.time()
    for i, s in enumerate(all_samples):
        cid = s["_cell_id"]
        answer = s["answer"].lower().strip()
        relation = s["relation_type"]

        # Strategy 1: Standard
        try:
            std_resp = run_inference(
                model, processor, process_vision_info, s,
                OPEN_PROMPT.format(question=s["question"]),
            ).lower()
            std_correct = check_correct(std_resp, answer)
        except Exception:
            std_resp = ""
            std_correct = False
        results_standard[cid].append(std_correct)

        # Strategy 2: Binary verification using ground-truth answer
        # This is an ORACLE test: if binary verification of the correct answer
        # returns "true", the model can recognize the correct spatial relation
        statement = make_binary_statement(s["question"], answer, relation)
        try:
            bin_resp = run_inference(
                model, processor, process_vision_info, s,
                BINARY_VERIFY_PROMPT.format(statement=statement),
            ).lower().strip()
            # "true" means the model agrees the correct answer is right
            bin_correct = bin_resp.startswith("true")
        except Exception:
            bin_resp = ""
            bin_correct = False
        results_binary[cid].append(bin_correct)

        detail_log.append({
            "id": s["id"],
            "cell": cid,
            "question": s["question"],
            "answer": answer,
            "statement": statement,
            "std_response": std_resp,
            "std_correct": std_correct,
            "bin_response": bin_resp,
            "bin_correct": bin_correct,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            std_acc = np.mean([c for cc in results_standard.values() for c in cc])
            bin_acc = np.mean([c for cc in results_binary.values() for c in cc])
            print(f"  {i+1}/{len(all_samples)} std_acc={std_acc:.3f} "
                  f"bin_acc={bin_acc:.3f} elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0

    # ── Results ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    print(f"\n{'Cell':<30s} {'Standard':>10s} {'Binary(oracle)':>15s} {'Delta':>8s}")
    print("-" * 65)

    all_std = []
    all_bin = []
    cell_results = {}
    for cid in WORST_CELLS:
        if cid in results_standard:
            std_acc = float(np.mean(results_standard[cid]))
            bin_acc = float(np.mean(results_binary[cid]))
            delta = bin_acc - std_acc
            n = len(results_standard[cid])
            print(f"{cid:<30s} {std_acc:>10.4f} {bin_acc:>15.4f} {delta:>+8.4f}  (n={n})")
            all_std.extend(results_standard[cid])
            all_bin.extend(results_binary[cid])
            cell_results[cid] = {
                "standard_acc": std_acc,
                "binary_oracle_acc": bin_acc,
                "delta": delta,
                "n": n,
            }

    overall_std = float(np.mean(all_std))
    overall_bin = float(np.mean(all_bin))
    overall_delta = overall_bin - overall_std
    print(f"{'OVERALL':<30s} {overall_std:>10.4f} {overall_bin:>15.4f} {overall_delta:>+8.4f}")

    # ── Interpretation ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("INTERPRETATION")
    print(f"{'=' * 60}")

    if overall_delta > 0.05:
        signal = "POSITIVE"
        interp = ("Model CAN recognize correct spatial relations in binary format "
                   "but FAILS at open-ended generation. → Data reformulation / "
                   "verify-then-answer pipeline is worth pursuing.")
    elif overall_delta > 0.02:
        signal = "WEAK POSITIVE"
        interp = ("Binary format helps somewhat. Partial evidence for format gap.")
    elif overall_delta > -0.02:
        signal = "INCONCLUSIVE"
        interp = ("No clear format effect. Problem may be truly geometric, not format-dependent.")
    else:
        signal = "NEGATIVE"
        interp = ("Binary format doesn't help either. Problem is fundamentally perceptual.")

    print(f"  Signal: {signal}")
    print(f"  {interp}")
    print(f"  Time: {elapsed:.0f}s")

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        "pilot": "binary_reformat",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "checkpoint": str(CHECKPOINT),
        "split": args.split,
        "n_samples": len(all_samples),
        "elapsed_seconds": elapsed,
        "signal": signal,
        "overall": {
            "standard_acc": overall_std,
            "binary_oracle_acc": overall_bin,
            "delta": overall_delta,
        },
        "per_cell": cell_results,
        "interpretation": interp,
    }
    out_path = OUT_DIR / "pilot_binary_reformat.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Save detail log for analysis
    detail_path = OUT_DIR / "pilot_binary_reformat_detail.json"
    with open(detail_path, "w") as f:
        json.dump(detail_log, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"Detail log saved to {detail_path}")


if __name__ == "__main__":
    main()
