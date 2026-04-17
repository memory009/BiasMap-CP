#!/usr/bin/env python3
"""
Phase 2 Pilot: Object-Level Depth Auxiliary Training (SpatialRGPT-Inspired)

Tests the OBJECT-LEVEL DEPTH SUPERVISION hypothesis:
  "The previous region-level depth auxiliary (left/right, upper/lower) learned
   its task (loss 0.15->0.06) but failed to transfer because it didn't require
   object grounding. Object-level depth QA — naming specific objects and asking
   about their relative depth — forces the model to learn the same reasoning
   chain as spatial QA: language -> object grounding -> depth reasoning."

Key improvement over pilot_depth_auxiliary.py:
  - OLD: "Which side of the image is closer?" (region-level, shortcut-prone)
  - NEW: "Is the chair closer to the camera than the table?" (object-level)

Data sources:
  - GQA scene graphs: provide bounding boxes for all objects
  - GQA raw questions: semantic parse with object IDs
  - Depth Anything V2: monocular depth maps (already cached)

Go/No-Go (8B scale, vs Global FT worst_cell=0.495):
  POSITIVE:      worst_cell < 0.44  (>10% improvement over 8B global FT)
  WEAK POSITIVE: worst_cell 0.44-0.48
  NEGATIVE:      worst_cell >= 0.48 (no meaningful improvement)

  Note: 2B thresholds (0.48/0.51) are obsolete — all new pilots run at 8B.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/pilot_depth_object_level.py --epochs 3 --seed 1
"""
import argparse
import json
import os
import re
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

# -- Config -------------------------------------------------------------------
MODEL_ID     = "Qwen/Qwen3-VL-8B-Instruct"
DEPTH_MODEL  = "depth-anything/Depth-Anything-V2-Small-hf"
CACHE_DIR    = "/LOCAL2/psqhe8/hf_cache"
SPLITS_DIR   = Path("data/splits")
B1_DIR       = Path("results/sprint2/b1_diagnosis")
OUT_DIR      = Path("results/sprint2/pilots")
DEPTH_CACHE  = OUT_DIR / "depth_maps"
DEPTH_NP_CACHE = OUT_DIR / "depth_maps_npy"

# GQA raw data paths
GQA_RAW_DIR       = Path("data/raw/gqa")
GQA_QUESTIONS_FILE = GQA_RAW_DIR / "train_balanced_questions.json"
GQA_SCENE_GRAPHS   = GQA_RAW_DIR / "train_sceneGraphs.json"

WORST_CELLS = [
    "in_front_of|True|gqa",
    "inside|False|gqa",
    "under|False|gqa",
    "behind|True|gqa",
]

# LoRA config (aligned with main 8B experiments)
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

# Training config (aligned with main 8B experiments)
LR             = 2e-4
WEIGHT_DECAY   = 0.01
MAX_GRAD_NORM  = 1.0
MICRO_BS       = 1
GRAD_ACCUM     = 16
WARMUP_RATIO   = 0.03
MIN_CELL_SUPPORT = 20
AUX_RATIO      = 1.0

# -- Prompt templates ---------------------------------------------------------
# Original QA prompts (same as specialist LoRA -- single image, no depth map)
BINARY_PROMPT = 'Look at the image. Is the following spatial statement true or false?\n\nStatement: "{caption}"\n\nAnswer with ONLY "true" or "false".'
OPEN_PROMPT   = 'Look at the image carefully. Answer the following spatial reasoning question with a short answer.\n\nQuestion: {question}\n\nAnswer:'
SPATIAL_PROMPT = 'Look at the image carefully. Answer the following spatial reasoning question.\n\nQuestion: {question}\n\nChoose the correct answer from: {choices}\n\nAnswer with ONLY the letter or the exact answer text, nothing else.'

# Object-level depth auxiliary prompts
AUX_BINARY_PROMPT = (
    'Look at this image carefully. '
    'Is the {obj_a} closer to the camera than the {obj_b}?\n\n'
    'Answer with ONLY "yes" or "no".'
)
AUX_WHICH_PROMPT = (
    'Look at this image carefully. '
    'Which object is closer to the camera: the {obj_a} or the {obj_b}?\n\n'
    'Answer with ONLY the object name, nothing else.'
)
AUX_VERIFY_PROMPT = (
    'Look at this image. Is the following spatial statement true or false?\n\n'
    'Statement: "The {obj_a} is in front of the {obj_b}."\n\n'
    'Answer with ONLY "true" or "false".'
)


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


# -- Depth map generation (reuse from previous pilot) -------------------------

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
            np.save(DEPTH_NP_CACHE / f"{img_id}.npy", depth_np)

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


# -- Object-level auxiliary generation ----------------------------------------

def extract_object_pair(raw_entry, scene_graph_objects):
    """
    Extract the two objects involved in a spatial relation from raw GQA entry.

    Returns dict with anchor/related object info, or None if extraction fails.
    """
    semantic = raw_entry.get("semantic", [])
    if not semantic:
        return None

    anchor_id, anchor_name = None, None
    related_id, related_name = None, None

    for step in semantic:
        arg = step.get("argument", "")
        op = step.get("operation", "")
        m = re.search(r"\((\d+)\)", arg)
        if not m:
            continue
        obj_id = m.group(1)

        if op == "select":
            anchor_id = obj_id
            anchor_name = arg.split("(")[0].strip()
        elif op in ("relate", "verify rel", "choose rel"):
            related_id = obj_id
            # Parse: "name,relation,direction (id)" or "_,relation,direction (id)"
            parts = arg.split("(")[0].strip().split(",")
            if parts:
                related_name = parts[0].strip()

    if not anchor_id or not related_id:
        return None

    # Look up names and bboxes from scene graph (more specific than semantic parse)
    if anchor_id not in scene_graph_objects or related_id not in scene_graph_objects:
        return None

    anchor_sg = scene_graph_objects[anchor_id]
    related_sg = scene_graph_objects[related_id]

    # Use scene graph names (they're more specific than semantic parse placeholders)
    anchor_name_sg = anchor_sg.get("name", anchor_name or "object")
    related_name_sg = related_sg.get("name", related_name or "object")

    # Skip if both objects have the same name (ambiguous for auxiliary QA)
    # But allow it if they're different instances
    anchor_bbox = (anchor_sg["x"], anchor_sg["y"], anchor_sg["w"], anchor_sg["h"])
    related_bbox = (related_sg["x"], related_sg["y"], related_sg["w"], related_sg["h"])

    return {
        "anchor_id": anchor_id,
        "anchor_name": anchor_name_sg,
        "anchor_bbox": anchor_bbox,
        "related_id": related_id,
        "related_name": related_name_sg,
        "related_bbox": related_bbox,
    }


def compute_object_depth(depth_np, bbox):
    """Mean depth within an object's bounding box region."""
    x, y, w, h = bbox
    h_img, w_img = depth_np.shape
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(w_img, int(x + w)), min(h_img, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    region = depth_np[y1:y2, x1:x2]
    if region.size == 0:
        return None
    return float(region.mean())


def generate_object_level_aux(train_samples, raw_questions, scene_graphs, rng):
    """
    Generate object-level depth auxiliary QA pairs.

    For each worst-cell GQA sample, extracts the two objects from the semantic
    parse, looks up their bboxes in the scene graph, computes depth comparison
    using Depth Anything V2 maps, and generates auxiliary QA.
    """
    aux_samples = []
    stats = Counter()

    for s in train_samples:
        if s["dataset"] != "gqa":
            stats["skip_non_gqa"] += 1
            continue

        raw_id = s["id"].replace("gqa_", "")
        if raw_id not in raw_questions:
            stats["skip_not_in_raw"] += 1
            continue

        raw = raw_questions[raw_id]
        image_id = raw.get("imageId", "")

        if image_id not in scene_graphs:
            stats["skip_no_scene_graph"] += 1
            continue

        sg = scene_graphs[image_id]
        sg_objects = sg.get("objects", {})

        # Extract object pair from semantic parse
        pair = extract_object_pair(raw, sg_objects)
        if pair is None:
            stats["skip_no_pair"] += 1
            continue

        # Skip tiny objects (bbox area < 100 px)
        anchor_area = pair["anchor_bbox"][2] * pair["anchor_bbox"][3]
        related_area = pair["related_bbox"][2] * pair["related_bbox"][3]
        if anchor_area < 100 or related_area < 100:
            stats["skip_tiny_bbox"] += 1
            continue

        # Load depth map
        depth_np = load_depth_np(s)
        if depth_np is None:
            stats["skip_no_depth"] += 1
            continue

        # Compute object-level depths
        anchor_depth = compute_object_depth(depth_np, pair["anchor_bbox"])
        related_depth = compute_object_depth(depth_np, pair["related_bbox"])
        if anchor_depth is None or related_depth is None:
            stats["skip_bad_depth"] += 1
            continue

        # Skip ambiguous depth (< 5% relative difference)
        depth_range = float(depth_np.max() - depth_np.min())
        if depth_range < 1e-6 or abs(anchor_depth - related_depth) < 0.05 * depth_range:
            stats["skip_ambiguous"] += 1
            continue

        # Depth Anything V2: HIGHER value = FARTHER from camera
        # So LOWER depth = closer to camera
        anchor_closer = anchor_depth < related_depth

        obj_a = pair["anchor_name"]
        obj_b = pair["related_name"]

        # Generate auxiliary QA (pick 1 type randomly)
        aux_type = rng.choice(["binary", "which", "verify"])

        if aux_type == "binary":
            question = AUX_BINARY_PROMPT.format(obj_a=obj_a, obj_b=obj_b)
            answer = "yes" if anchor_closer else "no"
            type_tag = "depth_obj_binary"

        elif aux_type == "which":
            question = AUX_WHICH_PROMPT.format(obj_a=obj_a, obj_b=obj_b)
            answer = obj_a if anchor_closer else obj_b
            type_tag = "depth_obj_which"

        else:  # verify
            # Randomly decide claim direction (50/50) to balance true/false
            if rng.random() < 0.5:
                claim_true = anchor_closer
                question = AUX_VERIFY_PROMPT.format(obj_a=obj_a, obj_b=obj_b)
            else:
                claim_true = not anchor_closer
                question = AUX_VERIFY_PROMPT.format(obj_a=obj_b, obj_b=obj_a)
            answer = "true" if claim_true else "false"
            type_tag = "depth_obj_verify"

        aux_samples.append({
            "image_path": s["image_path"],
            "question": question,
            "answer": answer,
            "is_auxiliary": True,
            "aux_type": type_tag,
            "dataset": s["dataset"],
            "relation_type": s["relation_type"],
            "depth_ambiguity": s.get("depth_ambiguity", False),
        })
        stats[type_tag] += 1

    return aux_samples, stats


# -- Tokenization (single image, standard QA) --------------------------------

def tokenize_train_example(processor, sample, process_vision_info):
    """Tokenize a training example -- works for both original and auxiliary QA."""
    if sample.get("is_auxiliary"):
        prompt = sample["question"]
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
    """Evaluate per-cell loss -- ONLY on original QA, NOT auxiliary."""
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
    parser.add_argument("--aux_weight", type=float, default=1.0,
                        help="Loss weight for auxiliary samples (default 1.0)")
    parser.add_argument("--model_id", type=str, default=MODEL_ID,
                        help="Model ID (default: Qwen3-VL-2B)")
    parser.add_argument("--run_tag", type=str, default=None,
                        help="Optional suffix appended to the default run directory "
                             "(save-path only; does not change training recipe). "
                             "Example: --run_tag r1_replication -> "
                             "pilot_depth_object_level_seed{seed}_{model_tag}_r1_replication")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Optional absolute/relative output directory override. "
                             "If set, takes precedence over --run_tag. Save-path only.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed + 42)
    torch.manual_seed(args.seed + 42)
    np.random.seed(args.seed + 42)

    # Encode model name in output dir to avoid collisions (e.g. 2B vs 8B)
    model_tag = args.model_id.split("/")[-1].lower().replace("-instruct", "").replace("-", "")
    default_name = f"pilot_depth_object_level_seed{args.seed}_{model_tag}"
    if args.output_dir:
        run_dir = Path(args.output_dir)
    elif args.run_tag:
        run_dir = OUT_DIR / f"{default_name}_{args.run_tag}"
    else:
        run_dir = OUT_DIR / default_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 2 Pilot: Object-Level Depth Auxiliary (SpatialRGPT-Inspired)")
    print(f"Target cells: {WORST_CELLS}")
    print(f"Depth model: {DEPTH_MODEL}")
    print(f"VLM: {args.model_id}")
    print(f"LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"Aux ratio: {args.aux_ratio}, Aux weight: {args.aux_weight}")
    print("=" * 60)
    print("\nNOTE: Object-level auxiliary QA from GQA scene graphs + depth maps.")
    print("      Single-image input. No depth map at inference.\n")

    # -- Phase 0a: Load worst-cell data ---------------------------------------
    print("Phase 0a: Loading worst-cell data...")
    train_samples = load_worst_cell_samples("train")
    eval_samples = load_worst_cell_samples("repair_val")
    all_samples = train_samples + eval_samples
    print(f"  Train: {len(train_samples)}, Eval: {len(eval_samples)}")

    # -- Phase 0b: Ensure depth maps cached ------------------------------------
    print("\nPhase 0b: Checking/generating depth maps...")
    generate_depth_maps(all_samples, device="cuda")

    # -- Phase 0c: Load GQA raw questions + scene graphs -----------------------
    print("\nPhase 0c: Loading GQA raw questions...")
    if not GQA_QUESTIONS_FILE.exists():
        print(f"  ERROR: {GQA_QUESTIONS_FILE} not found!")
        sys.exit(1)
    with open(GQA_QUESTIONS_FILE) as f:
        raw_questions = json.load(f)
    print(f"  Loaded {len(raw_questions)} raw GQA questions")

    print("\nPhase 0d: Loading GQA scene graphs...")
    if not GQA_SCENE_GRAPHS.exists():
        print(f"  ERROR: {GQA_SCENE_GRAPHS} not found!")
        print(f"  Run: cd data/raw/gqa && wget https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip && unzip -o sceneGraphs.zip && rm sceneGraphs.zip")
        sys.exit(1)
    with open(GQA_SCENE_GRAPHS) as f:
        scene_graphs = json.load(f)
    print(f"  Loaded scene graphs for {len(scene_graphs)} images")

    # -- Phase 0e: Generate object-level auxiliary samples ----------------------
    print("\nPhase 0e: Generating object-level depth auxiliary samples...")
    aux_train, aux_stats = generate_object_level_aux(
        train_samples, raw_questions, scene_graphs, rng
    )
    print(f"  Generated {len(aux_train)} auxiliary samples from {len(train_samples)} originals")
    print(f"  Generation stats:")
    for k, v in sorted(aux_stats.items()):
        print(f"    {k}: {v}")

    # Free raw data from memory (can be large)
    del raw_questions, scene_graphs

    # Mark original samples
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

    # -- Load VLM --------------------------------------------------------------
    from transformers import AutoProcessor, BitsAndBytesConfig, get_cosine_schedule_with_warmup
    from transformers import Qwen3VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, TaskType
    from qwen_vl_utils import process_vision_info

    print(f"\nLoading {args.model_id}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=CACHE_DIR)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, cache_dir=CACHE_DIR, quantization_config=bnb_config,
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

    # -- Training setup --------------------------------------------------------
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

    # -- Pre-training eval (original QA only) ----------------------------------
    print("\nPre-training eval on worst cells (original QA, no depth map)...")
    pre_eval = evaluate_loss(model, processor, process_vision_info, eval_samples, partition)
    print(f"  Pre-train worst_10pct_cvar: {pre_eval['worst_10pct_cvar']:.4f}")
    print(f"  Pre-train worst_cell_loss: {pre_eval['worst_cell_loss']:.4f}")
    for cid, loss in sorted(pre_eval["cell_losses"].items()):
        print(f"    {cid}: {loss:.4f}")

    # -- Training loop ---------------------------------------------------------
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

                # Apply auxiliary weight
                if sample.get("is_auxiliary"):
                    loss = outputs.loss * args.aux_weight / GRAD_ACCUM
                    epoch_loss_aux += outputs.loss.item()
                    n_aux += 1
                else:
                    loss = outputs.loss / GRAD_ACCUM
                    epoch_loss_orig += outputs.loss.item()
                    n_orig += 1

                loss.backward()

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
        print(f"\n  Epoch {epoch} -- orig_loss: {avg_orig_loss:.4f}, "
              f"aux_loss: {avg_aux_loss:.4f} "
              f"(n_orig={n_orig}, n_aux={n_aux})")

        # Eval (original QA only)
        print(f"  Evaluating (original QA only)...")
        eval_result = evaluate_loss(model, processor, process_vision_info, eval_samples, partition)
        w10 = eval_result["worst_10pct_cvar"]
        wc = eval_result["worst_cell_loss"]
        print(f"  worst_10pct_cvar: {w10:.4f}")
        print(f"  worst_cell_loss: {wc:.4f}")
        for cid, loss in sorted(eval_result["cell_losses"].items()):
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
            print(f"  * New best: W10%CVaR={w10:.4f}")

    elapsed = time.time() - train_start

    # -- Comparison (scale-aware baselines) ------------------------------------
    is_8b = "8B" in args.model_id or "8b" in args.model_id

    if is_8b:
        # 8B baselines (seed 1, from B2-8B experiments)
        # No 8B specialist or region-aux baseline exists (not yet run)
        GLOBAL_W10_CVAR   = 0.343   # 8B global FT seed1 W10%CVaR_loss
        GLOBAL_WORST_CELL = 0.495   # 8B global FT seed1 worst_cell_loss
        CVAR_CELL_W10_CVAR   = 0.341  # 8B cvar_cell seed1 W10%CVaR_loss
        CVAR_CELL_WORST_CELL = 0.510  # 8B cvar_cell seed1 worst_cell_loss
        global_cell_losses = {
            # 8B global FT 3-seed mean per-cell loss (s1/s2/s3 best epoch=2)
            "in_front_of|True|gqa": 0.4775,  # s1=0.4952, s2=0.4701, s3=0.4671
            "inside|False|gqa":     0.2904,  # s1=0.2767, s2=0.2776, s3=0.3168
            "under|False|gqa":      0.3223,  # s1=0.3188, s2=0.2924, s3=0.3555
            "behind|True|gqa":      0.2695,  # s1=0.2819, s2=0.2797, s3=0.2469
        }
        # Go/No-Go thresholds calibrated to 8B loss scale:
        # Global FT worst_cell=0.495; meaningful improvement = >10% reduction
        THRESHOLD_POSITIVE      = 0.44
        THRESHOLD_WEAK_POSITIVE = 0.48
    else:
        # 2B baselines
        GLOBAL_W10_CVAR      = 0.4077
        GLOBAL_WORST_CELL    = 0.5196
        CVAR_CELL_W10_CVAR   = float("nan")
        CVAR_CELL_WORST_CELL = float("nan")
        global_cell_losses = {
            "in_front_of|True|gqa": 0.5196,
            "inside|False|gqa":     0.4165,
            "under|False|gqa":      0.4144,
            "behind|True|gqa":      0.2802,
        }
        THRESHOLD_POSITIVE      = 0.48
        THRESHOLD_WEAK_POSITIVE = 0.51

    # 2B-only baselines (specialist + region aux — not available for 8B)
    SPECIALIST_W10_CVAR   = 0.5409 if not is_8b else float("nan")
    SPECIALIST_WORST_CELL = 0.5409 if not is_8b else float("nan")
    REGION_AUX_W10_CVAR   = 0.5397 if not is_8b else float("nan")
    REGION_AUX_WORST_CELL = 0.5397 if not is_8b else float("nan")
    specialist_cell_losses = {
        "in_front_of|True|gqa": 0.5409 if not is_8b else float("nan"),
        "inside|False|gqa":     0.4543 if not is_8b else float("nan"),
        "under|False|gqa":      0.5094 if not is_8b else float("nan"),
        "behind|True|gqa":      0.3166 if not is_8b else float("nan"),
    }
    region_aux_cell_losses = {
        "in_front_of|True|gqa": 0.5397 if not is_8b else float("nan"),
        "inside|False|gqa":     0.4375 if not is_8b else float("nan"),
        "under|False|gqa":      0.5267 if not is_8b else float("nan"),
        "behind|True|gqa":      0.3404 if not is_8b else float("nan"),
    }

    best_eval = epoch_history[best_epoch - 1]["eval"]
    obj_w10 = best_eval["worst_10pct_cvar"]
    obj_wc = best_eval["worst_cell_loss"]

    scale_tag = "8B" if is_8b else "2B"
    print(f"\n{'=' * 60}")
    print(f"COMPARISON ({scale_tag}): Object-Level Aux vs Global FT" +
          (" vs CVaR-Cell" if is_8b else " vs Region-Level Aux vs Specialist"))
    print(f"{'=' * 60}")
    if is_8b:
        print(f"  {'Metric':<25s} {'ObjLvlAux':>10s} {'GlobalFT':>10s} {'CVaRCell':>10s}")
        print(f"  {'-' * 55}")
        print(f"  {'W10% CVaR (loss)':<25s} {obj_w10:>10.4f} {GLOBAL_W10_CVAR:>10.4f} {CVAR_CELL_W10_CVAR:>10.4f}")
        print(f"  {'Worst cell loss':<25s} {obj_wc:>10.4f} {GLOBAL_WORST_CELL:>10.4f} {CVAR_CELL_WORST_CELL:>10.4f}")
    else:
        print(f"  {'Metric':<25s} {'ObjLevel':>10s} {'RegLevel':>10s} {'Specialist':>10s} {'GlobalFT':>10s}")
        print(f"  {'-' * 60}")
        print(f"  {'W10% CVaR':<25s} {obj_w10:>10.4f} {REGION_AUX_W10_CVAR:>10.4f} {SPECIALIST_W10_CVAR:>10.4f} {GLOBAL_W10_CVAR:>10.4f}")
        print(f"  {'Worst cell loss':<25s} {obj_wc:>10.4f} {REGION_AUX_WORST_CELL:>10.4f} {SPECIALIST_WORST_CELL:>10.4f} {GLOBAL_WORST_CELL:>10.4f}")

    print(f"\n  Per-cell comparison:")
    if is_8b:
        print(f"  {'Cell':<30s} {'ObjLvlAux':>10s} {'GlobalFT':>10s}")
        for cid in WORST_CELLS:
            ol = best_eval["cell_losses"].get(cid, float("nan"))
            gl = global_cell_losses.get(cid, float("nan"))
            print(f"  {cid:<30s} {ol:>10.4f} {gl:>10.4f}")
    else:
        print(f"  {'Cell':<30s} {'ObjLevel':>10s} {'RegLevel':>10s} {'Specialist':>10s} {'GlobalFT':>10s}")
        for cid in WORST_CELLS:
            ol = best_eval["cell_losses"].get(cid, float("nan"))
            rl = region_aux_cell_losses.get(cid, float("nan"))
            sp = specialist_cell_losses.get(cid, float("nan"))
            gl = global_cell_losses.get(cid, float("nan"))
            print(f"  {cid:<30s} {ol:>10.4f} {rl:>10.4f} {sp:>10.4f} {gl:>10.4f}")

    # Go/No-Go
    print(f"\n{'=' * 60}")
    print(f"GO/NO-GO ASSESSMENT ({scale_tag} scale, threshold POSITIVE<{THRESHOLD_POSITIVE}, WEAK<{THRESHOLD_WEAK_POSITIVE})")
    print(f"{'=' * 60}")

    if obj_wc < THRESHOLD_POSITIVE:
        signal = "POSITIVE"
        verdict = "Object-level depth auxiliary breaks the loss ceiling -> object grounding + depth IS the missing skill"
    elif obj_wc < THRESHOLD_WEAK_POSITIVE:
        signal = "WEAK POSITIVE"
        verdict = "Some improvement from object-level depth -> partial signal, consider scaling up or tuning aux_weight"
    else:
        signal = "NEGATIVE"
        verdict = "Object-level depth auxiliary can't improve worst cells -> problem may require architectural changes"

    depth_vs_global = obj_wc - GLOBAL_WORST_CELL
    depth_vs_region = obj_wc - REGION_AUX_WORST_CELL

    print(f"  Signal: {signal}")
    print(f"  Verdict: {verdict}")
    print(f"  vs Global FT:      {depth_vs_global:+.4f}")
    if not is_8b:
        print(f"  vs Specialist:     {obj_wc - SPECIALIST_WORST_CELL:+.4f}")
        print(f"  vs Region-Level:   {depth_vs_region:+.4f}")
    print(f"  Training time: {elapsed / 3600:.1f}h")

    # -- Save results ----------------------------------------------------------
    output = {
        "pilot": "depth_object_level_auxiliary",
        "pilot_type": "Object-level auxiliary (SpatialRGPT-inspired)",
        "hypothesis": "object_level_depth_supervision_bridges_grounding_gap",
        "improvement_over": "pilot_depth_auxiliary (region-level, NEGATIVE)",
        "timestamp": datetime.now().isoformat(),
        "model": args.model_id,
        "depth_model": DEPTH_MODEL,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "aux_weight": args.aux_weight,
        "worst_cells": WORST_CELLS,
        "n_train_original": len(train_samples),
        "n_train_auxiliary": len(aux_train),
        "n_train_combined": len(combined_train),
        "n_eval": len(eval_samples),
        "aux_generation_stats": dict(aux_stats),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "elapsed_hours": elapsed / 3600,
        "pre_training_eval": pre_eval,
        "epoch_history": epoch_history,
        "best_eval": best_eval,
        "scale": scale_tag,
        "comparison": {
            "global_ft_w10_cvar": GLOBAL_W10_CVAR,
            "global_ft_worst_cell": GLOBAL_WORST_CELL,
            "obj_level_w10_cvar": obj_w10,
            "obj_level_worst_cell": obj_wc,
            "delta_vs_global": depth_vs_global,
            # 8B-specific
            "cvar_cell_w10_cvar": CVAR_CELL_W10_CVAR if is_8b else None,
            "cvar_cell_worst_cell": CVAR_CELL_WORST_CELL if is_8b else None,
            # 2B-only
            "specialist_w10_cvar": SPECIALIST_W10_CVAR if not is_8b else None,
            "specialist_worst_cell": SPECIALIST_WORST_CELL if not is_8b else None,
            "region_aux_w10_cvar": REGION_AUX_W10_CVAR if not is_8b else None,
            "region_aux_worst_cell": REGION_AUX_WORST_CELL if not is_8b else None,
            "delta_vs_region_aux": depth_vs_region if not is_8b else None,
        },
        "signal": signal,
        "verdict": verdict,
    }

    out_path = run_dir / "pilot_depth_object_level.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
