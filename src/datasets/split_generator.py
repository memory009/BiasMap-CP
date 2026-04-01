"""Generate standard and 5 OOD splits for BiasMap-CP.

OOD splits:
  1. Frame-OOD     — mirrored left/right + held-out front/behind by scene
  2. Concept-OOD   — held-out (subject, object) pairs
  3. Tail-Risk     — rare relations, tiny/occluded objects
  4. Compositional-OOD — multi-relation GQA questions
  5. Shifted-Calibration — skewed relation distribution at test time
"""
import os
import json
import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
from PIL import Image
import numpy as np

from .base import SpatialQASample


SPLIT_NAMES = ["train", "cal", "test",
               "ood_frame", "ood_concept", "ood_tailrisk",
               "ood_compositional", "ood_shifted_cal"]


def stratified_split(samples: List[SpatialQASample],
                     train_r: float = 0.6, cal_r: float = 0.2,
                     seed: int = 42) -> Dict[str, List[SpatialQASample]]:
    """Split samples into train/cal/test stratified by relation_type."""
    random.seed(seed)
    by_relation = defaultdict(list)
    for s in samples:
        by_relation[s.relation_type].append(s)

    splits = {"train": [], "cal": [], "test": []}
    for rel, rel_samples in by_relation.items():
        random.shuffle(rel_samples)
        n = len(rel_samples)
        n_train = int(n * train_r)
        n_cal = int(n * cal_r)
        splits["train"].extend(rel_samples[:n_train])
        splits["cal"].extend(rel_samples[n_train:n_train + n_cal])
        splits["test"].extend(rel_samples[n_train + n_cal:])

    return splits


def generate_frame_ood(samples: List[SpatialQASample],
                       target_size: int = 4000,
                       seed: int = 42) -> List[SpatialQASample]:
    """Frame-OOD: mirror left↔right images + hold out front/behind from one scene type."""
    random.seed(seed)
    ood = []

    # Left/right samples — apply horizontal flip
    lr_samples = [s for s in samples if s.relation_type in ("left", "right")]
    lr_samples = random.sample(lr_samples, min(len(lr_samples), target_size // 2))
    for s in lr_samples:
        flipped = _mirror_sample(s)
        ood.append(flipped)

    # Front/behind from a single scene type (indoor)
    fb_samples = [s for s in samples
                  if s.relation_type in ("in_front_of", "behind")
                  and s.scene_type in (None, "indoor", "outdoor")]
    fb_samples = random.sample(fb_samples, min(len(fb_samples), target_size // 2))
    ood.extend(fb_samples)

    random.shuffle(ood)
    return ood[:target_size]


def _mirror_sample(s: SpatialQASample) -> SpatialQASample:
    """Create a mirrored version: flip image horizontally, swap left↔right in answer/relation."""
    import copy
    mirrored = copy.deepcopy(s)
    mirrored.id = s.id + "_mirror"

    # Try to create mirrored image file
    if s.image_path and os.path.exists(s.image_path):
        base, ext = os.path.splitext(s.image_path)
        mirror_path = base + "_mirror" + ext
        if not os.path.exists(mirror_path):
            try:
                img = Image.open(s.image_path)
                img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                img_flipped.save(mirror_path)
            except Exception:
                mirror_path = s.image_path  # fallback
        mirrored.image_path = mirror_path

    # Flip relation
    if s.relation_type == "left":
        mirrored.relation_type = "right"
        mirrored.answer = "right"
    elif s.relation_type == "right":
        mirrored.relation_type = "left"
        mirrored.answer = "left"

    # Flip bboxes if present (x' = 1 - x for normalized coords)
    if s.subject_bbox:
        b = s.subject_bbox
        mirrored.subject_bbox = [1 - b[2], b[1], 1 - b[0], b[3]]
    if s.object_bbox:
        b = s.object_bbox
        mirrored.object_bbox = [1 - b[2], b[1], 1 - b[0], b[3]]

    return mirrored


def generate_concept_ood(samples: List[SpatialQASample],
                         target_size: int = 4000,
                         holdout_frac: float = 0.2,
                         seed: int = 42) -> List[SpatialQASample]:
    """Concept-OOD: hold out 20% of (subject, object) pairs entirely."""
    random.seed(seed)

    # Count (subject, object) pairs
    pair_counter = Counter()
    for s in samples:
        if s.subject and s.object:
            pair_counter[(s.subject.lower(), s.object.lower())] += 1

    top_pairs = [p for p, _ in pair_counter.most_common(100)]
    n_holdout = max(1, int(len(top_pairs) * holdout_frac))
    n_holdout = min(n_holdout, len(top_pairs))
    holdout_pairs = set(random.sample(top_pairs, n_holdout)) if top_pairs else set()

    ood = [s for s in samples
           if s.subject and s.object
           and (s.subject.lower(), s.object.lower()) in holdout_pairs]

    random.shuffle(ood)
    return ood[:target_size]


def generate_tail_risk(samples: List[SpatialQASample],
                       target_size: int = 4000,
                       seed: int = 42) -> List[SpatialQASample]:
    """Tail-Risk: rare relations + tiny/occluded objects."""
    random.seed(seed)

    rel_counts = Counter(s.relation_type for s in samples)
    total = len(samples)
    rare_rels = {rel for rel, cnt in rel_counts.items()
                 if cnt / total < 0.10}  # bottom 10% by frequency

    ood = []
    for s in samples:
        is_tail = False
        if s.relation_type in rare_rels:
            is_tail = True
        if s.object_size_ratio is not None and (
            s.object_size_ratio < 0.1 or s.object_size_ratio > 10
        ):
            is_tail = True
        if s.occlusion_level == "heavy":
            is_tail = True
        if is_tail:
            ood.append(s)

    random.shuffle(ood)
    return ood[:target_size]


def generate_compositional_ood(samples: List[SpatialQASample],
                                target_size: int = 4000,
                                seed: int = 42) -> List[SpatialQASample]:
    """Compositional-OOD: multi-relation questions (require 2+ spatial keywords)."""
    random.seed(seed)

    multi_rel_keywords = [
        "left of", "right of", "above", "below", "behind", "in front of",
        "next to", "near", "far from", "on top of", "under", "beside",
    ]

    def count_relations(q: str) -> int:
        q_lower = q.lower()
        return sum(1 for kw in multi_rel_keywords if kw in q_lower)

    ood = [s for s in samples if s.dataset == "gqa" and count_relations(s.question) >= 2]

    # Fall back to any dataset if GQA not enough
    if len(ood) < target_size // 2:
        extra = [s for s in samples
                 if s.dataset != "gqa" and count_relations(s.question) >= 2]
        ood.extend(extra)

    random.shuffle(ood)
    return ood[:target_size]


def generate_shifted_calibration(samples: List[SpatialQASample],
                                  target_size: int = 4000,
                                  seed: int = 42) -> Tuple[
                                      List[SpatialQASample],
                                      List[SpatialQASample]]:
    """Shifted-Cal: calibration = uniform distribution; test = skewed (80% left/right)."""
    random.seed(seed)

    by_rel = defaultdict(list)
    for s in samples:
        by_rel[s.relation_type].append(s)

    # Uniform calibration: equal count per relation
    cal_per_rel = min(200, min(len(v) for v in by_rel.values() if len(v) > 0))
    shifted_cal = []
    for rel_samples in by_rel.values():
        random.shuffle(rel_samples)
        shifted_cal.extend(rel_samples[:cal_per_rel])

    # Skewed test: 80% left/right
    lr_pool = by_rel.get("left", []) + by_rel.get("right", [])
    other_pool = [s for rel, ss in by_rel.items()
                  for s in ss if rel not in ("left", "right")]
    n_lr = int(target_size * 0.8)
    n_other = target_size - n_lr
    random.shuffle(lr_pool)
    random.shuffle(other_pool)
    shifted_test = lr_pool[:n_lr] + other_pool[:n_other]
    random.shuffle(shifted_test)

    return shifted_cal[:target_size], shifted_test[:target_size]


def generate_all_splits(all_samples: List[SpatialQASample],
                        output_dir: str,
                        target_ood_size: int = 4000,
                        seed: int = 42) -> Dict[str, List[SpatialQASample]]:
    """Generate all splits and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    print(f"\nGenerating splits from {len(all_samples)} total samples...")

    # Standard splits
    standard = stratified_split(all_samples, seed=seed)
    print(f"  Standard: train={len(standard['train'])}, cal={len(standard['cal'])}, test={len(standard['test'])}")

    # OOD splits (use full sample pool for OOD generation to maximize coverage)
    ood_frame = generate_frame_ood(all_samples, target_ood_size, seed)
    ood_concept = generate_concept_ood(all_samples, target_ood_size, seed)
    ood_tailrisk = generate_tail_risk(all_samples, target_ood_size, seed)
    ood_comp = generate_compositional_ood(all_samples, target_ood_size, seed)
    ood_shifted_cal, ood_shifted_test = generate_shifted_calibration(
        standard["cal"] + standard["test"], target_ood_size, seed
    )

    print(f"  Frame-OOD: {len(ood_frame)}")
    print(f"  Concept-OOD: {len(ood_concept)}")
    print(f"  Tail-Risk: {len(ood_tailrisk)}")
    print(f"  Compositional-OOD: {len(ood_comp)}")
    print(f"  Shifted-Cal: cal={len(ood_shifted_cal)}, test={len(ood_shifted_test)}")

    all_splits = {
        "train": standard["train"],
        "cal": standard["cal"],
        "test": standard["test"],
        "ood_frame": ood_frame,
        "ood_concept": ood_concept,
        "ood_tailrisk": ood_tailrisk,
        "ood_compositional": ood_comp,
        "ood_shifted_cal_calib": ood_shifted_cal,
        "ood_shifted_cal_test": ood_shifted_test,
    }

    for split_name, split_samples in all_splits.items():
        out_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(out_path, "w") as f:
            for s in split_samples:
                f.write(s.to_json() + "\n")
        print(f"  Saved {split_name}: {len(split_samples)} → {out_path}")

    # Save split stats
    stats = {
        name: {
            "total": len(ss),
            "relation_distribution": dict(Counter(s.relation_type for s in ss)),
            "dataset_distribution": dict(Counter(s.dataset for s in ss)),
        }
        for name, ss in all_splits.items()
    }
    with open(os.path.join(output_dir, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSplit stats saved to {output_dir}/split_stats.json")

    return all_splits
