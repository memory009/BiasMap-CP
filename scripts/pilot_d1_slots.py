#!/usr/bin/env python3
"""
R2/R3/R4 pilot: D1 Named-Object Geometry Slots on top of the V2 LoRA recipe.

Contains:
  - NC cache loader + cell-conditional tau_h quantile (ALPHA_GATE = 0.30)
  - CP gate function with per-cell activation logging
  - Per-sample named-object matching + depth lookup (same logic as preflight)
  - D1SlotModule + SlotCrossAttentionAdapter (Strategy B forward hook)
  - Shared-config loader (refine-logs/r1_config.json)
  - Training loop SHELL with explicit injection points for the slot path

Forward-pass injection uses Strategy B:
  - A forward hook on the model's final LayerNorm adds a lightweight cross-attn
    residual. Queries from decoder hidden states (projected down to d_slot=256),
    keys/values from slot embeddings (d_slot). Output projected back up to d_model
    and gated by a learned scalar (init ~ 0). Ungated samples bypass the hook
    entirely — exact R1 behavior preserved.

DO NOT launch this script until:
  1. R1 (baseline replication) has completed and its best-epoch
     worst_cell_error_hard4 is within [0.3487, 0.3687].
  2. refine-logs/r1_config.json has been written (frozen shared config).
  3. refine-logs/nc_cache_qwen3vl8b_r1_seed1_train.json has been produced by
     scripts/compute_nc_cache.py.

Mode selection:
  --mode r2_selective   : CP-gated slot path (main pilot)
  --mode r3_always_on   : slot path on for every sample with n_match >= 1
  --mode r4_depth_token : replace slot producer with DepthTokenModule
                          (no slot-attn, no box-pooled vision features,
                          only [depth, box_coords] → MLP; adapter cross-attn
                          block is retained) — architecture ablation

Parameter separation (user-required):
  ALPHA_GATE = 0.30   (CP routing parameter, used HERE)
  ALPHA_CVAR = 0.10   (CVaR evaluation parameter, kept in metrics only)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Reuse constants + helpers from R1 to guarantee apples-to-apples with V2.
from scripts.pilot_depth_object_level import (
    MODEL_ID,
    DEPTH_MODEL,
    CACHE_DIR,
    SPLITS_DIR,
    GQA_QUESTIONS_FILE,
    GQA_SCENE_GRAPHS,
    WORST_CELLS,
    DEPTH_NP_CACHE,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    TARGET_MODULES,
    LR,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    MICRO_BS,
    GRAD_ACCUM,
    WARMUP_RATIO,
    MIN_CELL_SUPPORT,
    build_prompt,
    build_answer,
    get_cell_id,
    load_image,
    load_worst_cell_samples,
    compute_object_depth,
)

from src.repair.slot_module import D1SlotModule, DepthTokenModule, SlotCrossAttentionAdapter, SlotHookManager, K_MAX, D_SLOT

ALPHA_GATE = 0.30
ALPHA_CVAR = 0.10  # separate; never overlaps ALPHA_GATE in code
NC_CACHE_PATH_DEFAULT = "refine-logs/nc_cache_qwen3vl8b_r1_seed1_train.json"
SHARED_CONFIG_PATH_DEFAULT = "refine-logs/r1_config.json"
PAIR_AUX_WEIGHT_DEFAULT = 0.2

# Qwen3-VL-8B hidden size (verified at runtime against model.config.hidden_size)
D_MODEL_QWEN3VL_8B = 4096


# ---------------------------------------------------------------------------
# Named-object matching (shared with scripts/r2_preflight_coverage.py)
# ---------------------------------------------------------------------------

def extract_named_object_ids(raw_entry) -> list[str]:
    semantic = raw_entry.get("semantic", [])
    seen: set[str] = set()
    ordered: list[str] = []
    for step in semantic:
        arg = step.get("argument", "")
        for m in re.finditer(r"\((\d+)\)", arg):
            oid = m.group(1)
            if oid not in seen:
                seen.add(oid)
                ordered.append(oid)
    return ordered


def match_named_objects(raw_entry, sg_objects) -> list[dict]:
    out = []
    for oid in extract_named_object_ids(raw_entry):
        if oid not in sg_objects:
            continue
        sg = sg_objects[oid]
        bbox = (sg.get("x"), sg.get("y"), sg.get("w"), sg.get("h"))
        if None in bbox or not bbox[2] or not bbox[3]:
            continue
        out.append({"id": oid, "name": sg.get("name", "object"), "bbox": bbox})
        if len(out) >= K_MAX:
            break
    return out


def resolve_raw_qid(sample_id: str) -> str:
    return sample_id.split("_", 1)[1] if sample_id.startswith("gqa_") else sample_id


# ---------------------------------------------------------------------------
# NC cache + gate
# ---------------------------------------------------------------------------

class CPGate:
    """Fixed-threshold CP gate over Mondrian cells, sourced from a frozen NC cache.

    Gate rule:
        g(x) = 1[ m(x) in H4 and s(x) >= tau_{m(x)} and n_match(x) >= 1 ]
    """

    def __init__(self, cache_path: Path, mode: str):
        assert mode in {"r2_selective", "r3_always_on", "r4_depth_token"}
        data = json.loads(Path(cache_path).read_text())
        assert abs(data["alpha_gate"] - ALPHA_GATE) < 1e-9, \
            f"NC cache alpha_gate={data['alpha_gate']} != ALPHA_GATE={ALPHA_GATE}"
        self.mode = mode
        self.nc_scores: dict[str, float] = data["nc_scores"]
        self.taus: dict[str, float] = data["per_cell_taus"]
        self.cell_counts: dict[str, int] = data["per_cell_counts"]
        self._activation = Counter()
        self._seen = Counter()

    def __call__(self, sample, n_match: int) -> bool:
        cid = get_cell_id(sample)
        self._seen[cid] += 1
        if self.mode == "r3_always_on":
            fire = cid in WORST_CELLS and n_match >= 1
        elif self.mode == "r4_depth_token":
            fire = cid in WORST_CELLS and n_match >= 1
        else:  # r2_selective
            if cid not in WORST_CELLS or n_match < 1:
                fire = False
            else:
                s = self.nc_scores.get(sample["id"])
                tau = self.taus.get(cid)
                fire = s is not None and tau is not None and s >= tau
        if fire:
            self._activation[cid] += 1
        return fire

    def activation_summary(self) -> dict:
        out = {
            "gate/mode": self.mode,
            "gate/alpha_gate": ALPHA_GATE,
        }
        total_seen = sum(self._seen.values())
        total_fired = sum(self._activation.values())
        out["gate/activation_rate_all"] = total_fired / total_seen if total_seen else 0.0
        hard_seen = sum(self._seen[c] for c in WORST_CELLS)
        hard_fired = sum(self._activation[c] for c in WORST_CELLS)
        out["gate/activation_rate_hard4"] = hard_fired / hard_seen if hard_seen else 0.0
        for cell in WORST_CELLS:
            seen = self._seen[cell]
            fired = self._activation[cell]
            rate = fired / seen if seen else 0.0
            out[f"gate/activation_rate_cell/{cell}"] = rate
            out[f"gate/cell_seen/{cell}"] = seen
            out[f"gate/cell_fired/{cell}"] = fired
        return out

    def reset_window(self):
        self._activation = Counter()
        self._seen = Counter()


# ---------------------------------------------------------------------------
# Depth + visual feature lookup for matched objects
# ---------------------------------------------------------------------------

def load_cached_depth(sample) -> np.ndarray | None:
    img_id = Path(sample["image_path"]).stem
    npy = DEPTH_NP_CACHE / f"{img_id}.npy"
    if not npy.exists():
        return None
    return np.load(npy)


def box_coords_normalized(bbox: tuple[int, int, int, int], img_wh: tuple[int, int]) -> np.ndarray:
    x, y, w, h = bbox
    W, H = img_wh
    x1 = max(0.0, min(1.0, x / W))
    y1 = max(0.0, min(1.0, y / H))
    x2 = max(0.0, min(1.0, (x + w) / W))
    y2 = max(0.0, min(1.0, (y + h) / H))
    area = max(0.0, (x2 - x1) * (y2 - y1))
    return np.array([x1, y1, x2, y2, area], dtype=np.float32)


def box_pooled_visual(vision_tower_features: torch.Tensor, bbox, img_wh, feat_hw) -> torch.Tensor:
    """Average-pool vision-tower patch features that fall inside a pixel bbox.

    Args:
        vision_tower_features: (H_p, W_p, d_vis) patch features.
        bbox: (x, y, w, h) pixel coordinates.
        img_wh: (W_img, H_img) image pixel size.
        feat_hw: (H_p, W_p) number of vision patches.
    Returns:
        (d_vis,) average-pooled feature.
    """
    W_img, H_img = img_wh
    H_p, W_p = feat_hw
    x, y, w, h = bbox
    x1 = int(np.floor(x / W_img * W_p))
    y1 = int(np.floor(y / H_img * H_p))
    x2 = int(np.ceil((x + w) / W_img * W_p))
    y2 = int(np.ceil((y + h) / H_img * H_p))
    x1, x2 = max(0, x1), min(W_p, x2)
    y1, y2 = max(0, y1), min(H_p, y2)
    if x2 <= x1 or y2 <= y1:
        return vision_tower_features.reshape(-1, vision_tower_features.shape[-1]).mean(0)
    region = vision_tower_features[y1:y2, x1:x2]
    return region.reshape(-1, region.shape[-1]).mean(0)


# ---------------------------------------------------------------------------
# Slot-path forward injection — TODO after R1 validation
# ---------------------------------------------------------------------------

def forward_with_slots(model, hook_mgr, slot_mod, adapter,
                       base_inputs, labels,
                       slot_visual, slot_boxes, slot_depths, slot_mask):
    """Forward pass with Strategy B slot injection via the hook manager.

    Hook context is always cleared via finally, even if model() throws.
    """
    slots, pair_logits = slot_mod(slot_visual, slot_boxes, slot_depths, slot_mask)
    hook_mgr.set_slot_context(slots, slot_mask)
    try:
        outputs = model(**base_inputs, labels=labels)
    finally:
        hook_mgr.clear_slot_context()
    return outputs.loss, pair_logits


def forward_ungated(model, base_inputs, labels):
    """Exact R1 forward path — no hook context set, hook is a no-op."""
    outputs = model(**base_inputs, labels=labels)
    return outputs.loss


def evaluate_loss_with_slots(model, processor, process_vision_info, eval_samples,
                             partition, hook_mgr, slot_mod, adapter, gate,
                             raw_questions, scene_graphs,
                             _captured_pre_merger, d_vis, device,
                             is_depth_token: bool = False):
    """Eval that mirrors training inference: slot path active on gated samples."""
    from scripts.pilot_depth_object_level import tokenize_train_example, compute_object_depth
    model.eval()
    slot_mod.eval()
    adapter.eval()
    cell_losses = defaultdict(list)
    n_gated = 0
    n_ungated = 0
    pair_correct = 0
    pair_total = 0

    gate.reset_window()

    for i, s in enumerate(eval_samples):
        cid = partition.get_cell_by_features(s)
        if cid is None:
            continue
        try:
            inputs, labels = tokenize_train_example(processor, s, process_vision_info)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            qid = resolve_raw_qid(s.get("id", ""))
            raw = raw_questions.get(qid)
            matched = []
            if raw is not None:
                img_id = raw.get("imageId")
                sg_objs = scene_graphs.get(img_id, {}).get("objects", {})
                matched = match_named_objects(raw, sg_objs)

            n_match = len(matched)
            gate_on = gate(s, n_match)

            if gate_on and n_match >= 1:
                img = load_image(s["image_path"])
                img_wh = img.size
                depth_np_arr = load_cached_depth(s)

                sb_l, sd_l, d_vals = [], [], []
                for obj in matched:
                    sb_l.append(torch.tensor(box_coords_normalized(obj["bbox"], img_wh), device=device))
                    d_val = compute_object_depth(depth_np_arr, obj["bbox"]) if depth_np_arr is not None else 0.0
                    d_val = d_val if d_val is not None else 0.0
                    sd_l.append(torch.tensor([d_val], dtype=torch.bfloat16, device=device))
                    d_vals.append(d_val)
                sb = torch.stack(sb_l).unsqueeze(0).to(dtype=torch.bfloat16)
                sd = torch.stack(sd_l).unsqueeze(0)
                sm = torch.ones(1, n_match, dtype=torch.bool, device=device)

                built = False
                try:
                    if is_depth_token:
                        with torch.no_grad():
                            tokens, pair_logits = slot_mod(sb, sd, sm)
                        hook_mgr.set_slot_context(tokens, sm)
                        with torch.no_grad():
                            out = model(**inputs, labels=labels)
                        cell_losses[cid].append(out.loss.item())
                        n_gated += 1
                        built = True
                    else:
                        hook_mgr.clear_slot_context()
                        with torch.no_grad():
                            _ = model(**inputs)
                        pre_merger = _captured_pre_merger.get("feat")
                        grid_thw = inputs.get("image_grid_thw")
                        if pre_merger is not None and grid_thw is not None:
                            t_g, h_g, w_g = grid_thw[0].tolist()
                            n_grid = int(t_g * h_g * w_g)
                            if t_g == 1 and pre_merger.shape[0] >= n_grid:
                                patch_grid = pre_merger[:n_grid].reshape(int(h_g), int(w_g), d_vis)
                                sv_l = []
                                for obj in matched:
                                    pf = box_pooled_visual(patch_grid, obj["bbox"], img_wh, (int(h_g), int(w_g)))
                                    sv_l.append(pf)
                                sv = torch.stack(sv_l).unsqueeze(0).to(dtype=torch.bfloat16)
                                with torch.no_grad():
                                    loss_qa, pair_logits = forward_with_slots(
                                        model, hook_mgr, slot_mod, adapter,
                                        inputs, labels, sv, sb, sd, sm)
                                cell_losses[cid].append(loss_qa.item())
                                n_gated += 1
                                built = True
                finally:
                    hook_mgr.clear_slot_context()

                if built:
                    K = len(d_vals)
                    if K >= 2:
                        dv = torch.tensor(d_vals, device=device)
                        i_idx, j_idx = torch.meshgrid(
                            torch.arange(K, device=device),
                            torch.arange(K, device=device), indexing="ij")
                        pm = i_idx != j_idx
                        targets = (dv[i_idx[pm]] < dv[j_idx[pm]]).float()
                        preds = (pair_logits.squeeze(0) > 0).float()
                        pair_correct += int((preds == targets).sum().item())
                        pair_total += int(targets.numel())

                if not built:
                    with torch.no_grad():
                        out = model(**inputs, labels=labels)
                    cell_losses[cid].append(out.loss.item())
                    n_ungated += 1
            else:
                hook_mgr.clear_slot_context()
                with torch.no_grad():
                    out = model(**inputs, labels=labels)
                cell_losses[cid].append(out.loss.item())
                n_ungated += 1

        except Exception:
            hook_mgr.clear_slot_context()
            cell_losses[cid].append(10.0)

        if (i + 1) % 100 == 0:
            print(f"    eval: {i+1}/{len(eval_samples)}")

    cell_mean = {cid: float(np.mean(ls)) for cid, ls in cell_losses.items()
                 if len(ls) >= MIN_CELL_SUPPORT}
    if cell_mean:
        sorted_vals = sorted(cell_mean.values(), reverse=True)
        k = max(1, int(np.ceil(len(sorted_vals) * 0.1)))
        w10_cvar = float(np.mean(sorted_vals[:k]))
        worst_cell = float(max(cell_mean.values()))
    else:
        w10_cvar = worst_cell = 1.0
    overall = float(np.mean([l for ls in cell_losses.values() for l in ls]))
    pair_acc = pair_correct / pair_total if pair_total > 0 else 0.0

    model.train()
    slot_mod.train()
    adapter.train()

    return {
        "worst_10pct_cvar": w10_cvar,
        "worst_cell_loss": worst_cell,
        "overall_loss": overall,
        "n_cells": len(cell_mean),
        "cell_losses": cell_mean,
        "n_gated": n_gated,
        "n_ungated": n_ungated,
        "pair_aux_acc": pair_acc,
        "pair_total": pair_total,
        "gate_summary": gate.activation_summary(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_shared_config(path: Path) -> dict:
    data = json.loads(path.read_text())
    required = {
        "model_id", "lora_r", "lora_alpha", "lora_dropout", "target_modules",
        "lr", "weight_decay", "max_grad_norm", "micro_batch_size",
        "grad_accum_steps", "epochs", "aux_ratio", "aux_weight", "seed",
        "train_split_size", "eval_split_size",
    }
    missing = required - set(data)
    assert not missing, f"refine-logs/r1_config.json missing keys: {missing}"
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["r2_selective", "r3_always_on", "r4_depth_token"])
    parser.add_argument("--shared_config", type=str, default=SHARED_CONFIG_PATH_DEFAULT)
    parser.add_argument("--nc_cache", type=str, default=NC_CACHE_PATH_DEFAULT)
    parser.add_argument("--run_tag", type=str, required=True,
                        help="Suffix appended to run dir, e.g. 'r2_selective'.")
    parser.add_argument("--nc_cache_eval", type=str,
                        default="refine-logs/nc_cache_qwen3vl8b_r1_seed1_repair_val.json",
                        help="NC cache for the eval split (repair_val). Uses train-calibrated taus.")
    parser.add_argument("--pair_aux_weight", type=float, default=PAIR_AUX_WEIGHT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from shared config (rarely needed).")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Early stop after this many optimizer steps (for dry runs).")
    args = parser.parse_args()

    shared = load_shared_config(Path(args.shared_config))
    assert shared["model_id"] == MODEL_ID, \
        f"Shared config model_id={shared['model_id']} but pilot is pinned to {MODEL_ID}"

    print("=" * 70)
    print(f"R2-family pilot: {args.mode}")
    print(f"Shared config   : {args.shared_config}")
    print(f"NC cache        : {args.nc_cache}")
    print(f"Pair aux weight : {args.pair_aux_weight}")
    print(f"ALPHA_GATE      : {ALPHA_GATE}   (separate from ALPHA_CVAR={ALPHA_CVAR})")
    print("=" * 70)

    # ---- Data loading (mirrors R1 exactly) --------------------------------
    train_samples = load_worst_cell_samples("train")
    eval_samples = load_worst_cell_samples("repair_val")
    print(f"train={len(train_samples)} eval={len(eval_samples)}")

    # ---- NC cache + gate --------------------------------------------------
    gate = CPGate(Path(args.nc_cache), mode=args.mode)
    print("tau_h per cell:")
    for cell, tau in gate.taus.items():
        print(f"  {cell:<28} tau={tau:.4f}  |cell|={gate.cell_counts.get(cell, '?')}")

    # ---- GQA raw data (for named-object matching) -------------------------
    with open(GQA_QUESTIONS_FILE) as f:
        raw_questions = json.load(f)
    with open(GQA_SCENE_GRAPHS) as f:
        scene_graphs = json.load(f)
    print(f"GQA raw: {len(raw_questions)} questions, {len(scene_graphs)} scene graphs")

    # ---- Model + LoRA (mirrors R1 exactly) ----------------------------------
    from transformers import (Qwen3VLForConditionalGeneration, AutoProcessor,
                              BitsAndBytesConfig, get_cosine_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model, TaskType
    from qwen_vl_utils import process_vision_info
    from scripts.pilot_depth_object_level import (
        tokenize_train_example, evaluate_loss, generate_object_level_aux,
        generate_depth_maps, load_depth_np,
    )
    from src.diagnosis.mondrian_partition import MondrianPartition
    import torch.nn.functional as F
    from datetime import datetime

    seed = shared["seed"]
    epochs = args.epochs or shared["epochs"]
    rng = np.random.default_rng(seed + 42)
    torch.manual_seed(seed + 42)
    np.random.seed(seed + 42)

    OUT_DIR = Path("results/sprint2/pilots")
    model_tag = MODEL_ID.split("/")[-1].lower().replace("-instruct", "").replace("-", "")
    run_dir = OUT_DIR / f"d1_slots_{args.run_tag}_seed{seed}_{model_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {MODEL_ID} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    d_vis = model.config.vision_config.hidden_size
    d_model = model.config.text_config.hidden_size
    print(f"  d_vis={d_vis}  d_model={d_model}")
    assert d_model == D_MODEL_QWEN3VL_8B

    # ---- Slot / depth-token module + adapter + hook -------------------------
    device = next(model.parameters()).device
    is_depth_token = (args.mode == "r4_depth_token")
    if is_depth_token:
        slot_mod = DepthTokenModule(d_slot=D_SLOT).to(device, dtype=torch.bfloat16)
        print(f"  [R4] Using DepthTokenModule (no vision features, no slot-attn)")
    else:
        slot_mod = D1SlotModule(d_vis=d_vis, d_model=d_model).to(device, dtype=torch.bfloat16)
    adapter = SlotCrossAttentionAdapter(d_model=d_model, d_slot=D_SLOT).to(device, dtype=torch.bfloat16)
    hook_mgr = SlotHookManager(model, adapter).register()
    norm_name = type(hook_mgr._find_final_norm()).__name__
    print(f"  slot_mod params:  {slot_mod.count_trainable_parameters():,}")
    print(f"  adapter params:   {adapter.count_trainable_parameters():,}")
    print(f"  combined new:     {slot_mod.count_trainable_parameters() + adapter.count_trainable_parameters():,}")
    print(f"  gate_scalar init: {adapter.gate_scalar.item():.2f} -> sigmoid={torch.sigmoid(adapter.gate_scalar).item():.6f}")
    print(f"  hook norm: {norm_name}")

    # ---- Vision-merger hook for per-sample patch features ------------------
    _captured_pre_merger = {}
    if hasattr(model, "base_model"):
        _visual = model.base_model.model.model.visual
    else:
        _visual = model.model.visual
    _merger_hook = _visual.merger.register_forward_hook(
        lambda mod, inp, out: _captured_pre_merger.update({"feat": inp[0].detach()})
    )

    # ---- Aux data generation (mirrors R1) ---------------------------------
    all_samples = train_samples + eval_samples
    generate_depth_maps(all_samples, device=str(device))
    aux_train, aux_stats = generate_object_level_aux(
        train_samples, raw_questions, scene_graphs, rng)
    for s in train_samples:
        s["is_auxiliary"] = False
    n_aux_target = int(len(train_samples) * shared["aux_ratio"])
    if len(aux_train) > n_aux_target:
        idx = rng.choice(len(aux_train), n_aux_target, replace=False)
        aux_train = [aux_train[int(i)] for i in idx]
    combined_train = train_samples + aux_train
    print(f"  Combined: {len(combined_train)} ({len(train_samples)} orig + {len(aux_train)} aux)")

    partition = MondrianPartition.load(Path("results/sprint2/b1_diagnosis/partition.json"))

    # ---- Optimizer (LoRA + slot_mod + adapter) ----------------------------
    n_combined = len(combined_train)
    steps_per_epoch = n_combined // GRAD_ACCUM
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    all_params = (list(model.parameters()) +
                  list(slot_mod.parameters()) +
                  list(adapter.parameters()))
    optimizer = torch.optim.AdamW(
        all_params, lr=shared["lr"], weight_decay=shared["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"  Steps/epoch: {steps_per_epoch}, total: {total_steps}")

    # ---- Eval gate (repair_val NC cache, train-calibrated taus) -------------
    eval_gate = CPGate(Path(args.nc_cache_eval), mode=args.mode)
    print(f"Eval gate loaded from {args.nc_cache_eval}")

    # ---- Pre-training eval (slot-aware) -----------------------------------
    print("\nPre-training eval (slot-aware)...")
    pre_eval = evaluate_loss_with_slots(
        model, processor, process_vision_info, eval_samples, partition,
        hook_mgr, slot_mod, adapter, eval_gate,
        raw_questions, scene_graphs, _captured_pre_merger, d_vis, device,
        is_depth_token=is_depth_token)
    print(f"  W10%CVaR: {pre_eval['worst_10pct_cvar']:.4f}  worst_cell: {pre_eval['worst_cell_loss']:.4f}")
    print(f"  pair_aux_acc: {pre_eval['pair_aux_acc']:.4f}  gated/ungated: {pre_eval['n_gated']}/{pre_eval['n_ungated']}")

    # ---- Training loop ----------------------------------------------------
    model.train()
    slot_mod.train()
    adapter.train()
    best_metric = float("inf")
    best_epoch = 0
    epoch_history = []
    gate_history = []
    n_gated_total = 0
    n_ungated_total = 0

    train_start = time.time()
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{epochs}\n{'='*60}")
        indices = rng.permutation(n_combined)
        optimizer.zero_grad()
        epoch_loss_orig = 0.0
        epoch_loss_aux = 0.0
        epoch_loss_pair = 0.0
        n_orig = n_aux_done = n_steps = n_gated_epoch = 0

        gate.reset_window()

        for micro_step in range(n_combined):
            idx = int(indices[micro_step])
            sample = combined_train[idx]

            try:
                # Tokenize (identical to R1)
                inputs, labels = tokenize_train_example(
                    processor, sample, process_vision_info)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)

                # Named-object matching (skip for auxiliary samples)
                matched = []
                if not sample.get("is_auxiliary"):
                    qid = resolve_raw_qid(sample.get("id", ""))
                    raw = raw_questions.get(qid)
                    if raw is not None:
                        img_id = raw.get("imageId")
                        sg_objs = scene_graphs.get(img_id, {}).get("objects", {})
                        matched = match_named_objects(raw, sg_objs)

                n_match = len(matched)
                gate_on = gate(sample, n_match) if not sample.get("is_auxiliary") else False

                if gate_on and n_match >= 1:
                    # --- GATED PATH ---
                    img = load_image(sample["image_path"])
                    img_wh = img.size
                    depth_np = load_depth_np(sample)

                    sb_list, sd_list = [], []
                    for obj in matched:
                        sb_list.append(torch.tensor(
                            box_coords_normalized(obj["bbox"], img_wh), device=device))
                        d_val = compute_object_depth(depth_np, obj["bbox"]) if depth_np is not None else 0.0
                        d_val = d_val if d_val is not None else 0.0
                        sd_list.append(torch.tensor([d_val], dtype=torch.bfloat16, device=device))
                    sb = torch.stack(sb_list).unsqueeze(0).to(dtype=torch.bfloat16)
                    sd = torch.stack(sd_list).unsqueeze(0)
                    sm = torch.ones(1, n_match, dtype=torch.bool, device=device)

                    repair_active = False
                    if is_depth_token:
                        # R4: DepthTokenModule — no vision features, no merger hook
                        tokens, pair_logits = slot_mod(sb, sd, sm)
                        hook_mgr.set_slot_context(tokens, sm)
                        try:
                            outputs = model(**inputs, labels=labels)
                        finally:
                            hook_mgr.clear_slot_context()
                        loss_qa = outputs.loss
                        repair_active = True
                    else:
                        # R2/R3: D1SlotModule — needs vision features from merger hook
                        hook_mgr.clear_slot_context()
                        with torch.no_grad():
                            _ = model(**inputs)
                        pre_merger = _captured_pre_merger.get("feat")
                        grid_thw = inputs.get("image_grid_thw")
                        if pre_merger is not None and grid_thw is not None:
                            t_g, h_g, w_g = grid_thw[0].tolist()
                            n_grid = int(t_g * h_g * w_g)
                            if t_g == 1 and pre_merger.shape[0] >= n_grid:
                                patch_grid = pre_merger[:n_grid].reshape(int(h_g), int(w_g), d_vis)
                                sv_list = []
                                for obj in matched:
                                    pf = box_pooled_visual(
                                        patch_grid, obj["bbox"], img_wh, (int(h_g), int(w_g)))
                                    sv_list.append(pf)
                                sv = torch.stack(sv_list).unsqueeze(0).to(dtype=torch.bfloat16)
                                loss_qa, pair_logits = forward_with_slots(
                                    model, hook_mgr, slot_mod, adapter,
                                    inputs, labels, sv, sb, sd, sm)
                                repair_active = True
                        if not repair_active:
                            loss_qa = forward_ungated(model, inputs, labels)
                            pair_logits = None

                    if repair_active:
                        n_gated_epoch += 1

                    # Pair-order aux loss
                    if pair_logits is not None:
                        depths_vec = sd.squeeze(0).squeeze(-1)
                        K = depths_vec.shape[0]
                        if K >= 2:
                            i_idx, j_idx = torch.meshgrid(
                                torch.arange(K, device=device),
                                torch.arange(K, device=device), indexing="ij")
                            pair_mask = i_idx != j_idx
                            di = depths_vec[i_idx[pair_mask]]
                            dj = depths_vec[j_idx[pair_mask]]
                            pair_targets = (di < dj).float()
                            loss_pair = F.binary_cross_entropy_with_logits(
                                pair_logits.squeeze(0), pair_targets)
                            epoch_loss_pair += loss_pair.item()
                        else:
                            loss_pair = torch.tensor(0.0, device=device)
                        loss = loss_qa + args.pair_aux_weight * loss_pair
                    else:
                        loss = loss_qa
                else:
                    # --- UNGATED PATH: exact R1 behavior ---
                    loss = forward_ungated(model, inputs, labels)

                # Apply aux weight (same as R1)
                if sample.get("is_auxiliary"):
                    scaled_loss = loss * shared["aux_weight"] / GRAD_ACCUM
                    epoch_loss_aux += loss.item()
                    n_aux_done += 1
                else:
                    scaled_loss = loss / GRAD_ACCUM
                    epoch_loss_orig += loss.item()
                    n_orig += 1

                scaled_loss.backward()

            except Exception as e:
                if micro_step < 5:
                    print(f"  Error at step {micro_step}: {e}")
                    import traceback; traceback.print_exc()
                continue

            if (micro_step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(all_params, MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                n_steps += 1

                if n_steps % max(steps_per_epoch // 10, 1) == 0:
                    avg_orig = epoch_loss_orig / max(n_orig, 1)
                    avg_aux = epoch_loss_aux / max(n_aux_done, 1)
                    avg_pair = epoch_loss_pair / max(n_gated_epoch, 1)
                    gs = torch.sigmoid(adapter.gate_scalar).item()
                    print(f"  Step {n_steps}/{steps_per_epoch} | "
                          f"orig={avg_orig:.4f} aux={avg_aux:.4f} pair={avg_pair:.4f} | "
                          f"gated={n_gated_epoch} gate_sig={gs:.4f}")

                # Per-cell gate logging every 50 steps
                if n_steps % 50 == 0:
                    summary = gate.activation_summary()
                    summary["train/gate_scalar_sigmoid"] = torch.sigmoid(adapter.gate_scalar).item()
                    gate_history.append({"step": n_steps, "epoch": epoch, **summary})
                    gate.reset_window()

                if args.max_steps and n_steps >= args.max_steps:
                    print(f"\n  [dry run] max_steps={args.max_steps} reached, stopping.")
                    break

            if args.max_steps and n_steps >= args.max_steps:
                break

        n_gated_total += n_gated_epoch
        n_ungated_total += (n_orig - n_gated_epoch)

        avg_orig_loss = epoch_loss_orig / max(n_orig, 1)
        avg_aux_loss = epoch_loss_aux / max(n_aux_done, 1)
        print(f"\n  Epoch {epoch} | orig={avg_orig_loss:.4f} aux={avg_aux_loss:.4f} "
              f"gated={n_gated_epoch}/{n_orig}")

        # Eval (slot-aware)
        print("  Evaluating (slot-aware)...")
        eval_gate.reset_window()
        eval_result = evaluate_loss_with_slots(
            model, processor, process_vision_info, eval_samples, partition,
            hook_mgr, slot_mod, adapter, eval_gate,
            raw_questions, scene_graphs, _captured_pre_merger, d_vis, device,
            is_depth_token=is_depth_token)
        w10 = eval_result["worst_10pct_cvar"]
        wc = eval_result["worst_cell_loss"]
        gs = torch.sigmoid(adapter.gate_scalar).item()
        print(f"  W10%CVaR={w10:.4f}  worst_cell={wc:.4f}  gate_sig={gs:.4f}")
        print(f"  pair_aux_acc={eval_result['pair_aux_acc']:.4f}  gated/ungated={eval_result['n_gated']}/{eval_result['n_ungated']}")
        for cid, l in sorted(eval_result["cell_losses"].items()):
            print(f"    {cid}: {l:.4f}")

        epoch_history.append({
            "epoch": epoch, "train_loss_orig": avg_orig_loss,
            "train_loss_aux": avg_aux_loss, "n_gated": n_gated_epoch,
            "gate_scalar_sigmoid": gs,
            "eval_pair_aux_acc": eval_result["pair_aux_acc"],
            "eval_n_gated": eval_result["n_gated"],
            "eval_n_ungated": eval_result["n_ungated"],
            "eval": eval_result,
        })

        if w10 < best_metric:
            best_metric = w10
            best_epoch = epoch
            ckpt_dir = run_dir / "checkpoint-best"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
            torch.save(slot_mod.state_dict(), ckpt_dir / "slot_mod.pt")
            torch.save(adapter.state_dict(), ckpt_dir / "adapter.pt")
            print(f"  * New best: W10%CVaR={w10:.4f} (epoch {epoch})")

    elapsed_h = (time.time() - train_start) / 3600

    # ---- Save training results --------------------------------------------
    best_eval = epoch_history[best_epoch - 1]["eval"]
    output = {
        "pilot": f"d1_slots_{args.mode}",
        "mode": args.mode,
        "model": MODEL_ID,
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "elapsed_hours": elapsed_h,
        "alpha_gate": ALPHA_GATE,
        "alpha_cvar": ALPHA_CVAR,
        "pair_aux_weight": args.pair_aux_weight,
        "slot_mod_params": slot_mod.count_trainable_parameters(),
        "adapter_params": adapter.count_trainable_parameters(),
        "n_gated_total": n_gated_total,
        "n_ungated_total": n_ungated_total,
        "hook_norm": norm_name,
        "pre_training_eval": pre_eval,
        "epoch_history": epoch_history,
        "best_eval": best_eval,
        "gate_history": gate_history,
        "run_dir": str(run_dir),
        "nc_cache": args.nc_cache,
        "shared_config": args.shared_config,
        "timestamp": datetime.now().isoformat(),
    }
    (run_dir / "d1_slots_results.json").write_text(json.dumps(output, indent=2))

    print(f"\n{'='*60}")
    print(f"Training DONE — {args.mode}")
    print(f"  best epoch: {best_epoch}, W10%CVaR(loss): {best_eval['worst_10pct_cvar']:.4f}")
    print(f"  elapsed: {elapsed_h:.2f}h")
    print(f"{'='*60}")

    # ---- Auto generation eval on best checkpoint --------------------------
    # Restore best-epoch state for slot_mod + adapter, re-register hook.
    ckpt_dir = run_dir / "checkpoint-best"
    print(f"\nAuto generation eval on best checkpoint (epoch {best_epoch})...")
    slot_mod.load_state_dict(torch.load(ckpt_dir / "slot_mod.pt", map_location=device, weights_only=True))
    adapter.load_state_dict(torch.load(ckpt_dir / "adapter.pt", map_location=device, weights_only=True))
    # hook_mgr is still registered with the same adapter instance — weights updated in-place.

    from scripts.eval_d1_generation import run_generation_eval
    eval_gate.reset_window()
    gen_result = run_generation_eval(
        model, processor, process_vision_info, eval_samples, partition,
        hook_mgr, slot_mod, adapter, eval_gate,
        raw_questions, scene_graphs, _captured_pre_merger, d_vis, device,
        label=f"GENERATION EVAL — {args.mode} (best epoch {best_epoch})",
        is_depth_token=is_depth_token,
    )
    gen_result["checkpoint"] = str(ckpt_dir)
    gen_result["mode"] = args.mode
    gen_result["best_epoch"] = best_epoch
    gen_result["training_elapsed_hours"] = elapsed_h

    gen_out_path = run_dir / "eval_generation_results.json"
    gen_out_path.write_text(json.dumps(gen_result, indent=2))
    print(f"  Saved -> {gen_out_path}")

    _merger_hook.remove()
    hook_mgr.remove()

    # ---- Final summary (generation-based, the formal metric) --------------
    print(f"\n{'='*60}")
    print(f"FINAL — {args.mode}")
    print(f"  Generation worst_cell_error: {gen_result['worst_cell_error']:.4f}")
    print(f"  Generation overall_accuracy: {gen_result['overall_accuracy']:.4f}")
    print(f"  W10%CVaR(error):             {gen_result['worst_10pct_cvar_error']:.4f}")
    print(f"  Gated/ungated:               {gen_result['n_gated']}/{gen_result['n_ungated']}")
    print(f"  gate_scalar sigmoid:         {gen_result['gate_scalar_sigmoid']:.4f}")
    print(f"  saved to {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
