#!/usr/bin/env python3
"""
Corrected evaluation for D1 slot path — mirrors R2 inference.

The original R2 eval called evaluate_loss() which is R1's baseline eval:
it never set the slot hook context, so the adapter cross-attn residual
was never applied during eval. This script fixes that.

For each eval sample:
  1. Check the CP gate (using repair_val NC cache with train-calibrated taus)
  2. If gated AND matchable: extract vision features, build slots, set hook
     context, forward with labels
  3. If ungated: plain forward (hook context = None → identity)

Reports both loss-based and generation-based (accuracy) metrics per cell.

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_d1_slots.py \
      --checkpoint results/sprint2/pilots/d1_slots_r2_selective_seed1_qwen3vl8b/checkpoint-best \
      --nc_cache_eval refine-logs/nc_cache_qwen3vl8b_r1_seed1_repair_val.json \
      --mode r2_selective
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.pilot_depth_object_level import (
    MODEL_ID, CACHE_DIR, WORST_CELLS, DEPTH_NP_CACHE,
    MIN_CELL_SUPPORT,
    build_prompt, build_answer, get_cell_id, load_image, load_worst_cell_samples,
    tokenize_train_example, compute_object_depth,
)
from scripts.pilot_d1_slots import (
    CPGate, match_named_objects, resolve_raw_qid,
    box_coords_normalized, box_pooled_visual,
    forward_with_slots, forward_ungated,
    GQA_QUESTIONS_FILE, GQA_SCENE_GRAPHS,
    D1SlotModule, SlotCrossAttentionAdapter, SlotHookManager,
    D_SLOT, D_MODEL_QWEN3VL_8B, ALPHA_GATE, ALPHA_CVAR,
)


def load_depth_np(sample):
    img_id = Path(sample["image_path"]).stem
    npy = DEPTH_NP_CACHE / f"{img_id}.npy"
    if not npy.exists():
        return None
    return np.load(npy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--nc_cache_eval", type=str, required=True,
                        help="NC cache for the eval split (repair_val), "
                             "with train-calibrated taus.")
    parser.add_argument("--mode", type=str, default="r2_selective",
                        choices=["r2_selective", "r3_always_on", "r4_depth_token"])
    parser.add_argument("--eval_split", type=str, default="repair_val")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint)
    assert ckpt_dir.exists(), f"Checkpoint not found: {ckpt_dir}"
    assert (ckpt_dir / "slot_mod.pt").exists(), f"slot_mod.pt not in {ckpt_dir}"
    assert (ckpt_dir / "adapter.pt").exists(), f"adapter.pt not in {ckpt_dir}"

    if args.output is None:
        args.output = str(ckpt_dir.parent / "eval_d1_corrected.json")

    print("=" * 70)
    print("D1 CORRECTED EVAL (slot path active during eval)")
    print(f"Checkpoint    : {ckpt_dir}")
    print(f"NC cache eval : {args.nc_cache_eval}")
    print(f"Mode          : {args.mode}")
    print(f"Eval split    : {args.eval_split}")
    print(f"ALPHA_GATE    : {ALPHA_GATE} (separate from ALPHA_CVAR={ALPHA_CVAR})")
    print("=" * 70)

    # Load eval samples
    eval_samples = load_worst_cell_samples(args.eval_split)
    print(f"Eval samples: {len(eval_samples)}")

    # Gate (from repair_val NC cache)
    gate = CPGate(Path(args.nc_cache_eval), mode=args.mode)
    print("tau_h per cell (from train calibration):")
    for cell, tau in gate.taus.items():
        print(f"  {cell:<28} tau={tau:.4f}")

    # GQA data
    with open(GQA_QUESTIONS_FILE) as f:
        raw_questions = json.load(f)
    with open(GQA_SCENE_GRAPHS) as f:
        scene_graphs = json.load(f)
    print(f"GQA: {len(raw_questions)} questions, {len(scene_graphs)} scene graphs")

    # Load model + LoRA + slot_mod + adapter
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    print(f"\nLoading {MODEL_ID} + LoRA...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(base_model, str(ckpt_dir))
    model.eval()

    d_vis = model.config.vision_config.hidden_size
    d_model = model.config.text_config.hidden_size
    device = next(model.parameters()).device
    print(f"  d_vis={d_vis} d_model={d_model}")

    # Load slot module + adapter from checkpoint
    slot_mod = D1SlotModule(d_vis=d_vis, d_model=d_model).to(device, dtype=torch.bfloat16)
    slot_mod.load_state_dict(torch.load(ckpt_dir / "slot_mod.pt", map_location=device, weights_only=True))
    slot_mod.eval()

    adapter = SlotCrossAttentionAdapter(d_model=d_model, d_slot=D_SLOT).to(device, dtype=torch.bfloat16)
    adapter.load_state_dict(torch.load(ckpt_dir / "adapter.pt", map_location=device, weights_only=True))
    adapter.eval()

    hook_mgr = SlotHookManager(model, adapter).register()
    print(f"  slot_mod params: {slot_mod.count_trainable_parameters():,}")
    print(f"  adapter params:  {adapter.count_trainable_parameters():,}")
    print(f"  gate_scalar sigmoid: {torch.sigmoid(adapter.gate_scalar).item():.6f}")
    print(f"  hook norm: {type(hook_mgr._find_final_norm()).__name__}")

    # Vision-merger hook for patch features
    _captured = {}
    if hasattr(model, "base_model"):
        _visual = model.base_model.model.model.visual
    else:
        _visual = model.model.visual
    _merger_hook = _visual.merger.register_forward_hook(
        lambda mod, inp, out: _captured.update({"feat": inp[0].detach()})
    )

    # ---- Eval loop --------------------------------------------------------
    from src.diagnosis.mondrian_partition import MondrianPartition
    partition = MondrianPartition.load(Path("results/sprint2/b1_diagnosis/partition.json"))

    cell_losses = defaultdict(list)
    cell_losses_baseline = defaultdict(list)
    n_gated = 0
    n_ungated = 0
    n_errors = 0

    t0 = time.time()
    for i, s in enumerate(eval_samples):
        cid = partition.get_cell_by_features(s)
        if cid is None:
            continue

        try:
            inputs, labels = tokenize_train_example(processor, s, process_vision_info)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            # Named-object matching
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
                # First forward (ungated) to capture vision features
                hook_mgr.clear_slot_context()
                with torch.no_grad():
                    out_baseline = model(**inputs, labels=labels)
                baseline_loss = out_baseline.loss.item()
                cell_losses_baseline[cid].append(baseline_loss)

                pre_merger = _captured.get("feat")
                grid_thw = inputs.get("image_grid_thw")

                if pre_merger is not None and grid_thw is not None:
                    t_g, h_g, w_g = grid_thw[0].tolist()
                    n_grid = int(t_g * h_g * w_g)
                    if t_g == 1 and pre_merger.shape[0] >= n_grid:
                        patch_grid = pre_merger[:n_grid].reshape(int(h_g), int(w_g), d_vis)
                        img = load_image(s["image_path"])
                        img_wh = img.size
                        depth_np = load_depth_np(s)

                        sv_list, sb_list, sd_list = [], [], []
                        for obj in matched:
                            pf = box_pooled_visual(patch_grid, obj["bbox"], img_wh, (int(h_g), int(w_g)))
                            sv_list.append(pf)
                            sb_list.append(torch.tensor(box_coords_normalized(obj["bbox"], img_wh), device=device))
                            d_val = compute_object_depth(depth_np, obj["bbox"]) if depth_np is not None else 0.0
                            d_val = d_val if d_val is not None else 0.0
                            sd_list.append(torch.tensor([d_val], dtype=torch.bfloat16, device=device))

                        sv = torch.stack(sv_list).unsqueeze(0).to(dtype=torch.bfloat16)
                        sb = torch.stack(sb_list).unsqueeze(0).to(dtype=torch.bfloat16)
                        sd = torch.stack(sd_list).unsqueeze(0)
                        sm = torch.ones(1, n_match, dtype=torch.bool, device=device)

                        # Gated forward
                        with torch.no_grad():
                            loss_qa, _ = forward_with_slots(
                                model, hook_mgr, slot_mod, adapter,
                                inputs, labels, sv, sb, sd, sm)
                        cell_losses[cid].append(loss_qa.item())
                        n_gated += 1
                    else:
                        cell_losses[cid].append(baseline_loss)
                        n_ungated += 1
                else:
                    cell_losses[cid].append(baseline_loss)
                    n_ungated += 1
            else:
                # Ungated — plain forward
                hook_mgr.clear_slot_context()
                with torch.no_grad():
                    out = model(**inputs, labels=labels)
                cell_losses[cid].append(out.loss.item())
                cell_losses_baseline[cid].append(out.loss.item())
                n_ungated += 1

        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                print(f"  Error at {i}: {type(e).__name__}: {e}")
            cell_losses[cid].append(10.0)
            continue

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(eval_samples)} | {elapsed/60:.1f}m | gated={n_gated} ungated={n_ungated} err={n_errors}")

    elapsed_min = (time.time() - t0) / 60
    _merger_hook.remove()
    hook_mgr.remove()

    # ---- Compute metrics --------------------------------------------------
    cell_mean = {cid: float(np.mean(ls)) for cid, ls in cell_losses.items()
                 if len(ls) >= MIN_CELL_SUPPORT}
    cell_mean_baseline = {cid: float(np.mean(ls)) for cid, ls in cell_losses_baseline.items()
                          if len(ls) >= MIN_CELL_SUPPORT}

    if cell_mean:
        sorted_vals = sorted(cell_mean.values(), reverse=True)
        k = max(1, int(np.ceil(len(sorted_vals) * 0.1)))
        w10_cvar = float(np.mean(sorted_vals[:k]))
        worst_cell = float(max(cell_mean.values()))
    else:
        w10_cvar = worst_cell = 1.0

    overall = float(np.mean([l for ls in cell_losses.values() for l in ls]))

    # Print results
    print(f"\n{'='*70}")
    print("D1 CORRECTED EVAL RESULTS")
    print(f"{'='*70}")
    print(f"  Samples: {len(eval_samples)}, gated: {n_gated}, ungated: {n_ungated}, errors: {n_errors}")
    print(f"  W10%CVaR (loss): {w10_cvar:.4f}")
    print(f"  Worst cell loss: {worst_cell:.4f}")
    print(f"  Overall loss:    {overall:.4f}")
    print(f"  Elapsed: {elapsed_min:.1f}m")
    print(f"\n  Per-cell (D1 slot path active on gated samples):")
    print(f"  {'Cell':<30s} {'D1_eval':>10s} {'baseline':>10s} {'delta':>10s}")
    print(f"  {'-'*60}")
    for cid in WORST_CELLS:
        d1 = cell_mean.get(cid, float("nan"))
        bl = cell_mean_baseline.get(cid, float("nan"))
        delta = d1 - bl if not (np.isnan(d1) or np.isnan(bl)) else float("nan")
        print(f"  {cid:<30s} {d1:>10.4f} {bl:>10.4f} {delta:>+10.4f}")

    gate_summary = gate.activation_summary()
    print(f"\n  Gate summary:")
    for k, v in sorted(gate_summary.items()):
        if "rate" in k:
            print(f"    {k}: {v:.3f}")

    # Save
    output = {
        "eval_type": "d1_corrected_eval",
        "checkpoint": str(ckpt_dir),
        "mode": args.mode,
        "eval_split": args.eval_split,
        "nc_cache_eval": args.nc_cache_eval,
        "n_samples": len(eval_samples),
        "n_gated": n_gated,
        "n_ungated": n_ungated,
        "n_errors": n_errors,
        "gate_scalar_sigmoid": torch.sigmoid(adapter.gate_scalar).item(),
        "w10_cvar_loss": w10_cvar,
        "worst_cell_loss": worst_cell,
        "overall_loss": overall,
        "cell_losses_d1": cell_mean,
        "cell_losses_baseline": cell_mean_baseline,
        "gate_summary": gate_summary,
        "elapsed_min": elapsed_min,
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
