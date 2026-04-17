#!/usr/bin/env python3
"""
Unified slot-active generation-based accuracy eval for D1.

Core function `run_generation_eval()` is called from:
  - This script's main() (standalone eval from a checkpoint on disk)
  - pilot_d1_slots.py (auto-eval after training, using in-memory model)

For each eval sample:
  - Check CP gate (repair_val NC cache, train-calibrated taus)
  - If gated AND matchable: capture vision features via merger hook, build
    slots, set hook context, then model.generate()
  - If ungated: plain model.generate() (hook context = None)

Output format matches eval_depth_object_level.py:
  overall_accuracy / worst_cell_error / per-cell errors / W10%CVaR_error
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
    MODEL_ID, CACHE_DIR, WORST_CELLS, DEPTH_NP_CACHE, MIN_CELL_SUPPORT,
    build_prompt, get_cell_id, load_image, load_worst_cell_samples,
    compute_object_depth,
)
from scripts.pilot_d1_slots import (
    CPGate, match_named_objects, resolve_raw_qid,
    box_coords_normalized, box_pooled_visual,
    GQA_QUESTIONS_FILE, GQA_SCENE_GRAPHS,
    D1SlotModule, DepthTokenModule, SlotCrossAttentionAdapter, SlotHookManager,
    D_SLOT, D_MODEL_QWEN3VL_8B, ALPHA_GATE, ALPHA_CVAR,
)


def load_depth_np(sample):
    img_id = Path(sample["image_path"]).stem
    npy = DEPTH_NP_CACHE / f"{img_id}.npy"
    return np.load(npy) if npy.exists() else None


def run_generation_eval(
    model, processor, process_vision_info, eval_samples, partition,
    hook_mgr, slot_mod, adapter, gate,
    raw_questions, scene_graphs, captured_pre_merger: dict,
    d_vis: int, device,
    label: str = "D1 slot-active generation eval",
    is_depth_token: bool = False,
):
    """Core generation eval loop — reusable from standalone or in-memory.

    Args:
        captured_pre_merger: mutable dict updated by the merger hook with key "feat".
        All other args: already-loaded, already-on-device objects.

    Returns:
        dict with overall_accuracy, worst_cell_error, cell_errors, etc.
    """
    model.eval()
    slot_mod.eval()
    adapter.eval()

    results_gated = []
    results_ungated = []
    n_gated = n_ungated = n_errors = 0

    t0 = time.time()
    for i, s in enumerate(eval_samples):
        prompt = build_prompt(s)
        image = load_image(s["image_path"])
        cid = partition.get_cell_by_features(s)

        qid = resolve_raw_qid(s.get("id", ""))
        raw = raw_questions.get(qid)
        matched = []
        if raw is not None:
            img_id = raw.get("imageId")
            sg_objs = scene_graphs.get(img_id, {}).get("objects", {})
            matched = match_named_objects(raw, sg_objs)

        n_match = len(matched)
        gate_on = gate(s, n_match)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]

        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            img_inputs, vid_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=img_inputs, videos=vid_inputs,
                               padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            slot_set = False
            if gate_on and n_match >= 1:
                img_wh = image.size
                depth_np = load_depth_np(s)
                sb_l, sd_l = [], []
                for obj in matched:
                    sb_l.append(torch.tensor(box_coords_normalized(obj["bbox"], img_wh), device=device))
                    d_val = compute_object_depth(depth_np, obj["bbox"]) if depth_np is not None else 0.0
                    d_val = d_val if d_val is not None else 0.0
                    sd_l.append(torch.tensor([d_val], dtype=torch.bfloat16, device=device))
                sb = torch.stack(sb_l).unsqueeze(0).to(dtype=torch.bfloat16)
                sd = torch.stack(sd_l).unsqueeze(0)
                sm = torch.ones(1, n_match, dtype=torch.bool, device=device)

                try:
                    if is_depth_token:
                        with torch.no_grad():
                            tokens, _ = slot_mod(sb, sd, sm)
                        hook_mgr.set_slot_context(tokens, sm)
                        slot_set = True
                        n_gated += 1
                    else:
                        hook_mgr.clear_slot_context()
                        with torch.no_grad():
                            _ = model(**inputs)
                        pre_merger = captured_pre_merger.get("feat")
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
                                    slots, _ = slot_mod(sv, sb, sd, sm)
                                hook_mgr.set_slot_context(slots, sm)
                                slot_set = True
                                n_gated += 1
                except Exception:
                    hook_mgr.clear_slot_context()
                    raise

                if not slot_set:
                    hook_mgr.clear_slot_context()
                    n_ungated += 1
            else:
                hook_mgr.clear_slot_context()
                n_ungated += 1

            try:
                with torch.no_grad():
                    gen_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            finally:
                hook_mgr.clear_slot_context()

            input_len = inputs["input_ids"].shape[1]
            response = processor.decode(
                gen_ids[0][input_len:], skip_special_tokens=True).strip().lower()

            answer = s["answer"].lower().strip()
            is_correct = (
                response == answer
                or (answer in response)
                or (response in answer and len(response) >= 2)
            )
        except Exception as e:
            response = ""
            is_correct = False
            n_errors += 1
            hook_mgr.clear_slot_context()
            if n_errors <= 5:
                print(f"  Error at {i}: {type(e).__name__}: {e}")

        entry = {"correct": is_correct, "cell_id": cid, "response": response,
                 "answer": s["answer"].lower().strip(), "gated": slot_set}
        if slot_set:
            results_gated.append(entry)
        else:
            results_ungated.append(entry)

        if (i + 1) % 100 == 0:
            all_res = results_gated + results_ungated
            acc = np.mean([r["correct"] for r in all_res])
            print(f"  gen eval: {i+1}/{len(eval_samples)} acc={acc:.3f} gated={n_gated} ungated={n_ungated}")

    elapsed_min = (time.time() - t0) / 60

    all_results = results_gated + results_ungated
    overall_acc = float(np.mean([r["correct"] for r in all_results]))

    cell_errors = defaultdict(list)
    for r in all_results:
        if r["cell_id"] is not None:
            cell_errors[r["cell_id"]].append(1.0 - float(r["correct"]))

    cell_mean_error = {cid: float(np.mean(errs)) for cid, errs in cell_errors.items()
                       if len(errs) >= MIN_CELL_SUPPORT}

    sorted_vals = sorted(cell_mean_error.values(), reverse=True) if cell_mean_error else [1.0]
    k = max(1, int(np.ceil(len(sorted_vals) * 0.1)))
    w10_cvar_err = float(np.mean(sorted_vals[:k]))
    worst_cell_err = float(max(cell_mean_error.values())) if cell_mean_error else 1.0

    gated_acc = float(np.mean([r["correct"] for r in results_gated])) if results_gated else 0.0
    ungated_acc = float(np.mean([r["correct"] for r in results_ungated])) if results_ungated else 0.0

    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    print(f"  Overall accuracy:   {overall_acc:.4f}")
    print(f"  Worst cell error:   {worst_cell_err:.4f}")
    print(f"  W10% CVaR(error):   {w10_cvar_err:.4f}")
    print(f"  Gated acc:          {gated_acc:.4f} ({n_gated} samples)")
    print(f"  Ungated acc:        {ungated_acc:.4f} ({n_ungated} samples)")
    print(f"  Errors:             {n_errors}")
    print(f"  Elapsed:            {elapsed_min:.1f}m")
    print(f"\n  Per-cell errors:")
    for cid in WORST_CELLS:
        err = cell_mean_error.get(cid, float("nan"))
        print(f"    {cid}: {err:.4f}")

    return {
        "eval_type": "d1_slot_active_generation",
        "overall_accuracy": overall_acc,
        "worst_cell_error": worst_cell_err,
        "worst_10pct_cvar_error": w10_cvar_err,
        "n_evaluated": len(all_results),
        "n_gated": n_gated,
        "n_ungated": n_ungated,
        "n_errors": n_errors,
        "gated_accuracy": gated_acc,
        "ungated_accuracy": ungated_acc,
        "cell_errors": cell_mean_error,
        "gate_scalar_sigmoid": torch.sigmoid(adapter.gate_scalar).item(),
        "elapsed_min": elapsed_min,
    }


# ---------------------------------------------------------------------------
# Standalone entry point (loads checkpoint from disk)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--nc_cache_eval", type=str, required=True)
    parser.add_argument("--mode", type=str, default="r2_selective")
    parser.add_argument("--eval_split", type=str, default="repair_val")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint)
    assert (ckpt_dir / "slot_mod.pt").exists()
    assert (ckpt_dir / "adapter.pt").exists()
    if args.output is None:
        args.output = str(ckpt_dir.parent / "eval_generation_results.json")

    print(f"Checkpoint: {ckpt_dir}")
    print(f"NC cache:   {args.nc_cache_eval}")

    eval_samples = load_worst_cell_samples(args.eval_split)
    gate = CPGate(Path(args.nc_cache_eval), mode=args.mode)

    with open(GQA_QUESTIONS_FILE) as f:
        raw_questions = json.load(f)
    with open(GQA_SCENE_GRAPHS) as f:
        scene_graphs = json.load(f)

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from src.diagnosis.mondrian_partition import MondrianPartition

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(base_model, str(ckpt_dir))
    model.eval()

    device = next(model.parameters()).device
    d_vis = model.config.vision_config.hidden_size
    d_model = model.config.text_config.hidden_size

    is_dt = (args.mode == "r4_depth_token")
    if is_dt:
        slot_mod = DepthTokenModule(d_slot=D_SLOT).to(device, dtype=torch.bfloat16)
    else:
        slot_mod = D1SlotModule(d_vis=d_vis, d_model=d_model).to(device, dtype=torch.bfloat16)
    slot_mod.load_state_dict(torch.load(ckpt_dir / "slot_mod.pt", map_location=device, weights_only=True))
    slot_mod.eval()

    adapter = SlotCrossAttentionAdapter(d_model=d_model, d_slot=D_SLOT).to(device, dtype=torch.bfloat16)
    adapter.load_state_dict(torch.load(ckpt_dir / "adapter.pt", map_location=device, weights_only=True))
    adapter.eval()

    hook_mgr = SlotHookManager(model, adapter).register()

    captured = {}
    if hasattr(model, "base_model"):
        _visual = model.base_model.model.model.visual
    else:
        _visual = model.model.visual
    merger_hook = _visual.merger.register_forward_hook(
        lambda mod, inp, out: captured.update({"feat": inp[0].detach()})
    )

    partition = MondrianPartition.load(Path("results/sprint2/b1_diagnosis/partition.json"))

    result = run_generation_eval(
        model, processor, process_vision_info, eval_samples, partition,
        hook_mgr, slot_mod, adapter, gate,
        raw_questions, scene_graphs, captured, d_vis, device,
        is_depth_token=is_dt,
    )
    result["checkpoint"] = str(ckpt_dir)
    result["mode"] = args.mode

    merger_hook.remove()
    hook_mgr.remove()

    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
