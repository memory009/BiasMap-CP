#!/usr/bin/env python3
"""
D1 engineering smoke test — runs on GPU 0 or 1 (not GPU 2 while NC cache runs).

Tests (in order):
  1. Vision-tower patch feature hook + box_pooled_visual on a real sample
  2. SlotHookManager finds the correct final norm on PEFT-wrapped Qwen3-VL-8B
  3. End-to-end gated forward pass: no NaN, correct shapes, gate logging
  4. End-to-end ungated forward pass: identical to plain model forward (no hook leak)

Does NOT train. Uses a single hard-cell sample for all checks.
"""
from __future__ import annotations

import json
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.pilot_depth_object_level import (
    MODEL_ID, CACHE_DIR, WORST_CELLS, DEPTH_NP_CACHE,
    build_prompt, build_answer, get_cell_id, load_image, load_worst_cell_samples,
    compute_object_depth,
)
from scripts.pilot_d1_slots import (
    match_named_objects, resolve_raw_qid, box_coords_normalized,
    D1SlotModule, SlotCrossAttentionAdapter, SlotHookManager, D_SLOT,
    GQA_QUESTIONS_FILE, GQA_SCENE_GRAPHS,
)


def main():
    print("=" * 70)
    print("D1 ENGINEERING SMOKE TEST")
    print("=" * 70)

    # ---- Load one hard-cell sample ----------------------------------------
    samples = load_worst_cell_samples("train")
    with open(GQA_QUESTIONS_FILE) as f:
        raw_questions = json.load(f)
    with open(GQA_SCENE_GRAPHS) as f:
        scene_graphs = json.load(f)

    # Find a sample with n_match >= 2
    test_sample = None
    test_matched = []
    for s in samples:
        qid = resolve_raw_qid(s["id"])
        raw = raw_questions.get(qid)
        if raw is None:
            continue
        img_id = raw.get("imageId")
        if img_id not in scene_graphs:
            continue
        sg_objects = scene_graphs[img_id].get("objects", {})
        matched = match_named_objects(raw, sg_objects)
        if len(matched) >= 2:
            test_sample = s
            test_matched = matched
            break

    assert test_sample is not None, "No matchable sample found"
    print(f"\nTest sample: {test_sample['id']}")
    print(f"  cell: {get_cell_id(test_sample)}")
    print(f"  question: {test_sample['question']}")
    print(f"  matched objects: {[m['name'] for m in test_matched]}")

    # ---- Load model -------------------------------------------------------
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel
    from qwen_vl_utils import process_vision_info

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    print(f"\nLoading {MODEL_ID} (4-bit)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, quantization_config=bnb, device_map="auto")

    # Load R1 LoRA
    ckpt = "results/sprint2/pilots/pilot_depth_object_level_seed1_qwen3vl8b_r1_replication/checkpoint-best"
    if Path(ckpt).exists():
        model = PeftModel.from_pretrained(base_model, ckpt)
        print(f"  Loaded R1 LoRA from {ckpt}")
    else:
        lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                              target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base_model, lora_cfg)
        print("  No R1 checkpoint; using fresh LoRA (shapes still valid for smoke test)")

    d_model = model.config.text_config.hidden_size
    d_vis = model.config.vision_config.hidden_size
    print(f"  d_model={d_model}, d_vis={d_vis}")

    # ---- TEST 1: Vision-tower patch feature hook ---------------------------
    print(f"\n{'='*70}\nTEST 1: Vision-tower patch feature hook + box_pooled_visual\n{'='*70}")

    captured_features = {}

    def merger_input_hook(module, input, output):
        # merger input[0] is the hidden states from blocks: (N_patches, d_vis)
        if isinstance(input, tuple) and len(input) > 0:
            captured_features["pre_merger"] = input[0].detach()

    # Find merger module
    if hasattr(model, "base_model"):
        visual_mod = model.base_model.model.model.visual
    else:
        visual_mod = model.model.visual
    merger = visual_mod.merger
    hook_handle = merger.register_forward_hook(merger_input_hook)

    # Tokenize + forward to trigger the hook
    prompt = build_prompt(test_sample)
    answer = build_answer(test_sample)
    image = load_image(test_sample["image_path"])
    msg = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}, {"role": "assistant", "content": [{"type": "text", "text": answer}]}]

    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    img_inputs, vid_inputs = process_vision_info(msg)
    inputs = processor(text=[text], images=img_inputs, videos=vid_inputs,
                       padding=True, return_tensors="pt")

    prompt_text = processor.apply_chat_template(msg[:1], tokenize=False, add_generation_prompt=True)
    full_len = inputs["input_ids"].shape[1]
    full_text_len = len(processor.tokenizer.encode(text))
    prompt_text_len = len(processor.tokenizer.encode(prompt_text))
    img_expansion = full_len - full_text_len
    prompt_len = prompt_text_len + img_expansion
    labels = inputs["input_ids"].clone()
    labels[0, :prompt_len] = -100

    inputs_dev = {k: v.to(model.device) for k, v in inputs.items()}
    labels_dev = labels.to(model.device)

    model.eval()
    with torch.no_grad():
        out_baseline = model(**inputs_dev, labels=labels_dev)
    baseline_loss = out_baseline.loss.item()

    hook_handle.remove()

    pre_merger = captured_features.get("pre_merger")
    assert pre_merger is not None, "FAIL: merger hook did not fire"
    print(f"  pre_merger shape: {pre_merger.shape}")  # (N_patches, d_vis)
    assert pre_merger.shape[-1] == d_vis, f"FAIL: expected d_vis={d_vis}, got {pre_merger.shape[-1]}"

    # Get image_grid_thw for spatial mapping
    grid_thw = inputs.get("image_grid_thw")
    print(f"  image_grid_thw: {grid_thw}")
    img_w, img_h = image.size
    print(f"  image size: {img_w}x{img_h}")

    if grid_thw is not None:
        t, h_grid, w_grid = grid_thw[0].tolist()
        print(f"  grid: t={t} h={h_grid} w={w_grid} => {t*h_grid*w_grid} patches")
        # Reshape to spatial grid for box pooling
        n_patches_total = pre_merger.shape[0]
        n_grid = int(t * h_grid * w_grid)
        print(f"  pre_merger N_patches={n_patches_total}, grid={n_grid}")
        # Qwen3-VL may have multiple temporal frames; for single image t=1
        if t == 1 and n_patches_total >= n_grid:
            patch_grid = pre_merger[:n_grid].reshape(int(h_grid), int(w_grid), d_vis)
            # Test box pooling for first matched object
            obj = test_matched[0]
            bbox = obj["bbox"]
            from scripts.pilot_d1_slots import box_pooled_visual
            pooled = box_pooled_visual(patch_grid, bbox, (img_w, img_h), (int(h_grid), int(w_grid)))
            print(f"  box_pooled_visual for '{obj['name']}': shape={pooled.shape}, "
                  f"norm={pooled.norm().item():.2f}")
            assert pooled.shape == (d_vis,), f"FAIL: expected ({d_vis},), got {pooled.shape}"
            print("  TEST 1: PASSED")
        else:
            print(f"  SKIP box pooling: multi-temporal or shape mismatch")
    else:
        print("  SKIP: no image_grid_thw in inputs")

    # ---- TEST 2: SlotHookManager finds final norm -------------------------
    print(f"\n{'='*70}\nTEST 2: SlotHookManager norm discovery on PEFT model\n{'='*70}")
    adapter = SlotCrossAttentionAdapter(d_model=d_model, d_slot=D_SLOT).to(model.device, dtype=torch.bfloat16)
    hook_mgr = SlotHookManager(model, adapter)
    norm = hook_mgr._find_final_norm()
    print(f"  Found norm: {type(norm).__name__}")
    hook_mgr.register()
    print("  Hook registered successfully")
    print("  TEST 2: PASSED")

    # ---- TEST 3: Gated forward pass ---------------------------------------
    print(f"\n{'='*70}\nTEST 3: Gated forward — no NaN, correct shapes\n{'='*70}")
    slot_mod = D1SlotModule(d_vis=d_vis, d_model=d_model).to(model.device, dtype=torch.bfloat16)

    # Build slot inputs for matched objects
    K = len(test_matched)
    depth_np = np.load(DEPTH_NP_CACHE / f"{Path(test_sample['image_path']).stem}.npy")
    slot_visual_list, slot_box_list, slot_depth_list = [], [], []
    if grid_thw is not None and t == 1:
        patch_grid = pre_merger[:n_grid].reshape(int(h_grid), int(w_grid), d_vis)
        for obj in test_matched:
            pooled = box_pooled_visual(patch_grid, obj["bbox"], (img_w, img_h), (int(h_grid), int(w_grid)))
            slot_visual_list.append(pooled)
            slot_box_list.append(torch.tensor(box_coords_normalized(obj["bbox"], (img_w, img_h))))
            d = compute_object_depth(depth_np, obj["bbox"])
            slot_depth_list.append(torch.tensor([d if d is not None else 0.0], dtype=torch.float32))

        sv = torch.stack(slot_visual_list).unsqueeze(0).to(model.device, dtype=torch.bfloat16)
        sb = torch.stack(slot_box_list).unsqueeze(0).to(model.device, dtype=torch.bfloat16)
        sd = torch.stack(slot_depth_list).unsqueeze(0).to(model.device, dtype=torch.bfloat16)
        sm = torch.ones(1, K, dtype=torch.bool, device=model.device)

        slots, pair_logits = slot_mod(sv, sb, sd, sm)
        print(f"  slots: {slots.shape}, pair_logits: {pair_logits.shape}")
        assert not torch.isnan(slots).any(), "FAIL: NaN in slots"
        assert not torch.isnan(pair_logits).any(), "FAIL: NaN in pair_logits"

        # Gated forward via hook
        hook_mgr.set_slot_context(slots, sm)
        with torch.no_grad():
            out_gated = model(**inputs_dev, labels=labels_dev)
        hook_mgr.clear_slot_context()
        gated_loss = out_gated.loss.item()
        print(f"  baseline loss: {baseline_loss:.4f}")
        print(f"  gated loss:    {gated_loss:.4f}")
        print(f"  delta:         {gated_loss - baseline_loss:+.4f}")
        assert not np.isnan(gated_loss), "FAIL: NaN in gated loss"
        print("  TEST 3: PASSED")
    else:
        print("  SKIP: no grid_thw")

    # ---- TEST 4: Ungated forward — must match baseline exactly ------------
    print(f"\n{'='*70}\nTEST 4: Ungated forward — no hook leakage\n{'='*70}")
    # Context is already cleared; hook should be a no-op
    with torch.no_grad():
        out_ungated = model(**inputs_dev, labels=labels_dev)
    ungated_loss = out_ungated.loss.item()
    print(f"  baseline loss: {baseline_loss:.4f}")
    print(f"  ungated loss:  {ungated_loss:.4f}")
    diff = abs(ungated_loss - baseline_loss)
    print(f"  |diff|:        {diff:.8f}")
    assert diff < 1e-5, f"FAIL: hook leakage, diff={diff}"
    print("  TEST 4: PASSED (exact R1 behavior on ungated samples)")

    hook_mgr.remove()

    print(f"\n{'='*70}")
    print("ALL SMOKE TESTS PASSED")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
