#!/usr/bin/env python3
"""Verify that offset-based prompt_len matches double-processor approach."""
import json
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    cache_dir="/LOCAL2/psqhe8/hf_cache",
    trust_remote_code=True,
)

samples = []
with open("data/splits/train.jsonl") as f:
    for i, line in enumerate(f):
        s = json.loads(line)
        if i in [0, 5000, 20000, 40000, 53000]:
            samples.append(s)

for s in samples:
    image = Image.open(s["image_path"]).convert("RGB")
    answer = s["answer"].lower().strip()
    prompt = f'Q: {s["question"]}'

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

    text_full = processor.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
    text_prompt = processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)

    img_in, vid_in = process_vision_info(messages_full)
    inputs_full = processor(text=[text_full], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

    img_in_p, vid_in_p = process_vision_info(messages_prompt)
    inputs_prompt = processor(text=[text_prompt], images=img_in_p, videos=vid_in_p, padding=True, return_tensors="pt")

    # Method 1: double processor (current)
    prompt_len_double = inputs_prompt["input_ids"].shape[1]

    # Method 2: offset calculation (proposed optimization)
    full_actual = inputs_full["input_ids"].shape[1]
    full_text = len(processor.tokenizer.encode(text_full))
    prompt_text = len(processor.tokenizer.encode(text_prompt))
    image_expansion = full_actual - full_text
    prompt_len_offset = prompt_text + image_expansion

    answer_tokens = full_actual - prompt_len_double

    match = "OK" if prompt_len_double == prompt_len_offset else "MISMATCH!"
    print(f"{s['dataset']:8s} img_exp={image_expansion:4d} "
          f"prompt_double={prompt_len_double:4d} prompt_offset={prompt_len_offset:4d} "
          f"answer_toks={answer_tokens:2d} {match}")
