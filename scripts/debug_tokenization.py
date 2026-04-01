#!/usr/bin/env python3
"""Debug: verify label masking is correct in tokenize_train_example."""
import json, sys, torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_ID  = "Qwen/Qwen2-VL-2B-Instruct"
CACHE_DIR = "/LOCAL2/psqhe8/hf_cache"

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)

# Load one sample
with open("data/splits/train.jsonl") as f:
    sample = json.loads(f.readline())

print(f"Sample: {sample['id']}")
print(f"Answer: {sample['answer']}")

# Build messages
answer = sample["answer"].lower().strip()
image = Image.open(sample["image_path"]).convert("RGB")
prompt_text = f'Look at the image. Is the following spatial statement true or false?\n\nStatement: "{sample["question"]}"\n\nAnswer with ONLY "true" or "false".'

messages_full = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt_text},
    ]},
    {"role": "assistant", "content": [
        {"type": "text", "text": answer},
    ]},
]
messages_prompt = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt_text},
    ]},
]

# Chat templates
text_full = processor.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
text_prompt = processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)

print(f"\n--- Text lengths ---")
print(f"text_full chars:   {len(text_full)}")
print(f"text_prompt chars: {len(text_prompt)}")
print(f"Diff (answer portion chars): {len(text_full) - len(text_prompt)}")

# Text-only tokenization (current approach)
prompt_ids_textonly = processor.tokenizer.encode(text_prompt)
full_ids_textonly = processor.tokenizer.encode(text_full)
print(f"\n--- Text-only tokenizer.encode() ---")
print(f"full_ids_textonly:   {len(full_ids_textonly)}")
print(f"prompt_ids_textonly: {len(prompt_ids_textonly)}")
print(f"Answer tokens (text-only): {len(full_ids_textonly) - len(prompt_ids_textonly)}")

# Processor tokenization (with image expansion)
image_inputs, video_inputs = process_vision_info(messages_full)
inputs_full = processor(text=[text_full], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

image_inputs_p, video_inputs_p = process_vision_info(messages_prompt)
inputs_prompt = processor(text=[text_prompt], images=image_inputs_p, videos=video_inputs_p, padding=True, return_tensors="pt")

actual_full_len = inputs_full["input_ids"].shape[1]
actual_prompt_len = inputs_prompt["input_ids"].shape[1]

print(f"\n--- Processor (with image expansion) ---")
print(f"actual full length:   {actual_full_len}")
print(f"actual prompt length: {actual_prompt_len}")
print(f"Answer tokens (actual): {actual_full_len - actual_prompt_len}")
print(f"Image expansion:        {actual_full_len - len(full_ids_textonly)}")

# The bug
print(f"\n--- BUG ANALYSIS ---")
print(f"Current prompt_len (text-only):  {len(prompt_ids_textonly)}")
print(f"Correct prompt_len (with image): {actual_prompt_len}")
print(f"Tokens being WRONGLY included in loss: {actual_prompt_len - len(prompt_ids_textonly)}")

# Show what tokens are at the boundary
input_ids = inputs_full["input_ids"][0]
print(f"\n--- Token boundary check ---")
wrong_mask_end = len(prompt_ids_textonly)
correct_mask_end = actual_prompt_len

print(f"Tokens around WRONG boundary (pos {wrong_mask_end-3} to {wrong_mask_end+3}):")
for i in range(max(0, wrong_mask_end-3), min(actual_full_len, wrong_mask_end+4)):
    tok = processor.tokenizer.decode([input_ids[i].item()])
    print(f"  [{i:4d}] id={input_ids[i].item():6d}  '{tok}'  {'<-- WRONG boundary' if i == wrong_mask_end else ''}")

print(f"\nTokens around CORRECT boundary (pos {correct_mask_end-3} to {correct_mask_end+3}):")
for i in range(max(0, correct_mask_end-3), min(actual_full_len, correct_mask_end+4)):
    tok = processor.tokenizer.decode([input_ids[i].item()])
    print(f"  [{i:4d}] id={input_ids[i].item():6d}  '{tok}'  {'<-- CORRECT boundary' if i == correct_mask_end else ''}")

# Count special image tokens
image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
n_image_pad = (input_ids == image_pad_id).sum().item()
print(f"\nTotal <|image_pad|> tokens in input_ids: {n_image_pad}")
print(f"These should ALL be masked (label=-100) but current code only masks first {len(prompt_ids_textonly)} tokens")
