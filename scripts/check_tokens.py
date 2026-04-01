"""Check first-token IDs for WhatsUp choices across tokenizers."""
import json
from transformers import AutoTokenizer

with open("data/processed/whatsup.jsonl") as f:
    sample = json.loads(f.readline())

choices = sample["choices"]
print("Choices:", choices)

# OPT tokenizer (BLIP2)
tok = AutoTokenizer.from_pretrained("facebook/opt-2.7b", cache_dir="/LOCAL2/psqhe8/hf_cache")
print("\n=== OPT (BLIP2) tokenizer ===")
for c in choices:
    tokens = tok.encode(c, add_special_tokens=False)
    print(f"  '{c[:45]}' -> first_token_id={tokens[0]}, decoded='{tok.decode([tokens[0]])}'")

# Qwen2 tokenizer
tok2 = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", cache_dir="/LOCAL2/psqhe8/hf_cache", trust_remote_code=True)
print("\n=== Qwen2 tokenizer ===")
for c in choices:
    tokens = tok2.encode(c, add_special_tokens=False)
    print(f"  '{c[:45]}' -> first_token_id={tokens[0]}, decoded='{tok2.decode([tokens[0]])}'")

# LLaVA tokenizer (LLaMA)
tok3 = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir="/LOCAL2/psqhe8/hf_cache")
print("\n=== LLaMA (LLaVA) tokenizer ===")
for c in choices:
    tokens = tok3.encode(c, add_special_tokens=False)
    print(f"  '{c[:45]}' -> first_token_id={tokens[0]}, decoded='{tok3.decode([tokens[0]])}'")

# Also check a second sample with different choices
with open("data/processed/whatsup.jsonl") as f:
    lines = [json.loads(l) for l in f]

# Find a sample where answer is NOT the first choice
for s in lines:
    if s["answer"] != s["choices"][0]:
        print(f"\nSample where answer != choices[0]:")
        print(f"  Choices: {s['choices']}")
        print(f"  Answer: {s['answer']}")
        break
