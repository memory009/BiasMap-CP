"""Minimal test: verify sequence scoring works for WhatsUp (all choices share first token 'A')."""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.datasets.base import SpatialQASample, BaseDataset
from src.models.blip2 import BLIP2Model

CACHE_DIR = "/LOCAL/psqhe8/hf_cache"

# Load just 5 WhatsUp test samples
with open("/LOCAL2/psqhe8/BiasMap-CP/data/splits/test.jsonl") as f:
    all_samples = [json.loads(l) for l in f]

whatsup_raw = [s for s in all_samples if s.get("dataset") == "whatsup"][:5]
if not whatsup_raw:
    print("No WhatsUp samples in test split, checking processed...")
    with open("/LOCAL2/psqhe8/BiasMap-CP/data/processed/whatsup.jsonl") as f:
        whatsup_raw = [json.loads(l) for l in f][:5]

samples = []
for r in whatsup_raw:
    samples.append(SpatialQASample(
        id=r["id"], dataset="whatsup",
        image_path=r.get("image_path", ""),
        question=r["question"], answer=r["answer"],
        choices=r.get("choices"), relation_type=r.get("relation_type", "unknown"),
        subject=r.get("subject"), object=r.get("object"),
        subject_bbox=r.get("subject_bbox"), object_bbox=r.get("object_bbox"),
        scene_type=r.get("scene_type"), viewpoint=r.get("viewpoint"),
        occlusion_level=r.get("occlusion_level"),
        object_size_ratio=r.get("object_size_ratio"),
        depth_ambiguity=r.get("depth_ambiguity"),
    ))

print(f"Testing {len(samples)} WhatsUp samples with BLIP2...")
print(f"Sample choices: {samples[0].choices}")
print(f"Sample answer: {samples[0].answer}")

model = BLIP2Model("Salesforce/blip2-opt-2.7b", cache_dir=CACHE_DIR, dtype="bfloat16")
model.load()

for i, sample in enumerate(samples):
    out = model.predict_sample(sample)
    logits_str = {k[:30]: f"{v:.3f}" for k, v in out.logits.items()}
    probs_str = {k[:30]: f"{v:.3f}" for k, v in out.probabilities.items()}
    print(f"\n--- Sample {i} ---")
    print(f"  Answer: {sample.answer[:50]}")
    print(f"  Predicted: {out.predicted_answer[:50]}")
    print(f"  Correct: {out.correct}")
    print(f"  Raw response: '{out.raw_response[:50]}'")
    print(f"  Logits: {logits_str}")
    print(f"  Probs: {probs_str}")
    print(f"  NC score: {out.nonconformity_score:.3f}")

    # Verify logits are NOT uniform
    vals = list(out.logits.values())
    if len(set(round(v, 3) for v in vals)) == 1:
        print("  WARNING: Logits still uniform! Fix may not be working.")
    else:
        print("  OK: Logits differentiated (sequence scoring working)")

del model.model, model
torch.cuda.empty_cache()
print("\nTest complete.")
