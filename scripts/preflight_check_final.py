"""Final pre-flight check: test ALL 4 models × ALL 4 question types.
Must pass 100% before full run."""
import os, sys, json, torch, gc, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.datasets.base import SpatialQASample, BaseDataset
from src.models.base_vlm import BaseVLM
from src.models.blip2 import BLIP2Model
from src.models.llava import LLaVAModel
from src.models.qwen2vl import Qwen2VLModel

CACHE_DIR = "/LOCAL/psqhe8/hf_cache"
SPLITS_ROOT = "/LOCAL/psqhe8/BiasMap-CP/data/splits"

# Load test samples
test_samples = BaseDataset.load_processed(os.path.join(SPLITS_ROOT, "test.jsonl"))
vsr = [s for s in test_samples if s.dataset == "vsr"][:2]
whatsup = [s for s in test_samples if s.dataset == "whatsup"][:2]
gqa = [s for s in test_samples if s.dataset == "gqa"]
gqa_yn = [s for s in gqa if s.answer.lower().strip() in ("yes", "no")][:2]
gqa_open = [s for s in gqa if s.answer.lower().strip() not in ("yes", "no")][:2]

test_cases = {
    "VSR(T/F)": vsr,
    "WhatsUp(MCQ)": whatsup,
    "GQA(Y/N)": gqa_yn,
    "GQA(Open)": gqa_open,
}

models = [
    ("BLIP2", BLIP2Model, "Salesforce/blip2-opt-2.7b"),
    ("LLaVA-7B", LLaVAModel, "llava-hf/llava-1.5-7b-hf"),
    ("Qwen2-VL-2B", Qwen2VLModel, "Qwen/Qwen2-VL-2B-Instruct"),
    ("Qwen2-VL-7B", Qwen2VLModel, "Qwen/Qwen2-VL-7B-Instruct"),
]

results = {}
total_pass = 0
total_tests = 0

for model_name, ModelClass, hf_id in models:
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    model = ModelClass(hf_id, cache_dir=CACHE_DIR, dtype="bfloat16")
    model.load()

    for case_name, samples in test_cases.items():
        total_tests += 1
        s = samples[0]
        try:
            out = model.predict_sample(s)

            # Validate output
            errors = []

            # Check logits are not empty
            if not out.logits:
                errors.append("empty logits")

            # WhatsUp: logits must NOT be uniform
            if case_name == "WhatsUp(MCQ)":
                vals = list(out.logits.values())
                if len(set(round(v, 3) for v in vals)) <= 1:
                    errors.append(f"UNIFORM logits {vals[:3]}")
                if abs(out.nonconformity_score - 0.75) < 0.02:
                    errors.append(f"NC score ~0.75 (uniform)")

            # Open-ended: must have __other__ key
            if case_name == "GQA(Open)":
                if "__other__" not in out.probabilities:
                    errors.append("missing __other__ in probs")

            # Probs must sum to ~1.0
            prob_sum = sum(out.probabilities.values())
            if abs(prob_sum - 1.0) > 0.01:
                errors.append(f"probs sum={prob_sum:.4f}")

            if errors:
                status = f"FAIL: {'; '.join(errors)}"
            else:
                status = "PASS ✓"
                total_pass += 1

            # Print concise result
            logit_str = {k[:20]: f"{v:.2f}" for k, v in list(out.logits.items())[:3]}
            print(f"  {case_name:15s} | {status}")
            print(f"    logits={logit_str}  pred='{out.predicted_answer[:25]}' raw='{out.raw_response[:25]}' nc={out.nonconformity_score:.3f}")

        except Exception as e:
            print(f"  {case_name:15s} | CRASH: {e}")
            total_tests  # already incremented

    del model.model, model
    gc.collect()
    torch.cuda.empty_cache()

print(f"\n{'='*60}")
print(f"  RESULT: {total_pass}/{total_tests} passed")
if total_pass == total_tests:
    print(f"  ALL CHECKS PASSED ✓ — safe to run full baselines")
else:
    print(f"  SOME CHECKS FAILED ✗ — DO NOT run full baselines")
print(f"{'='*60}")
