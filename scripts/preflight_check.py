"""Pre-flight check: test ALL code paths before full run.
Tests every model × every question type with a few samples each.
"""
import os, sys, json, torch, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.datasets.base import SpatialQASample, ModelOutput
from src.models.base_vlm import BaseVLM

CACHE_DIR = "/LOCAL/psqhe8/hf_cache"
SPLITS_ROOT = "/LOCAL/psqhe8/BiasMap-CP/data/splits"

# ============================================================
# Part 1: Load test samples (2 from each dataset/type)
# ============================================================
print("=" * 60)
print("PART 1: Loading test samples")
print("=" * 60)

# Load from test split
from src.datasets.base import BaseDataset
test_samples = BaseDataset.load_processed(os.path.join(SPLITS_ROOT, "test.jsonl"))

vsr_samples = [s for s in test_samples if s.dataset == "vsr"][:3]
whatsup_samples = [s for s in test_samples if s.dataset == "whatsup"][:3]
gqa_samples = [s for s in test_samples if s.dataset == "gqa"]

# Separate GQA by type
gqa_yn = [s for s in gqa_samples if s.answer.lower().strip() in ("yes", "no")][:3]
gqa_open = [s for s in gqa_samples if s.answer.lower().strip() not in ("yes", "no")][:3]

print(f"VSR: {len(vsr_samples)} samples (binary T/F)")
print(f"  choices: {vsr_samples[0].choices}, answer: {vsr_samples[0].answer}")
print(f"WhatsUp: {len(whatsup_samples)} samples (MCQ)")
print(f"  choices: {whatsup_samples[0].choices[:2]}... answer: {whatsup_samples[0].answer[:40]}")
print(f"GQA yes/no: {len(gqa_yn)} samples")
print(f"  choices: {gqa_yn[0].choices}, answer: {gqa_yn[0].answer}")
print(f"GQA open: {len(gqa_open)} samples")
print(f"  choices: {gqa_open[0].choices}, answer: {gqa_open[0].answer}")

all_test_samples = {
    "vsr_binary": vsr_samples,
    "whatsup_mcq": whatsup_samples,
    "gqa_yesno": gqa_yn,
    "gqa_open": gqa_open,
}

# ============================================================
# Part 2: Verify infer_choices and build_prompt for each type
# ============================================================
print("\n" + "=" * 60)
print("PART 2: Verify infer_choices / build_prompt / is_open_ended")
print("=" * 60)

# Use a dummy model to test base methods
class DummyModel(BaseVLM):
    def load(self): pass
    def predict_sample(self, s): pass

dummy = DummyModel("dummy")

for name, samples in all_test_samples.items():
    s = samples[0]
    choices = dummy.infer_choices(s)
    is_oe = dummy.is_open_ended(choices)
    prompt = dummy.build_prompt(s)

    # Check first_tokens_identical would need a tokenizer - skip for now
    print(f"\n--- {name} ---")
    print(f"  infer_choices: {choices[:3]}{'...' if len(choices) > 3 else ''}")
    print(f"  is_open_ended: {is_oe}")
    print(f"  prompt type: {'BINARY' if 'true or false' in prompt.lower() else 'SPATIAL/MCQ' if 'Choose' in prompt else 'OPEN'}")
    print(f"  prompt: {prompt[:100]}...")

    # Validate logic
    if name == "vsr_binary":
        assert choices == ["true", "false"], f"FAIL: VSR choices should be ['true','false'], got {choices}"
        assert not is_oe, "FAIL: VSR should not be open-ended"
        assert "true" in prompt.lower() and "false" in prompt.lower(), "FAIL: VSR prompt should mention true/false"
    elif name == "whatsup_mcq":
        assert len(choices) == 4, f"FAIL: WhatsUp should have 4 choices, got {len(choices)}"
        assert not is_oe, "FAIL: WhatsUp should not be open-ended"
        assert "Choose" in prompt, "FAIL: WhatsUp prompt should use SPATIAL_PROMPT with 'Choose'"
    elif name == "gqa_yesno":
        assert choices == ["yes", "no"], f"FAIL: GQA yes/no choices should be ['yes','no'], got {choices}"
        assert not is_oe, "FAIL: GQA yes/no should not be open-ended"
    elif name == "gqa_open":
        assert len(choices) == 1, f"FAIL: GQA open should have 1 choice (GT), got {choices}"
        assert is_oe, "FAIL: GQA open should be open-ended"
        assert "Answer:" in prompt, "FAIL: GQA open should use OPEN_PROMPT"
    print(f"  PASS ✓")

# ============================================================
# Part 3: Verify match_response_to_choice edge cases
# ============================================================
print("\n" + "=" * 60)
print("PART 3: Verify match_response_to_choice edge cases")
print("=" * 60)

tests = [
    ("", ["true", "false"], None, "empty string"),
    ("a", ["true", "false"], None, "single char"),
    ("true", ["true", "false"], "true", "exact match"),
    ("false.", ["true", "false"], "false", "partial match"),
    ("yes i think it is true", ["true", "false"], "true", "containment"),
    ("a beer bottle on a armchair",
     ["A beer bottle on a armchair", "A beer bottle under a armchair"],
     "A beer bottle on a armchair", "WhatsUp match"),
    ("", ["A beer bottle on a armchair", "A beer bottle under a armchair"],
     None, "WhatsUp empty"),
    ("a", ["A beer bottle on a armchair", "A beer bottle under a armchair"],
     None, "WhatsUp single char 'a'"),
]

all_pass = True
for raw, choices, expected, desc in tests:
    result = BaseVLM.match_response_to_choice(raw, choices)
    status = "PASS ✓" if result == expected else f"FAIL ✗ (got {result})"
    if result != expected:
        all_pass = False
    print(f"  {desc}: '{raw}' -> {result} {status}")

assert all_pass, "match_response_to_choice has failures!"
print("  All edge cases PASS ✓")

# ============================================================
# Part 4: Verify first_tokens_identical with real tokenizers
# ============================================================
print("\n" + "=" * 60)
print("PART 4: Verify first_tokens_identical with real tokenizers")
print("=" * 60)

from transformers import AutoTokenizer

tokenizers = {
    "OPT (BLIP2)": AutoTokenizer.from_pretrained("facebook/opt-2.7b", cache_dir=CACHE_DIR),
    "Qwen2": AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", cache_dir=CACHE_DIR, trust_remote_code=True),
    "LLaMA (LLaVA)": AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir=CACHE_DIR),
}

for tok_name, tok in tokenizers.items():
    print(f"\n  --- {tok_name} ---")

    # VSR: true/false -> should be DIFFERENT
    vsr_choices = ["true", "false"]
    vsr_identical = BaseVLM.first_tokens_identical(vsr_choices, tok)
    t_id = tok.encode("true", add_special_tokens=False)[0]
    f_id = tok.encode("false", add_special_tokens=False)[0]
    print(f"  VSR ['true','false']: identical={vsr_identical} (true={t_id}, false={f_id})")
    assert not vsr_identical, f"FAIL: VSR first tokens should differ for {tok_name}"

    # GQA yes/no -> should be DIFFERENT
    yn_choices = ["yes", "no"]
    yn_identical = BaseVLM.first_tokens_identical(yn_choices, tok)
    y_id = tok.encode("yes", add_special_tokens=False)[0]
    n_id = tok.encode("no", add_special_tokens=False)[0]
    print(f"  GQA ['yes','no']: identical={yn_identical} (yes={y_id}, no={n_id})")
    assert not yn_identical, f"FAIL: yes/no first tokens should differ for {tok_name}"

    # WhatsUp: all start with "A" -> should be IDENTICAL
    wu_choices = whatsup_samples[0].choices
    wu_identical = BaseVLM.first_tokens_identical(wu_choices, tok)
    a_id = tok.encode(wu_choices[0], add_special_tokens=False)[0]
    print(f"  WhatsUp: identical={wu_identical} (first token id={a_id})")
    assert wu_identical, f"FAIL: WhatsUp first tokens should be identical for {tok_name}"

    print(f"  {tok_name} PASS ✓")

# ============================================================
# Part 5: Test score_choices_by_sequence with each model
# ============================================================
print("\n" + "=" * 60)
print("PART 5: Test score_choices_by_sequence with each model")
print("=" * 60)

import gc

model_configs = [
    ("BLIP2", "src.models.blip2", "BLIP2Model", "Salesforce/blip2-opt-2.7b"),
    ("LLaVA", "src.models.llava", "LLaVAModel", "llava-hf/llava-1.5-7b-hf"),
    ("Qwen2-VL-2B", "src.models.qwen2vl", "Qwen2VLModel", "Qwen/Qwen2-VL-2B-Instruct"),
]

for model_name, module_path, class_name, hf_id in model_configs:
    print(f"\n--- Testing {model_name} ---")
    try:
        import importlib
        mod = importlib.import_module(module_path)
        ModelClass = getattr(mod, class_name)

        model = ModelClass(hf_id, cache_dir=CACHE_DIR, dtype="bfloat16")
        model.load()

        errors = []

        # Test 1: WhatsUp (sequence scoring path)
        print(f"  [WhatsUp] Testing sequence scoring...")
        s = whatsup_samples[0]
        out = model.predict_sample(s)
        vals = list(out.logits.values())
        unique_vals = len(set(round(v, 3) for v in vals))
        if unique_vals == 1:
            errors.append(f"WhatsUp logits still uniform: {out.logits}")
        else:
            print(f"    Logits differentiated ({unique_vals} unique values) ✓")
            print(f"    Probs: { {k[:25]: f'{v:.3f}' for k,v in out.probabilities.items()} }")
            print(f"    NC score: {out.nonconformity_score:.3f} (should NOT be 0.750)")
            if abs(out.nonconformity_score - 0.75) < 0.01:
                errors.append(f"WhatsUp NC score is ~0.75 (uniform), sequence scoring not working")

        # Test 2: VSR (first-token scoring path)
        print(f"  [VSR] Testing first-token scoring...")
        s = vsr_samples[0]
        out = model.predict_sample(s)
        if len(out.logits) != 2:
            errors.append(f"VSR should have 2 logits, got {len(out.logits)}")
        else:
            print(f"    Logits: {out.logits}")
            print(f"    Predicted: {out.predicted_answer}, Correct: {out.correct}")
            print(f"    Raw response: '{out.raw_response}'")

        # Test 3: GQA yes/no (first-token scoring path)
        print(f"  [GQA yes/no] Testing...")
        s = gqa_yn[0]
        out = model.predict_sample(s)
        if set(out.logits.keys()) != {"yes", "no"}:
            errors.append(f"GQA yes/no logits keys should be yes/no, got {list(out.logits.keys())}")
        else:
            print(f"    Logits: {out.logits}")
            print(f"    Predicted: {out.predicted_answer}, Raw: '{out.raw_response}'")

        # Test 4: GQA open-ended (open-ended path)
        print(f"  [GQA open] Testing...")
        s = gqa_open[0]
        out = model.predict_sample(s)
        if "__other__" not in out.probabilities:
            errors.append(f"GQA open should have '__other__' in probs, got {list(out.probabilities.keys())}")
        else:
            print(f"    Probs: {out.probabilities}")
            print(f"    Predicted: '{out.predicted_answer}', GT: '{s.answer}', Correct: {out.correct}")
            print(f"    Raw: '{out.raw_response}'")

        if errors:
            for e in errors:
                print(f"  FAIL ✗: {e}")
        else:
            print(f"  {model_name} ALL PATHS PASS ✓")

        # Cleanup
        del model.model
        del model
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        try:
            del model.model
            del model
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()

# ============================================================
# Part 6: Verify dataset integrity
# ============================================================
print("\n" + "=" * 60)
print("PART 6: Verify dataset / split integrity")
print("=" * 60)

splits = ["test", "cal", "ood_frame", "ood_concept", "ood_tailrisk",
          "ood_compositional", "ood_shifted_cal_test"]

for split in splits:
    path = os.path.join(SPLITS_ROOT, f"{split}.jsonl")
    if not os.path.exists(path):
        print(f"  {split}: NOT FOUND ✗")
        continue
    samples = BaseDataset.load_processed(path)
    by_ds = {}
    for s in samples:
        by_ds[s.dataset] = by_ds.get(s.dataset, 0) + 1

    # Check image paths exist (sample a few)
    import random
    random.seed(42)
    sample_check = random.sample(samples, min(10, len(samples)))
    missing_imgs = sum(1 for s in sample_check if not os.path.exists(s.image_path))

    print(f"  {split}: {len(samples)} total | {by_ds} | missing_img={missing_imgs}/10")

print("\n" + "=" * 60)
print("PRE-FLIGHT CHECK COMPLETE")
print("=" * 60)
