"""Abstract VLM interface for BiasMap-CP."""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import os
import torch
import numpy as np
from PIL import Image

from ..datasets.base import SpatialQASample, ModelOutput


SPATIAL_PROMPT = """Look at the image carefully. Answer the following spatial reasoning question.

Question: {question}

Choose the correct answer from: {choices}

Answer with ONLY the letter or the exact answer text, nothing else."""

BINARY_PROMPT = """Look at the image. Is the following spatial statement true or false?

Statement: "{caption}"

Answer with ONLY "true" or "false"."""

OPEN_PROMPT = """Look at the image carefully. Answer the following spatial reasoning question with a short answer.

Question: {question}

Answer:"""


class BaseVLM(ABC):
    """Abstract base class for VLM inference."""

    def __init__(self, model_id: str, cache_dir: Optional[str] = None,
                 dtype: str = "bfloat16", device: str = "auto"):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.dtype_str = dtype
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.device = device
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self):
        """Load model and processor."""
        pass

    @abstractmethod
    def predict_sample(self, sample: SpatialQASample) -> ModelOutput:
        """Run inference on a single sample, returning ModelOutput with logits."""
        pass

    def predict_batch(self, samples: List[SpatialQASample],
                      batch_size: int = 4) -> List[ModelOutput]:
        """Run inference on a list of samples."""
        results = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            for sample in batch:
                try:
                    out = self.predict_sample(sample)
                except Exception as e:
                    print(f"  Error on sample {sample.id}: {e}")
                    out = self._error_output(sample)
                results.append(out)
            if (i // batch_size) % 10 == 0:
                print(f"  {i + len(batch)}/{len(samples)} done")
        return results

    def _error_output(self, sample: SpatialQASample) -> ModelOutput:
        choices = sample.choices or [sample.answer, "unknown"]
        uniform = 1.0 / len(choices)
        probs = {c: uniform for c in choices}
        return ModelOutput(
            sample_id=sample.id,
            dataset=sample.dataset,
            split="unknown",
            model=self.model_id,
            logits=probs,
            probabilities=probs,
            predicted_answer="unknown",
            correct=False,
            nonconformity_score=1.0,
            raw_response="ERROR",
        )

    def infer_choices(self, sample: SpatialQASample) -> List[str]:
        """Infer answer choices for a sample when choices are not provided.

        Handles three cases:
        1. Binary true/false (VSR): ["true", "false"]
        2. Binary yes/no (GQA yes/no questions): ["yes", "no"]
        3. Open-ended (GQA open questions): [answer] — single-element list
           signals that raw_response matching should be used for accuracy,
           and first-token probability for CP score.
        """
        if sample.choices:
            return sample.choices
        answer_lower = sample.answer.lower().strip()
        if answer_lower in ("true", "false"):
            return ["true", "false"]
        if answer_lower in ("yes", "no"):
            return ["yes", "no"]
        # Open-ended: return single-element list to signal open-ended mode
        return [answer_lower]

    def is_open_ended(self, choices: List[str]) -> bool:
        """Check if choices indicate open-ended (single GT answer, no alternatives)."""
        return len(choices) == 1

    def build_prompt(self, sample: SpatialQASample) -> str:
        choices = sample.choices or self.infer_choices(sample)
        if len(choices) == 2 and set(c.lower() for c in choices) == {"true", "false"}:
            q = sample.question
            if '"' in q:
                stmt = q.split('"')[1]
            else:
                stmt = q
            return BINARY_PROMPT.format(caption=stmt)
        elif len(choices) >= 2:
            choices_str = " / ".join(choices)
            return SPATIAL_PROMPT.format(question=sample.question, choices=choices_str)
        else:
            return OPEN_PROMPT.format(question=sample.question)

    def compute_open_ended_result(self, raw_response: str, sample: SpatialQASample,
                                   last_logits, tokenizer) -> ModelOutput:
        """Handle open-ended questions: use raw_response for accuracy,
        first-token probability of GT for CP nonconformity score."""
        answer = sample.answer.lower().strip()
        resp = raw_response.strip().lower()

        # Accuracy: exact match or containment
        correct = (resp == answer) or (answer in resp) or (resp in answer)

        # CP: compute probability of GT answer's first token vs uniform over vocab
        # Use sigmoid of the GT token logit as a confidence proxy
        gt_tokens = tokenizer.encode(answer, add_special_tokens=False)
        if gt_tokens:
            gt_logit = float(last_logits[gt_tokens[0]].item())
            # Also get "yes" and "no" logits as reference points for yes/no-like answers
            confidence = float(torch.sigmoid(torch.tensor(gt_logit)).item())
        else:
            confidence = 0.5

        # Store as 2-class probability for CP compatibility
        probs = {answer: confidence, "__other__": 1.0 - confidence}
        nc_score = 1.0 - confidence

        return ModelOutput(
            sample_id=sample.id,
            dataset=sample.dataset,
            split="unknown",
            model=self.model_id,
            logits={answer: gt_logit if gt_tokens else 0.0, "__other__": 0.0},
            probabilities=probs,
            predicted_answer=resp,
            correct=correct,
            nonconformity_score=nc_score,
            raw_response=raw_response,
        )

    @staticmethod
    def normalize_answer(text: str) -> str:
        """Normalize answer text for matching."""
        import re
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def match_response_to_choice(raw_response: str, choices: List[str]) -> Optional[str]:
        """Safely match raw_response to one of the choices.
        Returns matched choice or None. Skips empty/too-short responses."""
        resp = raw_response.strip().lower()
        if len(resp) < 2:  # skip empty or single-char responses
            return None
        for choice in choices:
            if choice.lower() in resp or resp in choice.lower():
                return choice
        return None

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        if not image_path or not os.path.exists(image_path):
            return Image.new("RGB", (224, 224), (128, 128, 128))
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  Warning: cannot load {image_path}: {e}")
            return Image.new("RGB", (224, 224), (128, 128, 128))

    @staticmethod
    def softmax_dict(logits: Dict[str, float]) -> Dict[str, float]:
        vals = np.array(list(logits.values()), dtype=np.float64)
        vals = vals - vals.max()
        exp_vals = np.exp(vals)
        probs = exp_vals / exp_vals.sum()
        return {k: float(p) for k, p in zip(logits.keys(), probs)}

    @staticmethod
    def first_tokens_identical(choices: List[str], tokenizer) -> bool:
        """Check if all choices share the same first token."""
        first_ids = set()
        for c in choices:
            toks = tokenizer.encode(c, add_special_tokens=False)
            if toks:
                first_ids.add(toks[0])
        return len(first_ids) <= 1

    @staticmethod
    def score_choices_by_sequence(choices: List[str], model, tokenizer,
                                  input_ids, full_inputs=None) -> Dict[str, float]:
        """Score each choice by average log-probability of its full token sequence.
        Used when first-token scoring fails (e.g. all choices share first token).

        Args:
            full_inputs: Complete model inputs dict (for VLMs that need pixel_values etc).
                         If None, uses input_ids only.
        """
        import torch
        choice_scores = {}
        for choice in choices:
            choice_ids = tokenizer.encode(choice, add_special_tokens=False)
            if not choice_ids:
                choice_scores[choice] = -1e9
                continue
            # Build input + choice sequence
            choice_tensor = torch.tensor(choice_ids, device=input_ids.device).unsqueeze(0)
            extended_ids = torch.cat([input_ids, choice_tensor], dim=1)

            if full_inputs is not None:
                # Pass all model inputs (pixel_values, etc.) but drop attention_mask.
                # Dropping attention_mask lets the model create a default all-ones mask
                # matching the new input_ids length. This avoids size mismatches in
                # models that expand image tokens internally (e.g. Qwen2-VL).
                model_inputs = {k: v for k, v in full_inputs.items()
                                if k not in ("input_ids", "attention_mask")}
                model_inputs["input_ids"] = extended_ids
            else:
                model_inputs = {"input_ids": extended_ids}

            with torch.no_grad():
                out = model(**model_inputs)
            logits = out.logits[0]  # (seq_len, vocab_size)
            # Score = average log-prob of each choice token given prefix
            log_probs = []
            input_len = input_ids.shape[1]
            for i, tok_id in enumerate(choice_ids):
                pos = input_len - 1 + i  # position of logit predicting this token
                if pos < logits.shape[0]:
                    log_p = torch.log_softmax(logits[pos], dim=-1)[tok_id].item()
                    log_probs.append(log_p)
            choice_scores[choice] = sum(log_probs) / max(len(log_probs), 1)
        return choice_scores

    @staticmethod
    def get_nonconformity_score(probs: Dict[str, float], true_answer: str) -> float:
        """1 - probability of true answer (standard conformal nonconformity score)."""
        p = probs.get(true_answer, 0.0)
        return 1.0 - p
