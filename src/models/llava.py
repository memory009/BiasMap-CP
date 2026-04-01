"""LLaVA-1.5 model wrapper for BiasMap-CP."""
import torch
import numpy as np
from typing import List, Optional

from .base_vlm import BaseVLM
from ..datasets.base import SpatialQASample, ModelOutput


class LLaVAModel(BaseVLM):

    def load(self):
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        print(f"Loading {self.model_id}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            torch_dtype=self.dtype,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        print(f"  Loaded {self.model_id}")
        return self

    def predict_sample(self, sample: SpatialQASample) -> ModelOutput:
        image = self.load_image(sample.image_path)
        prompt_text = self.build_prompt(sample)
        choices = self.infer_choices(sample)

        # LLaVA uses a specific prompt format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        try:
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            inputs = self.processor(
                images=image, text=prompt, return_tensors="pt"
            ).to(self.model.device, dtype=self.dtype)
        except Exception:
            # Fallback format
            prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
            inputs = self.processor(
                images=image, text=prompt, return_tensors="pt"
            ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_logits = outputs.logits[0, -1, :]

            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
            )
            input_len = inputs["input_ids"].shape[1]
            raw_response = self.processor.decode(
                gen_ids[0][input_len:], skip_special_tokens=True
            ).strip().lower()

        # Open-ended: use raw_response matching + first-token confidence
        if self.is_open_ended(choices):
            return self.compute_open_ended_result(
                raw_response, sample, last_logits, self.processor.tokenizer
            )

        # Closed-ended: score each choice
        # Use sequence scoring if all choices share the same first token
        if self.first_tokens_identical(choices, self.processor.tokenizer):
            choice_logits = self.score_choices_by_sequence(
                choices, self.model, self.processor.tokenizer,
                inputs["input_ids"], full_inputs=inputs
            )
        else:
            choice_logits = {}
            for choice in choices:
                tokens = self.processor.tokenizer.encode(
                    choice, add_special_tokens=False
                )
                if tokens:
                    choice_logits[choice] = float(last_logits[tokens[0]].item())
                else:
                    choice_logits[choice] = -1e9

        probs = self.softmax_dict(choice_logits)
        predicted = max(probs, key=probs.get)

        # Try to match raw_response to a choice (skip empty responses)
        matched = self.match_response_to_choice(raw_response, choices)
        if matched:
            predicted = matched

        correct = self.normalize_answer(predicted) == self.normalize_answer(sample.answer)
        nc_score = self.get_nonconformity_score(probs, sample.answer)

        return ModelOutput(
            sample_id=sample.id,
            dataset=sample.dataset,
            split="unknown",
            model=self.model_id,
            logits=choice_logits,
            probabilities=probs,
            predicted_answer=predicted,
            correct=correct,
            nonconformity_score=nc_score,
            raw_response=raw_response,
        )
