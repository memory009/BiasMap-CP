"""Qwen2-VL model wrapper for BiasMap-CP."""
import os
import re
import torch
import numpy as np
from typing import List, Dict, Optional

from .base_vlm import BaseVLM
from ..datasets.base import SpatialQASample, ModelOutput


class Qwen2VLModel(BaseVLM):

    def _score_choices_by_generate(self, choices, inputs):
        """Score choices using generate(output_scores=True).
        Qwen2-VL's 3D position encoding prevents manual input_ids extension,
        so we use the model's own generate to get per-step logits and score
        each choice by average log-prob of its token sequence."""
        tokenizer = self.processor.tokenizer

        # Find max choice length in tokens
        choice_token_ids = [tokenizer.encode(c, add_special_tokens=False) for c in choices]
        max_len = max(len(t) for t in choice_token_ids)

        with torch.no_grad():
            gen_out = self.model.generate(
                **inputs,
                max_new_tokens=max_len,
                min_new_tokens=max_len,  # Force generation past divergence point
                do_sample=False,
                temperature=None,
                top_p=None,
                output_scores=True,
                return_dict_in_generate=True,
            )

        scores = gen_out.scores  # tuple of (batch, vocab_size), one per step

        choice_scores = {}
        for choice, toks in zip(choices, choice_token_ids):
            if not toks:
                choice_scores[choice] = -1e9
                continue
            log_probs = []
            for i, tok_id in enumerate(toks):
                if i < len(scores):
                    log_p = torch.log_softmax(scores[i][0], dim=-1)[tok_id].item()
                    log_probs.append(log_p)
            choice_scores[choice] = sum(log_probs) / max(len(log_probs), 1)
        return choice_scores

    def load(self):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        print(f"Loading {self.model_id}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"  Loaded {self.model_id} on {self.device}")
        return self

    def predict_sample(self, sample: SpatialQASample) -> ModelOutput:
        image = self.load_image(sample.image_path)
        prompt = self.build_prompt(sample)
        choices = self.infer_choices(sample)

        # Build Qwen2-VL message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            from qwen_vl_utils import process_vision_info
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
        except Exception as e:
            # Fallback: basic processor call
            inputs = self.processor(
                text=prompt, images=image, return_tensors="pt"
            ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_logits = outputs.logits[0, -1, :]  # (vocab_size,)

            # Generate raw response
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                temperature=None,
                top_p=None,
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
        if self.first_tokens_identical(choices, self.processor.tokenizer):
            # Qwen2-VL cannot extend input_ids directly (3D position encoding).
            # Use generate with output_scores to get per-step logits, then
            # score choices at their first diverging token position.
            choice_logits = self._score_choices_by_generate(
                choices, inputs
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
