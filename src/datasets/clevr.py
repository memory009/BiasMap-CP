"""CLEVR dataset loader — spatial questions only.
Source: https://cs.stanford.edu/people/jcjohns/clevr/
Task: Synthetic VQA with spatial reasoning.
"""
import os
import json
from typing import List
from .base import BaseDataset, SpatialQASample

SPATIAL_KEYWORDS = [
    "left", "right", "above", "below", "behind", "in front of",
    "front", "back", "near", "far",
]

DEPTH_RELATIONS = {"in_front_of", "behind", "near", "far"}


class CLEVRDataset(BaseDataset):

    def load(self, max_samples: int = 50000) -> List[SpatialQASample]:
        self.samples = []

        # CLEVR structure: CLEVR_v1.0/questions/CLEVR_{split}_questions.json
        #                  CLEVR_v1.0/images/{split}/{image_name}
        clevr_root = self.raw_dir
        # Try to find the CLEVR_v1.0 directory
        for candidate in [clevr_root,
                          os.path.join(clevr_root, "CLEVR_v1.0")]:
            if os.path.exists(os.path.join(candidate, "questions")):
                clevr_root = candidate
                break

        q_dir = os.path.join(clevr_root, "questions")
        img_dir = os.path.join(clevr_root, "images")

        q_files = []
        if os.path.exists(q_dir):
            for f in os.listdir(q_dir):
                if f.endswith(".json"):
                    q_files.append(os.path.join(q_dir, f))
        else:
            for root, _, files in os.walk(self.raw_dir):
                for f in files:
                    if "question" in f.lower() and f.endswith(".json"):
                        q_files.append(os.path.join(root, f))

        print(f"CLEVR: found question files: {q_files}")

        idx = 0
        for qfile in q_files:
            if idx >= max_samples:
                break
            split_name = "train"
            if "val" in qfile:
                split_name = "val"
            elif "test" in qfile:
                split_name = "test"

            try:
                with open(qfile) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  Error: {e}")
                continue

            questions = data.get("questions", data if isinstance(data, list) else [])
            for q in questions:
                if idx >= max_samples:
                    break
                question_text = q.get("question", "")
                if not self._is_spatial(question_text):
                    continue

                answer = str(q.get("answer", "")).lower()
                image_fname = q.get("image_filename", q.get("image", ""))
                img_path = self._find_image(img_dir, image_fname, split_name)

                relation = self._extract_relation(question_text)
                normalized_rel = self._normalize_relation(relation)

                sample = SpatialQASample(
                    id=f"clevr_{idx:06d}",
                    dataset="clevr",
                    image_path=img_path,
                    question=question_text,
                    answer=answer,
                    relation_type=normalized_rel,
                    scene_type="synthetic",
                    depth_ambiguity=normalized_rel in DEPTH_RELATIONS,
                )
                self.samples.append(sample)
                idx += 1

        print(f"CLEVR: loaded {len(self.samples)} spatial samples")
        return self.samples

    def _is_spatial(self, question: str) -> bool:
        q = question.lower()
        return any(kw in q for kw in SPATIAL_KEYWORDS)

    def _extract_relation(self, question: str) -> str:
        q = question.lower()
        for kw in ["in front of", "behind", "to the left of", "to the right of",
                   "left of", "right of", "above", "below", "near", "far"]:
            if kw in q:
                return kw
        for kw in ["left", "right", "front", "back"]:
            if kw in q:
                return kw
        return "unknown"

    def _normalize_relation(self, rel: str) -> str:
        mapping = {
            "to the left of": "left", "left of": "left", "left": "left",
            "to the right of": "right", "right of": "right", "right": "right",
            "in front of": "in_front_of", "front": "in_front_of",
            "back": "behind", "far from": "far",
        }
        return mapping.get(rel.lower().strip(), rel.replace(" ", "_"))

    def _find_image(self, img_dir: str, fname: str, split: str) -> str:
        if not fname:
            return ""
        for base in [os.path.join(img_dir, split), img_dir,
                     os.path.join(img_dir, "train"), os.path.join(img_dir, "val")]:
            p = os.path.join(base, fname)
            if os.path.exists(p):
                return p
        return os.path.join(img_dir, split, fname)
