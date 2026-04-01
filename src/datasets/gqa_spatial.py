"""GQA Spatial Subset loader.
Filters GQA questions to only spatial reasoning questions.
Source: https://cs.stanford.edu/people/doersch/gqa/
"""
import os
import json
from typing import List, Dict
from .base import BaseDataset, SpatialQASample

SPATIAL_KEYWORDS = [
    "left of", "right of", "above", "below", "behind", "in front of",
    "next to", "near", "far from", "on top of", "under", "beside",
    "between", "inside", "outside", "across from", "adjacent to",
    "to the left", "to the right",
]

DEPTH_RELATIONS = {"in_front_of", "behind", "near", "far"}

RELATION_MAP = {
    "to the left of": "left", "to the right of": "right",
    "left of": "left", "right of": "right",
    "on top of": "above", "in front of": "in_front_of",
    "far from": "far", "next to": "beside",
}


class GQASpatialDataset(BaseDataset):

    def load(self, max_samples: int = 100000) -> List[SpatialQASample]:
        self.samples = []
        raw_dir = self.raw_dir

        # Find question files
        q_files = []
        for root, _, files in os.walk(raw_dir):
            for f in files:
                if "question" in f.lower() and f.endswith(".json"):
                    q_files.append(os.path.join(root, f))

        print(f"GQA: found question files: {q_files[:5]}")

        idx = 0
        for qfile in q_files:
            if idx >= max_samples:
                break
            print(f"  Processing {qfile}...")
            try:
                with open(qfile) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  Error loading {qfile}: {e}")
                continue

            # GQA format: {qid: {question: ..., answer: ..., imageId: ..., ...}}
            if isinstance(data, dict):
                items = data.items()
            elif isinstance(data, list):
                items = enumerate(data)
            else:
                continue

            for qid, entry in items:
                if idx >= max_samples:
                    break
                if not isinstance(entry, dict):
                    continue

                question = entry.get("question", "")
                if not self._is_spatial(question):
                    continue

                answer = str(entry.get("answer", "")).lower()
                image_id = entry.get("imageId", entry.get("image_id", ""))
                img_path = self._find_image(raw_dir, image_id)

                relation = self._extract_relation(question)
                normalized_rel = self._normalize_relation(relation)

                sample = SpatialQASample(
                    id=f"gqa_{qid}",
                    dataset="gqa",
                    image_path=img_path,
                    question=question,
                    answer=answer,
                    relation_type=normalized_rel,
                    depth_ambiguity=normalized_rel in DEPTH_RELATIONS,
                    scene_type="outdoor",  # GQA is mostly real-world outdoor/indoor
                )
                self.samples.append(sample)
                idx += 1

        print(f"GQA: loaded {len(self.samples)} spatial samples (from {max_samples} max)")
        return self.samples

    def _is_spatial(self, question: str) -> bool:
        q_lower = question.lower()
        return any(kw in q_lower for kw in SPATIAL_KEYWORDS)

    def _extract_relation(self, question: str) -> str:
        q = question.lower()
        for kw in SPATIAL_KEYWORDS:
            if kw in q:
                return kw
        return "unknown"

    def _normalize_relation(self, relation: str) -> str:
        rel = relation.lower().strip()
        return RELATION_MAP.get(rel, rel.replace(" ", "_"))

    def _find_image(self, raw_dir: str, image_id: str) -> str:
        if not image_id:
            return ""
        # GQA images are typically in images/ subfolder as {imageId}.jpg
        for base in [os.path.join(raw_dir, "images"),
                     os.path.join(raw_dir, "allImages", "images"),
                     raw_dir]:
            for ext in [".jpg", ".png", ""]:
                p = os.path.join(base, f"{image_id}{ext}")
                if os.path.exists(p):
                    return p
        return os.path.join(raw_dir, "images", f"{image_id}.jpg")
