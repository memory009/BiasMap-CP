"""What'sUp dataset loader.
Source: https://github.com/amitakamath/whatsup_vlms
Task: Multiple-choice spatial QA — 4 caption options per image.

Format (controlled_images_dataset.json):
[{"image_path": "data/controlled_images/beer-bottle_on_armchair.jpeg",
  "caption_options": ["A beer bottle on a armchair", ...]}, ...]
The first caption_option is always the correct one.
"""
import os
import json
import re
from typing import List
from .base import BaseDataset, SpatialQASample

DEPTH_RELATIONS = {"in_front_of", "behind", "near", "far"}

RELATION_MAP = {
    "to the left of": "left", "to the right of": "right",
    "left of": "left", "right of": "right",
    "on top of": "above", "in front of": "in_front_of",
    "far from": "far", "next to": "beside",
    "on": "on", "under": "under",
}


class WhatsUpDataset(BaseDataset):

    def load(self) -> List[SpatialQASample]:
        self.samples = []

        # Primary: controlled_images_dataset.json
        ann_file = os.path.join(self.raw_dir, "controlled_images_dataset.json")
        if os.path.exists(ann_file):
            self._load_controlled(ann_file)

        # Also load controlled_clevr if available
        clevr_file = os.path.join(self.raw_dir, "controlled_clevr_dataset.json")
        if os.path.exists(clevr_file):
            self._load_controlled(clevr_file, scene_type="synthetic")

        # Fallback: search for any JSON file
        if not self.samples:
            for fname in os.listdir(self.raw_dir):
                if fname.endswith(".json") and "dataset" in fname:
                    self._load_controlled(os.path.join(self.raw_dir, fname))

        print(f"WhatsUp: loaded {len(self.samples)} samples")
        return self.samples

    def _load_controlled(self, ann_file: str, scene_type: str = "tabletop"):
        with open(ann_file) as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = list(data.values()) if isinstance(data, dict) else []

        idx = len(self.samples)
        for e in data:
            if not isinstance(e, dict):
                continue

            image_path_rel = e.get("image_path", e.get("image", ""))
            captions = e.get("caption_options", e.get("captions", []))
            if not captions:
                continue

            # First caption is always correct
            correct_caption = captions[0]
            relation = self._extract_relation(correct_caption)

            # Build absolute image path
            img_path = self._resolve_image(image_path_rel)

            # Build question: which caption correctly describes the image?
            question = "Which caption best describes the spatial relationship in the image?"

            sample = SpatialQASample(
                id=f"whatsup_{idx:06d}",
                dataset="whatsup",
                image_path=img_path,
                question=question,
                answer=correct_caption,
                relation_type=self._normalize_relation(relation),
                choices=captions,
                scene_type=scene_type,
                depth_ambiguity=self._normalize_relation(relation) in DEPTH_RELATIONS,
            )
            self.samples.append(sample)
            idx += 1

    def _resolve_image(self, image_path_rel: str) -> str:
        if not image_path_rel:
            return ""
        # Strip leading 'data/' prefix from annotation
        clean = image_path_rel.replace("data/", "", 1).lstrip("/")

        for base in [
            self.raw_dir,
            os.path.join(self.raw_dir, "data"),
        ]:
            p = os.path.join(base, clean)
            if os.path.exists(p):
                return p

        # Try just the filename
        fname = os.path.basename(clean)
        for dirpath, _, files in os.walk(self.raw_dir):
            if fname in files:
                return os.path.join(dirpath, fname)

        return os.path.join(self.raw_dir, clean)

    def _extract_relation(self, caption: str) -> str:
        c = caption.lower()
        for kw in ["to the left of", "to the right of", "in front of",
                   "on top of", "far from", "next to",
                   "left of", "right of",
                   "above", "below", "behind", "on", "under",
                   "beside", "near"]:
            if kw in c:
                return kw
        return "unknown"

    def _normalize_relation(self, relation: str) -> str:
        return RELATION_MAP.get(relation.lower().strip(),
                                relation.lower().replace(" ", "_"))
