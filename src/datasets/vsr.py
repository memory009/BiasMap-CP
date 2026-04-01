"""VSR (Visual Spatial Reasoning) dataset loader.
Source: https://github.com/cambridgeltl/visual-spatial-reasoning
Task: Binary True/False for spatial captions.
"""
import os
import json
from typing import List
from .base import BaseDataset, SpatialQASample

DEPTH_RELATIONS = {"in front of", "behind", "near", "far", "far from"}


class VSRDataset(BaseDataset):

    def load(self) -> List[SpatialQASample]:
        # VSR has train/dev/test splits in JSON files under data/ subdir
        # Each entry: {"image": "path/to/img.jpg", "caption": "...", "label": 0/1}
        self.samples = []
        data_dir = os.path.join(self.raw_dir, "data")
        if not os.path.exists(data_dir):
            # fallback: look for json files directly
            data_dir = self.raw_dir

        split_files = []
        for fname in ["train.jsonl", "dev.jsonl", "test.jsonl",
                      "train.json", "dev.json", "test.json"]:
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath):
                split_files.append(fpath)

        if not split_files:
            # try recursive search
            for root, _, files in os.walk(self.raw_dir):
                for f in files:
                    if f.endswith((".json", ".jsonl")) and any(
                        s in f for s in ["train", "dev", "test", "vsr"]
                    ):
                        split_files.append(os.path.join(root, f))

        print(f"VSR: found split files: {split_files}")

        idx = 0
        for fpath in split_files:
            split_name = os.path.splitext(os.path.basename(fpath))[0]
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Handle both array and object formats
                    if isinstance(entry, list):
                        entries = entry
                    else:
                        entries = [entry]

                    for e in entries:
                        image_file = e.get("image", e.get("image_path", ""))
                        caption = e.get("caption", e.get("question", ""))
                        label = e.get("label", e.get("answer", 0))
                        relation = e.get("relation", self._extract_relation(caption))

                        # Build image path
                        img_path = self._resolve_image(image_file)

                        answer = "true" if int(label) == 1 else "false"
                        subj, obj = self._extract_subject_object(caption, relation)
                        sample = SpatialQASample(
                            id=f"vsr_{split_name}_{idx:06d}",
                            dataset="vsr",
                            image_path=img_path,
                            question=f'Is the following statement true or false? "{caption}"',
                            answer=answer,
                            relation_type=self._normalize_relation(relation),
                            choices=["true", "false"],
                            subject=subj,
                            object=obj,
                            depth_ambiguity=self._normalize_relation(relation) in DEPTH_RELATIONS,
                        )
                        self.samples.append(sample)
                        idx += 1

        print(f"VSR: loaded {len(self.samples)} samples")
        return self.samples

    def _resolve_image(self, image_file: str) -> str:
        if os.path.isabs(image_file) and os.path.exists(image_file):
            return image_file
        # Common VSR image locations
        for base in [
            self.raw_dir,
            os.path.join(self.raw_dir, "images"),
            os.path.join(self.raw_dir, "data", "images"),
        ]:
            candidate = os.path.join(base, image_file)
            if os.path.exists(candidate):
                return candidate
        # Return as-is if not found (will be checked at inference time)
        return os.path.join(self.raw_dir, "images", image_file)

    def _extract_subject_object(self, caption: str, relation: str) -> tuple:
        """Parse 'The [SUBJECT] [RELATION] the [OBJECT]' from VSR caption."""
        import re
        cap = caption.strip().rstrip('.')
        rel_lower = relation.lower().strip()
        cap_lower = cap.lower()

        rel_pos = cap_lower.find(rel_lower)
        if rel_pos == -1:
            return None, None

        # Subject: strip leading article and trailing copula
        pre = cap[:rel_pos].strip()
        pre = re.sub(r'^(?:the |a |an )', '', pre, flags=re.IGNORECASE)
        pre = re.sub(r'\s+(?:is|are|was|were)$', '', pre, flags=re.IGNORECASE)
        subj = pre.strip().lower() or None

        # Object: strip leading article and trailing punctuation
        post = cap[rel_pos + len(rel_lower):].strip()
        post = re.sub(r'^(?:the |a |an )', '', post, flags=re.IGNORECASE)
        obj = post.strip().lower() or None

        return subj, obj

    def _extract_relation(self, caption: str) -> str:
        caption_lower = caption.lower()
        relations = [
            "to the left of", "to the right of", "above", "below",
            "in front of", "behind", "next to", "near", "far from",
            "on top of", "under", "beside", "between", "inside", "outside"
        ]
        for rel in relations:
            if rel in caption_lower:
                return rel
        return "unknown"

    def _normalize_relation(self, relation: str) -> str:
        mapping = {
            "to the left of": "left",
            "to the right of": "right",
            "on top of": "above",
            "in front of": "in_front_of",
            "far from": "far",
            "next to": "beside",
        }
        rel = relation.lower().strip()
        return mapping.get(rel, rel.replace(" ", "_"))
