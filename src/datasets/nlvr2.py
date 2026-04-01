"""NLVR2 dataset loader.
Source: https://lil.nlp.cornell.edu/nlvr/
Task: Image-sentence matching (True/False) with two images.
We focus on spatial statements.
"""
import os
import json
from typing import List
from .base import BaseDataset, SpatialQASample

SPATIAL_KEYWORDS = [
    "left", "right", "above", "below", "behind", "in front",
    "next to", "near", "far", "top", "bottom", "beside",
    "between", "inside", "outside", "front",
]


class NLVR2Dataset(BaseDataset):

    def load(self) -> List[SpatialQASample]:
        self.samples = []

        # NLVR2: jsonl or json files with sentence, label, identifier
        jsonl_files = []
        for root, _, files in os.walk(self.raw_dir):
            for f in files:
                if f.endswith((".jsonl", ".json")):
                    jsonl_files.append(os.path.join(root, f))

        print(f"NLVR2: found files: {jsonl_files[:5]}")

        idx = 0
        for fpath in jsonl_files:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except:
                        continue

                    sentence = e.get("sentence", "")
                    if not self._is_spatial(sentence):
                        continue

                    label = str(e.get("label", "False")).lower()
                    answer = "true" if label == "true" else "false"
                    identifier = e.get("identifier", f"nlvr2_{idx}")
                    directory = e.get("directory", "")

                    # NLVR2 uses two images; use the first as primary
                    img0 = e.get("img0", e.get("left_url", ""))
                    img1 = e.get("img1", e.get("right_url", ""))
                    img_path = self._find_image(directory, identifier, "0")

                    sample = SpatialQASample(
                        id=f"nlvr2_{identifier}",
                        dataset="nlvr2",
                        image_path=img_path,
                        question=f'Are the following spatial statements true? "{sentence}"',
                        answer=answer,
                        relation_type=self._extract_relation(sentence),
                        choices=["true", "false"],
                        depth_ambiguity=False,
                    )
                    self.samples.append(sample)
                    idx += 1

        print(f"NLVR2: loaded {len(self.samples)} spatial samples")
        return self.samples

    def _is_spatial(self, sentence: str) -> bool:
        s = sentence.lower()
        return any(kw in s for kw in SPATIAL_KEYWORDS)

    def _extract_relation(self, sentence: str) -> str:
        s = sentence.lower()
        for kw in ["to the left of", "to the right of", "left of", "right of",
                   "above", "below", "in front of", "behind", "next to",
                   "near", "beside", "between"]:
            if kw in s:
                return kw.replace(" ", "_")
        for kw in ["left", "right", "top", "bottom", "front", "back"]:
            if kw in s:
                return kw
        return "spatial"

    def _find_image(self, directory: str, identifier: str, suffix: str) -> str:
        # NLVR2 image path: images/{split}/{directory}/{identifier}-img{suffix}.png
        for base in [self.raw_dir, os.path.join(self.raw_dir, "images")]:
            for split in ["train", "dev", "test1", ""]:
                for ext in [".png", ".jpg"]:
                    parts = [base]
                    if split:
                        parts.append(split)
                    if directory:
                        parts.append(directory)
                    fname = f"{identifier}-img{suffix}{ext}"
                    p = os.path.join(*[str(x) for x in parts], fname)
                    if os.path.exists(p):
                        return p
        return os.path.join(self.raw_dir, "images", str(directory), f"{identifier}-img{suffix}.png")
