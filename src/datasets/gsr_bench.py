"""GSR-Bench (Grounded Spatial Relation) dataset loader.
Source: https://github.com/WenqiRajabi/GSR-BENCH
Task: Spatial relation classification with bounding boxes.
"""
import os
import json
from typing import List, Optional
from .base import BaseDataset, SpatialQASample

DEPTH_RELATIONS = {"in_front_of", "behind", "near", "far"}


class GSRBenchDataset(BaseDataset):

    def load(self) -> List[SpatialQASample]:
        self.samples = []

        json_files = []
        for root, _, files in os.walk(self.raw_dir):
            for f in files:
                if f.endswith((".json", ".jsonl")):
                    json_files.append(os.path.join(root, f))

        print(f"GSR-Bench: found files: {json_files[:5]}")

        idx = 0
        for fpath in json_files:
            try:
                with open(fpath) as f:
                    content = f.read().strip()
                    if content.startswith("["):
                        data = json.loads(content)
                        items = data
                    else:
                        items = [json.loads(line) for line in content.split("\n") if line.strip()]
            except Exception as e:
                print(f"  skip {fpath}: {e}")
                continue

            for e in items:
                if not isinstance(e, dict):
                    continue

                image_file = e.get("image", e.get("image_path", ""))
                relation = e.get("relation", e.get("predicate", "unknown"))
                subject = e.get("subject", e.get("object1", {}) if isinstance(e.get("object1"), dict) else None)
                obj = e.get("object", e.get("object2", {}) if isinstance(e.get("object2"), dict) else None)

                subject_name = subject.get("name", "") if isinstance(subject, dict) else str(subject or "")
                object_name = obj.get("name", "") if isinstance(obj, dict) else str(obj or "")
                subject_bbox = self._parse_bbox(subject)
                object_bbox = self._parse_bbox(obj)

                question = f"What is the spatial relation of {subject_name} with respect to {object_name}?"
                answer = self._normalize_relation(relation)

                img_path = self._resolve_image(image_file)

                # Compute occlusion if bboxes available
                occlusion = self._compute_occlusion(subject_bbox, object_bbox)
                size_ratio = self._compute_size_ratio(subject_bbox, object_bbox)

                sample = SpatialQASample(
                    id=f"gsr_{idx:06d}",
                    dataset="gsr_bench",
                    image_path=img_path,
                    question=question,
                    answer=answer,
                    relation_type=answer,
                    subject=subject_name,
                    object=object_name,
                    subject_bbox=subject_bbox,
                    object_bbox=object_bbox,
                    occlusion_level=occlusion,
                    object_size_ratio=size_ratio,
                    depth_ambiguity=answer in DEPTH_RELATIONS,
                )
                self.samples.append(sample)
                idx += 1

        print(f"GSR-Bench: loaded {len(self.samples)} samples")
        return self.samples

    def _resolve_image(self, image_file: str) -> str:
        if not image_file:
            return ""
        if os.path.isabs(image_file) and os.path.exists(image_file):
            return image_file
        for base in [self.raw_dir, os.path.join(self.raw_dir, "images")]:
            candidate = os.path.join(base, image_file)
            if os.path.exists(candidate):
                return candidate
        return os.path.join(self.raw_dir, "images", image_file)

    def _parse_bbox(self, obj) -> Optional[List[float]]:
        if not isinstance(obj, dict):
            return None
        for key_combo in [("x1", "y1", "x2", "y2"), ("xmin", "ymin", "xmax", "ymax"),
                          ("left", "top", "right", "bottom")]:
            if all(k in obj for k in key_combo):
                return [float(obj[k]) for k in key_combo]
        if "bbox" in obj:
            bbox = obj["bbox"]
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                return [float(v) for v in bbox]
        return None

    def _compute_occlusion(self, bbox1, bbox2) -> Optional[str]:
        if bbox1 is None or bbox2 is None:
            return None
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        if x2 <= x1 or y2 <= y1:
            return "none"
        inter = (x2 - x1) * (y2 - y1)
        area1 = max(1e-6, (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
        iou = inter / area1
        if iou < 0.1:
            return "none"
        elif iou < 0.3:
            return "partial"
        else:
            return "heavy"

    def _compute_size_ratio(self, bbox1, bbox2) -> Optional[float]:
        if bbox1 is None or bbox2 is None:
            return None
        area1 = max(1e-9, (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
        area2 = max(1e-9, (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]))
        return area1 / area2

    def _normalize_relation(self, relation: str) -> str:
        mapping = {
            "to the left of": "left", "to the right of": "right",
            "left of": "left", "right of": "right",
            "on top of": "above", "in front of": "in_front_of",
            "far from": "far", "next to": "beside",
        }
        rel = relation.lower().strip()
        return mapping.get(rel, rel.replace(" ", "_"))
