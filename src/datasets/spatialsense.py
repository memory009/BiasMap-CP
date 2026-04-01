"""SpatialSense dataset loader.
Source: https://github.com/sled-group/SpatialSense (requires email request)
Task: Spatial relation prediction between object pairs.
"""
import os
import json
from typing import List
from .base import BaseDataset, SpatialQASample

DEPTH_RELATIONS = {"in_front_of", "behind", "near", "far"}


class SpatialSenseDataset(BaseDataset):

    def load(self) -> List[SpatialQASample]:
        self.samples = []

        # SpatialSense: annotations.json with {annotations: [{image, relation, subject, object}]}
        ann_file = None
        for candidate in ["annotations.json", "spatialsense.json", "data.json"]:
            p = os.path.join(self.raw_dir, candidate)
            if os.path.exists(p):
                ann_file = p
                break

        if ann_file is None:
            # Try recursive search
            for root, _, files in os.walk(self.raw_dir):
                for f in files:
                    if f.endswith(".json"):
                        ann_file = os.path.join(root, f)
                        break
                if ann_file:
                    break

        if ann_file is None:
            print(f"SpatialSense: no annotation file found in {self.raw_dir}")
            return self.samples

        print(f"SpatialSense: loading from {ann_file}")
        with open(ann_file) as f:
            data = json.load(f)

        # Handle both top-level list and nested {annotations: [...]}
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("annotations", data.get("data", list(data.values())))
            if items and isinstance(items[0], list):
                items = [item for sub in items for item in sub]
        else:
            items = []

        for idx, e in enumerate(items):
            if not isinstance(e, dict):
                continue

            image_file = e.get("image", e.get("url", ""))
            relation = e.get("relation", e.get("predicate", "unknown"))
            subject = e.get("subject", {})
            obj = e.get("object", {})

            subject_name = subject.get("name", "") if isinstance(subject, dict) else str(subject)
            object_name = obj.get("name", "") if isinstance(obj, dict) else str(obj)
            subject_bbox = self._parse_bbox(subject)
            object_bbox = self._parse_bbox(obj)
            label = e.get("label", 1)  # 1=true, 0=false
            answer = self._normalize_relation(relation) if int(label) == 1 else "false"

            img_path = self._resolve_image(image_file)
            normalized_rel = self._normalize_relation(relation)
            occlusion = self._compute_occlusion(subject_bbox, object_bbox)
            size_ratio = self._compute_size_ratio(subject_bbox, object_bbox)

            sample = SpatialQASample(
                id=f"spatialsense_{idx:06d}",
                dataset="spatialsense",
                image_path=img_path,
                question=f"Is {subject_name} {relation} {object_name}?",
                answer=answer,
                relation_type=normalized_rel,
                subject=subject_name,
                object=object_name,
                subject_bbox=subject_bbox,
                object_bbox=object_bbox,
                choices=[normalized_rel, "false"],
                occlusion_level=occlusion,
                object_size_ratio=size_ratio,
                depth_ambiguity=normalized_rel in DEPTH_RELATIONS,
            )
            self.samples.append(sample)

        print(f"SpatialSense: loaded {len(self.samples)} samples")
        return self.samples

    def _resolve_image(self, image_file: str) -> str:
        if not image_file:
            return ""
        if image_file.startswith("http"):
            # URL — save path for later download
            return image_file
        if os.path.isabs(image_file) and os.path.exists(image_file):
            return image_file
        for base in [self.raw_dir, os.path.join(self.raw_dir, "images")]:
            p = os.path.join(base, image_file)
            if os.path.exists(p):
                return p
        return os.path.join(self.raw_dir, "images", image_file)

    def _parse_bbox(self, obj) -> List[float]:
        if not isinstance(obj, dict):
            return None
        for keys in [("x", "y", "w", "h"), ("x1", "y1", "x2", "y2"),
                     ("xmin", "ymin", "xmax", "ymax")]:
            if all(k in obj for k in keys):
                vals = [float(obj[k]) for k in keys]
                # Convert xywh to xyxy if needed
                if keys[2] == "w":
                    return [vals[0], vals[1], vals[0]+vals[2], vals[1]+vals[3]]
                return vals
        if "bbox" in obj:
            b = obj["bbox"]
            if len(b) == 4:
                return [float(v) for v in b]
        return None

    def _compute_occlusion(self, b1, b2):
        if b1 is None or b2 is None:
            return None
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        if x2 <= x1 or y2 <= y1:
            return "none"
        inter = (x2-x1) * (y2-y1)
        area = max(1e-6, (b1[2]-b1[0]) * (b1[3]-b1[1]))
        iou = inter / area
        return "none" if iou < 0.1 else ("partial" if iou < 0.3 else "heavy")

    def _compute_size_ratio(self, b1, b2):
        if b1 is None or b2 is None:
            return None
        a1 = max(1e-9, (b1[2]-b1[0]) * (b1[3]-b1[1]))
        a2 = max(1e-9, (b2[2]-b2[0]) * (b2[3]-b2[1]))
        return a1 / a2

    def _normalize_relation(self, rel: str) -> str:
        mapping = {
            "to the left of": "left", "to the right of": "right",
            "left of": "left", "right of": "right",
            "on top of": "above", "in front of": "in_front_of",
            "far from": "far", "next to": "beside",
        }
        r = rel.lower().strip()
        return mapping.get(r, r.replace(" ", "_"))
