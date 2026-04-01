"""Unified data structures for BiasMap-CP."""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json


@dataclass
class SpatialQASample:
    id: str
    dataset: str
    image_path: str
    question: str
    answer: str
    relation_type: str
    subject: Optional[str] = None
    object: Optional[str] = None
    subject_bbox: Optional[List[float]] = None   # [x1,y1,x2,y2] normalized
    object_bbox: Optional[List[float]] = None
    choices: Optional[List[str]] = None          # answer choices if MCQ
    # Metadata factors for Mondrian cells
    scene_type: Optional[str] = None             # indoor/outdoor/synthetic/tabletop
    viewpoint: Optional[str] = None              # front/side/top/arbitrary
    occlusion_level: Optional[str] = None        # none/partial/heavy
    object_size_ratio: Optional[float] = None    # subject_area / object_area
    depth_ambiguity: Optional[bool] = None       # True if relation is depth-sensitive

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpatialQASample":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class ModelOutput:
    sample_id: str
    dataset: str
    split: str
    model: str
    logits: Dict[str, float]           # answer_text -> logit
    probabilities: Dict[str, float]    # answer_text -> softmax prob
    predicted_answer: str
    correct: bool
    nonconformity_score: float         # 1 - p(true_answer), for conformal prediction
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelOutput":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class BaseDataset:
    """Abstract base class for spatial QA datasets."""

    def __init__(self, raw_dir: str):
        self.raw_dir = raw_dir
        self.samples: List[SpatialQASample] = []

    def load(self) -> List[SpatialQASample]:
        raise NotImplementedError

    def save_processed(self, output_path: str):
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            for s in self.samples:
                f.write(s.to_json() + "\n")
        print(f"Saved {len(self.samples)} samples to {output_path}")

    @staticmethod
    def load_processed(path: str) -> List[SpatialQASample]:
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(SpatialQASample.from_dict(json.loads(line)))
        return samples
