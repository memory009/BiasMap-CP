"""Extract metadata factors for Mondrian cell construction.

Factors:
  - scene_type: indoor/outdoor/synthetic/tabletop
  - viewpoint: front/side/top/arbitrary
  - occlusion_level: none/partial/heavy
  - object_size_ratio: area(subject)/area(object)
  - depth_ambiguity: True for front/behind/near/far relations
"""
from typing import List, Optional, Tuple
import os

from ..datasets.base import SpatialQASample

DEPTH_RELATIONS = {"in_front_of", "behind", "near", "far"}
SYNTHETIC_DATASETS = {"clevr"}

DATASET_SCENE_TYPES = {
    "clevr": "synthetic",
    "gqa": "outdoor",
    "vsr": "outdoor",
    "whatsup": "outdoor",
    "nlvr2": "outdoor",
    "gsr_bench": "outdoor",
    "spatialsense": "outdoor",
}


def extract_metadata(sample: SpatialQASample,
                     use_clip: bool = False) -> SpatialQASample:
    """Fill in metadata fields for a sample in-place."""
    # scene_type
    if sample.scene_type is None:
        sample.scene_type = DATASET_SCENE_TYPES.get(sample.dataset, "unknown")

    # depth_ambiguity
    if sample.depth_ambiguity is None:
        sample.depth_ambiguity = sample.relation_type in DEPTH_RELATIONS

    # occlusion and size_ratio from bboxes
    if sample.subject_bbox and sample.object_bbox:
        if sample.occlusion_level is None:
            sample.occlusion_level = _compute_occlusion(
                sample.subject_bbox, sample.object_bbox
            )
        if sample.object_size_ratio is None:
            sample.object_size_ratio = _compute_size_ratio(
                sample.subject_bbox, sample.object_bbox
            )

    # viewpoint heuristic from image or dataset
    if sample.viewpoint is None:
        sample.viewpoint = _estimate_viewpoint(sample)

    return sample


def extract_metadata_batch(samples: List[SpatialQASample],
                            use_clip: bool = False) -> List[SpatialQASample]:
    """Extract metadata for a list of samples."""
    for s in samples:
        extract_metadata(s, use_clip=use_clip)
    return samples


def _compute_occlusion(b1: List[float], b2: List[float]) -> str:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return "none"
    inter = (x2 - x1) * (y2 - y1)
    area = max(1e-9, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    iou = inter / area
    if iou < 0.1:
        return "none"
    elif iou < 0.3:
        return "partial"
    else:
        return "heavy"


def _compute_size_ratio(b1: List[float], b2: List[float]) -> float:
    a1 = max(1e-9, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1e-9, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return a1 / a2


def _estimate_viewpoint(sample: SpatialQASample) -> str:
    """Heuristic viewpoint estimation without CLIP."""
    if sample.dataset == "clevr":
        return "top"  # CLEVR is top-down-ish
    if sample.relation_type in ("above", "below", "on_top_of", "under"):
        return "front"
    if sample.relation_type in ("in_front_of", "behind"):
        return "side"
    # If bboxes available, estimate from vertical distribution
    if sample.subject_bbox and sample.object_bbox:
        s_center_y = (sample.subject_bbox[1] + sample.subject_bbox[3]) / 2
        o_center_y = (sample.object_bbox[1] + sample.object_bbox[3]) / 2
        if abs(s_center_y - o_center_y) > 0.3:
            return "front"  # large vertical separation suggests front-facing
    return "arbitrary"
