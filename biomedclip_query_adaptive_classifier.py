"""
biomedclip_query_adaptive_classifier — Data structures and morphology feature
extraction for query-adaptive cell classification.

Provides SupportCandidate / SupportRecord / QueryRecord data classes and
functions for computing morphology-based features (area, perimeter, circularity,
colour statistics, …) from cell mask + image.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from biomedclip_zeroshot_cell_classify import InstanceInfo
from biomedclip_fewshot_support_experiment import normalize_feature

MORPH_FEATURE_DIM = 12
LOG_AREA_INDEX = 0


@dataclass
class SupportCandidate:
    """Metadata for a support sample before feature extraction."""
    split: str
    image_name: str
    image_path: str
    label_path: str
    instance_id: int
    class_id: int


@dataclass
class SupportRecord:
    """A support sample with pre-computed features."""
    candidate: SupportCandidate
    image_feature: np.ndarray
    morph_feature: np.ndarray
    bbox: Tuple[int, int, int, int]
    roi_rgb: np.ndarray


@dataclass
class QueryRecord:
    """A query (test) sample with pre-computed features."""
    split: str
    image_path: str
    image_name: str
    instance_id: int
    gt_class_id: int
    bbox: Tuple[int, int, int, int]
    image_feature: np.ndarray
    morph_feature: np.ndarray
    roi_rgb: np.ndarray


def compute_morphology_features(
    *,
    image: np.ndarray,
    instance: InstanceInfo,
    context_margin_ratio: float = 0.30,
) -> np.ndarray:
    """Extract a fixed-length morphology feature vector from a cell instance.

    Features (12-dim):
      [0]  log_area           — log(pixel area of mask)
      [1]  log_perimeter      — log(contour perimeter)
      [2]  circularity        — 4*pi*area / perimeter^2
      [3]  aspect_ratio       — bbox width / height
      [4]  solidity           — area / convex_hull_area
      [5]  mean_R             — mean red channel inside mask (0-1)
      [6]  mean_G             — mean green channel
      [7]  mean_B             — mean blue channel
      [8]  std_intensity      — std of grayscale inside mask
      [9]  eccentricity       — from fitted ellipse (0-1)
      [10] extent             — area / bbox_area
      [11] equiv_diameter     — sqrt(4*area/pi) normalised by bbox diagonal
    """
    mask = instance.mask.astype(np.uint8)
    area = int(np.sum(mask))
    if area < 1:
        return np.zeros(MORPH_FEATURE_DIM, dtype=np.float32)

    x1, y1, x2, y2 = instance.bbox
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    bbox_area = bw * bh

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(MORPH_FEATURE_DIM, dtype=np.float32)
    cnt = max(contours, key=cv2.contourArea)

    perimeter = cv2.arcLength(cnt, True)
    circularity = (4 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    aspect_ratio = bw / bh

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    roi = image[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2].astype(bool)

    if image.ndim == 3 and roi.shape[2] >= 3:
        pixels = roi[roi_mask].astype(np.float32) / 255.0
        if len(pixels) == 0:
            mean_r = mean_g = mean_b = 0.0
        else:
            mean_r = float(np.mean(pixels[:, 0]))
            mean_g = float(np.mean(pixels[:, 1]))
            mean_b = float(np.mean(pixels[:, 2]))
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        mean_r = mean_g = mean_b = 0.0
        gray = roi if roi.ndim == 2 else roi[:, :, 0]

    gray_pixels = gray[roi_mask].astype(np.float32)
    std_intensity = float(np.std(gray_pixels)) / 255.0 if len(gray_pixels) > 1 else 0.0

    if len(cnt) >= 5:
        try:
            ellipse = cv2.fitEllipse(cnt)
            (_, (ma, MA), _) = ellipse
            eccentricity = math.sqrt(1 - (min(ma, MA) / max(ma, MA)) ** 2) if max(ma, MA) > 0 else 0.0
        except cv2.error:
            eccentricity = 0.0
    else:
        eccentricity = 0.0

    extent = area / bbox_area if bbox_area > 0 else 0.0
    equiv_diameter = math.sqrt(4 * area / math.pi)
    bbox_diag = math.sqrt(bw ** 2 + bh ** 2)
    norm_diameter = equiv_diameter / bbox_diag if bbox_diag > 0 else 0.0

    feat = np.array([
        math.log(max(area, 1)),        # 0: log_area
        math.log(max(perimeter, 1)),    # 1: log_perimeter
        circularity,                     # 2
        aspect_ratio,                    # 3
        solidity,                        # 4
        mean_r,                          # 5
        mean_g,                          # 6
        mean_b,                          # 7
        std_intensity,                   # 8
        eccentricity,                    # 9
        extent,                          # 10
        norm_diameter,                   # 11
    ], dtype=np.float32)

    return feat


def build_morphology_stats(
    supports: Sequence[SupportRecord],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-dimension mean and std of morphology features across all supports."""
    if not supports:
        return np.zeros(MORPH_FEATURE_DIM, dtype=np.float32), np.ones(MORPH_FEATURE_DIM, dtype=np.float32)
    feats = np.stack([s.morph_feature for s in supports], axis=0)
    mean = feats.mean(axis=0).astype(np.float32)
    std = feats.std(axis=0).astype(np.float32)
    std = np.clip(std, 1e-6, None)
    return mean, std


def normalize_morphology_feature(
    feat: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Z-score normalise a morphology feature vector."""
    return ((feat - mean) / std).astype(np.float32)


def build_text_score_lookup(
    text_prototypes: Dict[int, np.ndarray],
    class_ids: Sequence[int],
    query_feature: np.ndarray,
) -> Dict[int, float]:
    """Compute text similarity scores for a query against each class."""
    lookup: Dict[int, float] = {}
    for cid in class_ids:
        if cid in text_prototypes:
            lookup[cid] = float(query_feature @ text_prototypes[cid])
        else:
            lookup[cid] = 0.0
    return lookup


def build_class_morph_prototypes(
    selected_supports: Dict[int, List[SupportRecord]],
) -> Dict[int, np.ndarray]:
    """Build per-class morphology prototypes (mean of support morph features)."""
    prototypes: Dict[int, np.ndarray] = {}
    for cid, records in selected_supports.items():
        if not records:
            continue
        feats = np.stack([r.morph_feature for r in records], axis=0)
        prototypes[cid] = feats.mean(axis=0).astype(np.float32)
    return prototypes


__all__ = [
    "SupportRecord",
    "QueryRecord",
    "SupportCandidate",
    "build_morphology_stats",
    "build_text_score_lookup",
    "normalize_morphology_feature",
    "compute_morphology_features",
    "build_class_morph_prototypes",
    "MORPH_FEATURE_DIM",
    "LOG_AREA_INDEX",
]
