"""
Morphology Constraints — Hard constraints based on cytological prior knowledge
for BALF cell classification.

These rules encode domain knowledge that the visual model may not capture:
- Eosinophils: bilobed nucleus + coarse granules → medium area, low circularity
- Neutrophils: multilobed nucleus → medium area, very low circularity
- Lymphocytes: high nuclear-cytoplasmic ratio → small area, high circularity
- Macrophages: large irregular cells → large area, low circularity

The constraints adjust classification scores as a post-hoc correction,
providing interpretable and deterministic improvements.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from biomedclip_query_adaptive_classifier import LOG_AREA_INDEX


@dataclass
class CellMorphProfile:
    """Expected morphological profile for a cell class."""
    class_id: int
    name: str
    log_area_range: Tuple[float, float]
    circularity_range: Tuple[float, float]
    aspect_ratio_range: Tuple[float, float]
    solidity_min: float = 0.0
    eccentricity_range: Tuple[float, float] = (0.0, 1.0)
    color_bias: Optional[Tuple[float, float, float]] = None
    penalty_weight: float = 0.05
    bonus_weight: float = 0.03


BALF_CELL_PROFILES = {
    3: CellMorphProfile(
        class_id=3,
        name="Eosinophil",
        log_area_range=(7.0, 11.0),
        circularity_range=(0.3, 0.90),
        aspect_ratio_range=(0.5, 2.0),
        solidity_min=0.6,
        eccentricity_range=(0.0, 0.8),
        color_bias=(0.55, 0.35, 0.45),
        penalty_weight=0.02,
        bonus_weight=0.01,
    ),
    4: CellMorphProfile(
        class_id=4,
        name="Neutrophil",
        log_area_range=(7.0, 11.0),
        circularity_range=(0.25, 0.85),
        aspect_ratio_range=(0.4, 2.5),
        solidity_min=0.5,
        eccentricity_range=(0.0, 0.85),
        penalty_weight=0.02,
        bonus_weight=0.01,
    ),
    5: CellMorphProfile(
        class_id=5,
        name="Lymphocyte",
        log_area_range=(5.0, 9.5),
        circularity_range=(0.55, 1.0),
        aspect_ratio_range=(0.6, 1.6),
        solidity_min=0.7,
        eccentricity_range=(0.0, 0.6),
        penalty_weight=0.02,
        bonus_weight=0.015,
    ),
    6: CellMorphProfile(
        class_id=6,
        name="Macrophage",
        log_area_range=(8.0, 14.0),
        circularity_range=(0.15, 0.90),
        aspect_ratio_range=(0.3, 3.0),
        solidity_min=0.4,
        eccentricity_range=(0.0, 0.95),
        penalty_weight=0.02,
        bonus_weight=0.01,
    ),
}


def _in_range(value: float, rng: Tuple[float, float], soft_margin: float = 0.1) -> float:
    """Return a score in [-1, 1] indicating how well value fits the range.
    1.0 = center, 0.0 = edge, negative = outside."""
    lo, hi = rng
    center = (lo + hi) / 2
    half_span = (hi - lo) / 2
    if half_span < 1e-8:
        return 0.0

    dist = abs(value - center) / half_span
    if dist <= 1.0:
        return 1.0 - dist
    excess = (dist - 1.0) * (half_span / soft_margin) if soft_margin > 0 else dist - 1.0
    return -min(excess, 1.0)


def compute_morphology_adjustments(
    morph_feature: np.ndarray,
    class_ids: List[int],
    profiles: Optional[Dict[int, CellMorphProfile]] = None,
) -> np.ndarray:
    """Compute per-class score adjustments based on morphology constraints.

    Args:
        morph_feature: 12-dim morphology feature vector from compute_morphology_features()
            [0] log_area, [1] log_perimeter, [2] circularity, [3] aspect_ratio,
            [4] solidity, [5] mean_R, [6] mean_G, [7] mean_B,
            [8] std_intensity, [9] eccentricity, [10] extent, [11] norm_diameter
        class_ids: list of class IDs to score
        profiles: optional custom profiles (defaults to BALF_CELL_PROFILES)

    Returns:
        adjustments: numpy array of score adjustments per class (same order as class_ids)
    """
    if profiles is None:
        profiles = BALF_CELL_PROFILES

    adjustments = np.zeros(len(class_ids), dtype=np.float32)

    log_area = morph_feature[0]
    circularity = morph_feature[2]
    aspect_ratio = morph_feature[3]
    solidity = morph_feature[4]
    eccentricity = morph_feature[9]

    for idx, cid in enumerate(class_ids):
        if cid not in profiles:
            continue
        profile = profiles[cid]

        scores = []

        area_fit = _in_range(log_area, profile.log_area_range, soft_margin=0.5)
        scores.append(area_fit * 2.0)

        circ_fit = _in_range(circularity, profile.circularity_range, soft_margin=0.15)
        scores.append(circ_fit * 1.5)

        ar_fit = _in_range(aspect_ratio, profile.aspect_ratio_range, soft_margin=0.3)
        scores.append(ar_fit)

        if solidity < profile.solidity_min:
            scores.append(-1.0)

        ecc_fit = _in_range(eccentricity, profile.eccentricity_range, soft_margin=0.2)
        scores.append(ecc_fit * 0.5)

        mean_score = np.mean(scores) if scores else 0.0

        if mean_score > 0:
            adjustments[idx] = float(mean_score * profile.bonus_weight)
        else:
            adjustments[idx] = float(mean_score * profile.penalty_weight)

    return adjustments


def apply_morphology_constraints(
    raw_scores: np.ndarray,
    morph_feature: np.ndarray,
    class_ids: List[int],
    strength: float = 1.0,
    profiles: Optional[Dict[int, CellMorphProfile]] = None,
) -> np.ndarray:
    """Apply morphology-based hard constraints to raw classification scores.

    Args:
        raw_scores: original classification scores (same order as class_ids)
        morph_feature: 12-dim morphology feature vector
        class_ids: list of class IDs
        strength: scaling factor for adjustments (0=disabled, 1=full)
        profiles: optional custom profiles

    Returns:
        adjusted_scores: corrected scores
    """
    adjustments = compute_morphology_adjustments(morph_feature, class_ids, profiles)
    return raw_scores + adjustments * strength


def detect_eosinophil_granularity(
    roi_rgb: np.ndarray,
    threshold: float = 0.15,
) -> float:
    """Detect eosinophilic granularity from a cell ROI crop.

    Eosinophils have characteristic red-orange cytoplasmic granules that
    create a distinctive texture pattern.

    Returns a granularity score in [0, 1], higher = more granular.
    """
    import cv2

    if roi_rgb is None or roi_rgb.size == 0:
        return 0.0

    if roi_rgb.ndim != 3:
        return 0.0

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    eosin_mask = ((h < 25) | (h > 160)) & (s > 50) & (v > 80)
    eosin_ratio = np.sum(eosin_mask) / max(eosin_mask.size, 1)

    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_var = float(np.var(laplacian)) / 10000.0

    score = min(1.0, eosin_ratio * 2.0 + texture_var * 0.5)
    return float(score)
