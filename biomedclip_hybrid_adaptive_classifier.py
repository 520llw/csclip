"""
biomedclip_hybrid_adaptive_classifier — Hybrid scoring pipeline that fuses
global image prototypes, text prototypes, per-support adaptive affinity, and
morphology features for robust few-shot cell classification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from biomedclip_fewshot_support_experiment import normalize_feature
from biomedclip_query_adaptive_classifier import (
    SupportRecord,
    QueryRecord,
    normalize_morphology_feature,
    LOG_AREA_INDEX,
)

CLASS_NAMES: List[str] = [
    "CCEC", "RBC", "SEC", "Eosinophil", "Neutrophil", "Lymphocyte", "Macrophage"
]
MIN_SIZE_SIGMA: float = 0.3


@dataclass
class HybridConfig:
    """Hyper-parameters for the hybrid scoring pipeline."""
    global_image_weight: float = 1.0
    global_text_weight: float = 0.0
    adaptive_image_weight: float = 1.0
    adaptive_morph_weight: float = 0.0
    support_image_affinity_weight: float = 1.0
    support_morph_affinity_weight: float = 0.0
    support_temperature: float = 0.05
    final_temperature: float = 0.03
    adaptive_scale_min: float = 0.0
    adaptive_scale_max: float = 0.25
    margin_low: float = 0.03
    margin_high: float = 0.12
    eosinophil_bias: float = 0.0
    eos_verifier_enabled: bool = False
    eos_verifier_margin_max: float = 0.08
    eos_verifier_lymph_delta: float = 0.02
    eos_verifier_macro_delta: float = 0.02
    eos_verifier_threshold: float = 0.02
    size_weight: float = 0.0


@dataclass
class SupportReliabilityConfig:
    """Control support reliability prior computation."""
    enabled: bool = False
    log_prior_strength: float = 0.35
    min_prior: float = 0.35


def softmax_np(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Temperature-scaled softmax in numpy."""
    t = max(temperature, 1e-8)
    shifted = (scores - scores.max()) / t
    exp_s = np.exp(shifted)
    return exp_s / np.clip(exp_s.sum(), 1e-12, None)


def _prototypes_from_support_records(
    selected_supports: Dict[int, List[SupportRecord]],
) -> Dict[int, np.ndarray]:
    """Compute per-class image prototypes (L2-normed mean of support features)."""
    prototypes: Dict[int, np.ndarray] = {}
    for cid, records in selected_supports.items():
        if not records:
            continue
        feats = np.stack([r.image_feature for r in records], axis=0)
        proto = feats.mean(axis=0).astype(np.float32)
        prototypes[cid] = normalize_feature(proto)
    return prototypes


def build_class_morph_prototypes(
    selected_supports: Dict[int, List[SupportRecord]],
) -> Dict[int, np.ndarray]:
    """Per-class mean morphology feature vector."""
    prototypes: Dict[int, np.ndarray] = {}
    for cid, records in selected_supports.items():
        if not records:
            continue
        feats = np.stack([r.morph_feature for r in records], axis=0)
        prototypes[cid] = feats.mean(axis=0).astype(np.float32)
    return prototypes


def build_support_reliability_priors(
    selected_supports: Dict[int, List[SupportRecord]],
    global_image_prototypes: Dict[int, np.ndarray],
    cfg: SupportReliabilityConfig,
) -> Tuple[Dict[Tuple[str, int], float], Dict[int, float]]:
    """Compute per-support reliability weights based on cosine similarity to
    their class prototype.  Returns (per-support priors dict, per-class mean)."""
    priors: Dict[Tuple[str, int], float] = {}
    class_means: Dict[int, float] = {}

    if not cfg.enabled:
        for cid, records in selected_supports.items():
            for r in records:
                key = (r.candidate.image_name, r.candidate.instance_id)
                priors[key] = 1.0
            class_means[cid] = 1.0
        return priors, class_means

    for cid, records in selected_supports.items():
        proto = global_image_prototypes.get(cid)
        if proto is None or not records:
            for r in records:
                key = (r.candidate.image_name, r.candidate.instance_id)
                priors[key] = 1.0
            class_means[cid] = 1.0
            continue

        sims = [float(r.image_feature @ proto) for r in records]
        mean_sim = float(np.mean(sims))
        class_means[cid] = mean_sim

        for r, sim in zip(records, sims):
            key = (r.candidate.image_name, r.candidate.instance_id)
            w = max(cfg.min_prior, sim * cfg.log_prior_strength + (1 - cfg.log_prior_strength))
            priors[key] = float(np.clip(w, cfg.min_prior, 1.0))

    return priors, class_means


def _compute_support_affinity_scores(
    query_image_feature: np.ndarray,
    query_morph_norm: np.ndarray,
    class_supports_norm: Dict[int, List[Tuple[SupportRecord, np.ndarray]]],
    cfg: HybridConfig,
    support_reliability_priors: Optional[Dict[Tuple[str, int], float]] = None,
    support_reliability_cfg: Optional[SupportReliabilityConfig] = None,
) -> Dict[int, float]:
    """Compute per-class affinity by aggregating similarity to each support."""
    class_scores: Dict[int, float] = {}

    for cid, pairs in class_supports_norm.items():
        if not pairs:
            class_scores[cid] = 0.0
            continue

        weighted_sims = []
        for record, morph_norm in pairs:
            img_sim = float(query_image_feature @ record.image_feature)
            morph_sim = float(query_morph_norm @ morph_norm) if cfg.support_morph_affinity_weight > 0 else 0.0

            combined = (
                cfg.support_image_affinity_weight * img_sim
                + cfg.support_morph_affinity_weight * morph_sim
            )

            reliability = 1.0
            if support_reliability_priors is not None:
                key = (record.candidate.image_name, record.candidate.instance_id)
                reliability = support_reliability_priors.get(key, 1.0)

            weighted_sims.append(combined * reliability)

        exp_sims = np.exp(np.array(weighted_sims, dtype=np.float32) / max(cfg.support_temperature, 1e-8))
        class_scores[cid] = float(np.sum(exp_sims * np.array(weighted_sims)) / np.clip(np.sum(exp_sims), 1e-12, None))

    return class_scores


def _compute_query_score_details(
    *,
    query: QueryRecord,
    class_supports_norm: Dict[int, List[Tuple[SupportRecord, np.ndarray]]],
    global_image_prototypes: Dict[int, np.ndarray],
    text_prototypes: Dict[int, np.ndarray],
    morph_mean: np.ndarray,
    morph_std: np.ndarray,
    cfg: HybridConfig,
    size_prototypes: Optional[Dict[int, Tuple[float, float]]] = None,
    support_reliability_priors: Optional[Dict[Tuple[str, int], float]] = None,
    support_reliability_cfg: Optional[SupportReliabilityConfig] = None,
) -> Dict[str, Any]:
    """Core scoring: combine global-prototype, text, and support-adaptive channels.

    Returns a dict with 'raw_final_scores_np', 'class_ids', 'query_log_area',
    and diagnostic details.
    """
    class_ids = sorted(global_image_prototypes.keys())
    n = len(class_ids)

    query_morph_norm = normalize_morphology_feature(query.morph_feature, morph_mean, morph_std)

    # --- Channel 1: Global image prototype similarity ---
    global_img_scores = np.array(
        [float(query.image_feature @ global_image_prototypes[cid]) for cid in class_ids],
        dtype=np.float32,
    )

    # --- Channel 2: Text prototype similarity ---
    global_text_scores = np.array(
        [float(query.image_feature @ text_prototypes[cid]) if cid in text_prototypes else 0.0
         for cid in class_ids],
        dtype=np.float32,
    )

    # --- Channel 3: Support-adaptive affinity ---
    affinity_lookup = _compute_support_affinity_scores(
        query.image_feature, query_morph_norm, class_supports_norm, cfg,
        support_reliability_priors, support_reliability_cfg,
    )
    adaptive_scores = np.array(
        [affinity_lookup.get(cid, 0.0) for cid in class_ids],
        dtype=np.float32,
    )

    # --- Confidence-aware adaptive scaling ---
    base_probs = softmax_np(global_img_scores, temperature=cfg.final_temperature)
    sorted_probs = np.sort(base_probs)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1]) if n > 1 else 1.0

    if margin < cfg.margin_low:
        adaptive_scale = cfg.adaptive_scale_max
    elif margin > cfg.margin_high:
        adaptive_scale = cfg.adaptive_scale_min
    else:
        t = (margin - cfg.margin_low) / max(cfg.margin_high - cfg.margin_low, 1e-8)
        adaptive_scale = cfg.adaptive_scale_max * (1 - t) + cfg.adaptive_scale_min * t

    # --- Fusion ---
    final_scores = (
        cfg.global_image_weight * global_img_scores
        + cfg.global_text_weight * global_text_scores
        + adaptive_scale * (
            cfg.adaptive_image_weight * adaptive_scores
        )
    )

    # Eosinophil bias
    if cfg.eosinophil_bias != 0.0:
        for i, cid in enumerate(class_ids):
            if cid == 3:
                final_scores[i] += cfg.eosinophil_bias

    query_log_area = float(query.morph_feature[LOG_AREA_INDEX])

    return {
        "raw_final_scores_np": final_scores,
        "class_ids": class_ids,
        "query_log_area": query_log_area,
        "global_img_scores": global_img_scores,
        "global_text_scores": global_text_scores,
        "adaptive_scores": adaptive_scores,
        "adaptive_scale": adaptive_scale,
        "margin": margin,
    }


def _current_prediction_from_scores(
    scores: np.ndarray,
    class_ids: List[int],
    temperature: float = 0.03,
) -> Tuple[int, float, float]:
    """Return (pred_class_id, confidence, margin) from raw scores."""
    probs = softmax_np(scores, temperature)
    order = np.argsort(-probs)
    top1 = int(order[0])
    top2 = int(order[1]) if len(order) > 1 else top1
    return class_ids[top1], float(probs[top1]), float(probs[top1] - probs[top2])


__all__ = [
    "HybridConfig",
    "SupportReliabilityConfig",
    "CLASS_NAMES",
    "LOG_AREA_INDEX",
    "MIN_SIZE_SIGMA",
    "build_support_reliability_priors",
    "build_class_morph_prototypes",
    "_compute_query_score_details",
    "_current_prediction_from_scores",
    "_prototypes_from_support_records",
    "softmax_np",
]
