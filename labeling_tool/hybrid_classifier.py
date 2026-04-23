"""
Bridge module: exposes the hybrid-adaptive classifier (with pairwise size
refiner, support reliability, etc.) through the same prepare / predict /
evaluate API that the labeling-tool frontend already speaks.

This replaces the old simple-prototype path in fewshot_biomedclip.py with
the full pipeline from biomedclip_hybrid_adaptive_classifier.py.
"""
from __future__ import annotations

import math
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps
from skimage.draw import polygon as sk_polygon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from biomedclip_fewshot_support_experiment import (  # noqa: E402
    encode_multiscale_feature,
    normalize_feature,
)
from biomedclip_query_adaptive_classifier import (  # noqa: E402
    SupportRecord,
    QueryRecord,
    build_morphology_stats,
    build_text_score_lookup,
    normalize_morphology_feature,
    compute_morphology_features,
)
from biomedclip_query_adaptive_classifier import SupportCandidate  # noqa: E402
from biomedclip_hybrid_adaptive_classifier import (  # noqa: E402
    HybridConfig,
    SupportReliabilityConfig,
    build_support_reliability_priors,
    build_class_morph_prototypes,
    _compute_query_score_details,
    _current_prediction_from_scores,
    _prototypes_from_support_records,
    CLASS_NAMES as HAC_CLASS_NAMES,
    LOG_AREA_INDEX,
    MIN_SIZE_SIGMA,
    softmax_np,
)
from biomedclip_zeroshot_cell_classify import (  # noqa: E402
    InstanceInfo,
    ensure_local_biomedclip_dir,
    resolve_device,
)
from labeling_tool.paths import biomedclip_local_dir  # noqa: E402

LOCAL_BIOMEDCLIP_DIR = biomedclip_local_dir()


class CancelledError(Exception):
    """Raised when an operation is cancelled via a cancel token."""

DEFAULT_CONFIG = {
    "cell_margin_ratio": 0.15,
    "context_margin_ratio": 0.30,
    "background_value": 128,
    "cell_scale_weight": 0.90,
    "context_scale_weight": 0.10,
}

BEST_HYBRID_WEIGHTS = {
    "global_image_weight": 1.0,
    "global_text_weight": 0.0,
    "adaptive_image_weight": 1.0,
    "adaptive_morph_weight": 0.0,
    "support_image_affinity_weight": 1.0,
    "support_morph_affinity_weight": 0.0,
    "support_temperature": 0.05,
    "final_temperature": 0.03,
    "adaptive_scale_min": 0.0,
    "adaptive_scale_max": 0.25,
    "margin_low": 0.03,
    "margin_high": 0.12,
    "eosinophil_bias": 0.0,
}

DEFAULT_SIZE_REFINER = {
    "trigger_margin_max": 0.12,
    "min_separation_z": 0.3,
    "score_scale": 0.08,
    "max_adjust": 0.04,
}


@dataclass(frozen=True)
class PairwiseSizeRefinerConfig:
    enabled: bool = True
    trigger_margin_max: float = 0.12
    min_separation_z: float = 0.3
    score_scale: float = 0.08
    max_adjust: float = 0.04


def _maybe_apply_pairwise_size_refiner(
    current_scores: np.ndarray,
    score_details: Dict[str, Any],
    size_prototypes: Dict[int, Tuple[float, float]] | None,
    cfg: "PairwiseSizeRefinerConfig | None",
    final_temperature: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Conditionally adjust scores between top-1 and top-2 based on cell size."""
    diag: Dict[str, Any] = {"applied": False, "adjust": 0.0}
    if cfg is None or not cfg.enabled or size_prototypes is None:
        return current_scores.copy(), diag

    class_ids = score_details["class_ids"]
    probs = softmax_np(current_scores, temperature=final_temperature)
    order = np.argsort(-probs)
    top1_idx, top2_idx = int(order[0]), int(order[1])
    margin = float(probs[top1_idx] - probs[top2_idx])

    if margin > cfg.trigger_margin_max:
        return current_scores.copy(), diag

    top1_cid = class_ids[top1_idx]
    top2_cid = class_ids[top2_idx]
    query_log_area = score_details.get("query_log_area")
    if query_log_area is None:
        return current_scores.copy(), diag

    mean1, sigma1 = size_prototypes[top1_cid]
    mean2, sigma2 = size_prototypes[top2_cid]
    separation = abs(mean1 - mean2) / max((sigma1 + sigma2) / 2.0, 1e-6)
    if separation < cfg.min_separation_z:
        return current_scores.copy(), diag

    d1 = abs(query_log_area - mean1) / max(sigma1, 1e-6)
    d2 = abs(query_log_area - mean2) / max(sigma2, 1e-6)
    raw_adjust = cfg.score_scale * (d1 - d2)
    adjust = float(np.clip(raw_adjust, -cfg.max_adjust, cfg.max_adjust))

    refined = current_scores.copy()
    refined[top2_idx] += adjust
    refined[top1_idx] -= adjust
    diag = {
        "applied": True,
        "adjust": adjust,
        "top1_cid": top1_cid,
        "top2_cid": top2_cid,
        "separation_z": separation,
        "d1": d1,
        "d2": d2,
    }
    return refined, diag


_MODEL_CACHE: Dict[str, Any] | None = None


@dataclass(frozen=True)
class PreparedHybridClassifier:
    device: str
    model: Any
    preprocess: Any
    class_ids: List[int]
    class_names: Dict[int, str]
    support_counts: Dict[int, int]
    selected_supports: Dict[int, List[SupportRecord]]
    global_image_prototypes: Dict[int, np.ndarray]
    text_prototypes: Dict[int, np.ndarray]
    text_features: Any
    class_prompt_indices: Dict[int, List[int]]
    prompts: List[str]
    morph_mean: np.ndarray
    morph_std: np.ndarray
    class_supports_norm: Dict[int, List[Tuple[SupportRecord, np.ndarray]]]
    class_morph_prototypes: Dict[int, np.ndarray]
    size_prototypes: Dict[int, Tuple[float, float]] | None
    support_reliability_priors: Dict[Tuple[str, int], float] | None
    hybrid_cfg: HybridConfig
    size_refiner_cfg: PairwiseSizeRefinerConfig | None
    support_reliability_cfg: SupportReliabilityConfig | None


def _load_model_bundle(device_arg: str = "auto") -> Dict[str, Any]:
    global _MODEL_CACHE
    target_device = resolve_device(device_arg)
    if _MODEL_CACHE and _MODEL_CACHE["device"] == target_device:
        return _MODEL_CACHE

    from open_clip import create_model_from_pretrained
    local_dir = ensure_local_biomedclip_dir(LOCAL_BIOMEDCLIP_DIR)
    model_id = f"local-dir:{local_dir}"
    try:
        model, preprocess = create_model_from_pretrained(model_id, device=target_device)
        model = model.to(target_device)
        model.eval()
        _MODEL_CACHE = {"device": target_device, "model": model, "preprocess": preprocess}
    except Exception:
        if target_device == "cuda":
            model, preprocess = create_model_from_pretrained(model_id, device="cpu")
            model = model.to("cpu")
            model.eval()
            _MODEL_CACHE = {"device": "cpu", "model": model, "preprocess": preprocess}
        else:
            raise
    return _MODEL_CACHE


def _load_image_rgb(image_path: str, cache: Dict[str, np.ndarray]) -> np.ndarray:
    if image_path in cache:
        return cache[image_path]
    img = np.array(ImageOps.exif_transpose(Image.open(image_path)).convert("RGB"))
    cache[image_path] = img
    return img


def _annotation_to_instance(ann: Dict[str, Any], img_h: int, img_w: int, instance_id: int) -> InstanceInfo:
    ann_type = ann.get("ann_type", "polygon")
    class_id = int(ann.get("class_id", -1))
    points = [float(v) for v in ann.get("points", [])]

    if ann_type == "bbox":
        if len(points) < 4:
            raise ValueError("BBox needs at least 4 values.")
        cx, cy, bw, bh = points[:4]
        x1 = max(0, min(img_w - 1, int(round((cx - bw / 2.0) * img_w))))
        y1 = max(0, min(img_h - 1, int(round((cy - bh / 2.0) * img_h))))
        x2 = max(x1 + 1, min(img_w, int(round((cx + bw / 2.0) * img_w))))
        y2 = max(y1 + 1, min(img_h, int(round((cy + bh / 2.0) * img_h))))
        mask = np.zeros((img_h, img_w), dtype=bool)
        mask[y1:y2, x1:x2] = True
        return InstanceInfo(instance_id=instance_id, class_id=class_id, bbox=(x1, y1, x2, y2), mask=mask)

    if len(points) < 6:
        raise ValueError("Polygon needs at least 3 points.")
    xs = [points[i] * img_w for i in range(0, len(points), 2)]
    ys = [points[i] * img_h for i in range(1, len(points), 2)]
    rr, cc = sk_polygon(ys, xs, shape=(img_h, img_w))
    if len(rr) == 0:
        raise ValueError("Empty polygon mask.")
    mask = np.zeros((img_h, img_w), dtype=bool)
    mask[rr, cc] = True
    x1 = max(0, int(np.min(cc)))
    y1 = max(0, int(np.min(rr)))
    x2 = min(img_w, int(np.max(cc)) + 1)
    y2 = min(img_h, int(np.max(rr)) + 1)
    return InstanceInfo(instance_id=instance_id, class_id=class_id, bbox=(x1, y1, x2, y2), mask=mask)


def _encode_cell(
    bundle: Dict[str, Any],
    image: np.ndarray,
    instance: InstanceInfo,
    cfg: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (image_feature, morph_feature, roi_rgb) for one cell."""
    image_feature = encode_multiscale_feature(
        model=bundle["model"],
        preprocess=bundle["preprocess"],
        image=image,
        instance=instance,
        device=bundle["device"],
        cell_margin_ratio=cfg["cell_margin_ratio"],
        context_margin_ratio=cfg["context_margin_ratio"],
        background_value=int(cfg["background_value"]),
        cell_scale_weight=cfg["cell_scale_weight"],
        context_scale_weight=cfg["context_scale_weight"],
    )
    morph_feature = compute_morphology_features(
        image=image,
        instance=instance,
        context_margin_ratio=cfg["context_margin_ratio"],
    )
    x1, y1, x2, y2 = instance.bbox
    roi_rgb = image[y1:y2, x1:x2].copy()
    return image_feature, morph_feature, roi_rgb


def _build_text_protos(
    bundle,
    class_names,
    prompt_template=None,
    text_prompt_names: Dict[int, str] | None = None,
):
    """Build text prototypes from class names (or optional longer phrases per class)."""
    import torch
    from open_clip import get_tokenizer

    local_dir = ensure_local_biomedclip_dir(LOCAL_BIOMEDCLIP_DIR)
    model_id = f"local-dir:{local_dir}"
    tokenizer = get_tokenizer(model_id)

    template = prompt_template or "a photomicrograph of a {name} cell in BALF sample"
    name_src = text_prompt_names if text_prompt_names is not None else class_names
    prompts = []
    class_prompt_indices: Dict[int, List[int]] = {}
    for cid in sorted(class_names):
        class_prompt_indices[cid] = [len(prompts)]
        prompts.append(template.format(name=name_src[cid]))

    dev = bundle["device"]
    text_tokens = tokenizer(prompts).to(dev)
    with torch.no_grad():
        text_feat = bundle["model"].encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    text_features_np = text_feat.detach().cpu().numpy().astype(np.float32)
    text_prototypes: Dict[int, np.ndarray] = {}
    for cid in sorted(class_names):
        idxs = class_prompt_indices[cid]
        text_prototypes[cid] = normalize_feature(text_features_np[idxs[0]])

    return text_prototypes, text_features_np, class_prompt_indices, prompts


def prepare_classifier(
    support_items: Sequence[Dict[str, Any]],
    class_names: Dict[int, str],
    device: str = "auto",
    config: Dict[str, float] | None = None,
    enable_size_refiner: bool = True,
    size_refiner_config: Dict[str, float] | None = None,
    hybrid_weights: Dict[str, float] | None = None,
    prompt_template: str | None = None,
    text_prompt_names: Dict[int, str] | None = None,
) -> PreparedHybridClassifier:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    hw = {**BEST_HYBRID_WEIGHTS, **(hybrid_weights or {})}
    bundle = _load_model_bundle(device)
    image_cache: Dict[str, np.ndarray] = {}

    selected_supports: Dict[int, List[SupportRecord]] = defaultdict(list)

    for index, item in enumerate(support_items, start=1):
        image_path = item.get("image_path")
        if not image_path:
            raise ValueError("support item missing image_path.")
        image = _load_image_rgb(image_path, image_cache)
        instance = _annotation_to_instance(item, image.shape[0], image.shape[1], instance_id=index)
        image_feature, morph_feature, roi_rgb = _encode_cell(bundle, image, instance, cfg)

        cid = int(item["class_id"])
        candidate = SupportCandidate(
            split="support",
            image_name=Path(image_path).name,
            image_path=image_path,
            label_path="",
            instance_id=index,
            class_id=cid,
        )
        record = SupportRecord(
            candidate=candidate,
            image_feature=image_feature,
            morph_feature=morph_feature,
            bbox=instance.bbox,
            roi_rgb=roi_rgb,
        )
        selected_supports[cid].append(record)

    class_ids = sorted(class_names)
    missing = [class_names[cid] for cid in class_ids if not selected_supports.get(cid)]
    if missing:
        # 自动收窄：只对有 support 的类别分类，而非报错阻塞
        # 仍至少需要 2 个类别，否则无法多分类
        active_class_ids = [cid for cid in class_ids if selected_supports.get(cid)]
        if len(active_class_ids) < 2:
            raise ValueError(
                f"至少需要为 2 个类别提供 support。当前仅有: "
                f"{[class_names[c] for c in active_class_ids] or '无'}；"
                f"缺少: {', '.join(missing)}"
            )
        import warnings
        warnings.warn(
            f"以下类别没有 support，已从本次分类中排除: {', '.join(missing)}",
            stacklevel=2,
        )
        class_ids = active_class_ids
        class_names = {cid: class_names[cid] for cid in class_ids}

    support_counts = {cid: len(selected_supports[cid]) for cid in class_ids}

    global_image_prototypes = _prototypes_from_support_records(selected_supports)

    text_prototypes, text_features_np, class_prompt_indices, prompts = _build_text_protos(
        bundle, class_names, prompt_template=prompt_template, text_prompt_names=text_prompt_names
    )

    all_supports = [r for records in selected_supports.values() for r in records]
    morph_mean, morph_std = build_morphology_stats(all_supports)
    class_morph_prototypes = build_class_morph_prototypes(selected_supports)

    class_supports_norm: Dict[int, List[Tuple[SupportRecord, np.ndarray]]] = {}
    for cid, records in selected_supports.items():
        class_supports_norm[cid] = [
            (record, normalize_morphology_feature(record.morph_feature, morph_mean, morph_std))
            for record in records
        ]

    size_prototypes: Dict[int, Tuple[float, float]] | None = None
    if enable_size_refiner:
        size_prototypes = {}
        for cid, pairs in class_supports_norm.items():
            log_areas = np.array([r.morph_feature[LOG_AREA_INDEX] for r, _ in pairs], dtype=np.float64)
            mean_log = float(np.mean(log_areas))
            sigma = max(float(np.std(log_areas)), MIN_SIZE_SIGMA)
            size_prototypes[cid] = (mean_log, sigma)

    sr_cfg = SupportReliabilityConfig(enabled=False, log_prior_strength=0.35, min_prior=0.35)
    sr_priors, _ = build_support_reliability_priors(
        selected_supports=selected_supports,
        global_image_prototypes=global_image_prototypes,
        cfg=sr_cfg,
    )

    sr_config = DEFAULT_SIZE_REFINER.copy()
    if size_refiner_config:
        sr_config.update(size_refiner_config)
    size_refiner_cfg = PairwiseSizeRefinerConfig(
        enabled=enable_size_refiner,
        trigger_margin_max=sr_config["trigger_margin_max"],
        min_separation_z=sr_config["min_separation_z"],
        score_scale=sr_config["score_scale"],
        max_adjust=sr_config["max_adjust"],
    ) if enable_size_refiner else None

    hybrid_cfg = HybridConfig(
        global_image_weight=hw["global_image_weight"],
        global_text_weight=hw["global_text_weight"],
        adaptive_image_weight=hw["adaptive_image_weight"],
        adaptive_morph_weight=hw["adaptive_morph_weight"],
        support_image_affinity_weight=hw["support_image_affinity_weight"],
        support_morph_affinity_weight=hw["support_morph_affinity_weight"],
        support_temperature=hw["support_temperature"],
        final_temperature=hw["final_temperature"],
        adaptive_scale_min=hw["adaptive_scale_min"],
        adaptive_scale_max=hw["adaptive_scale_max"],
        margin_low=hw["margin_low"],
        margin_high=hw["margin_high"],
        eosinophil_bias=hw.get("eosinophil_bias", 0.0),
        eos_verifier_enabled=False,
        eos_verifier_margin_max=0.08,
        eos_verifier_lymph_delta=0.02,
        eos_verifier_macro_delta=0.02,
        eos_verifier_threshold=0.02,
        size_weight=0.0,
    )

    return PreparedHybridClassifier(
        device=bundle["device"],
        model=bundle["model"],
        preprocess=bundle["preprocess"],
        class_ids=class_ids,
        class_names=class_names,
        support_counts=support_counts,
        selected_supports=selected_supports,
        global_image_prototypes=global_image_prototypes,
        text_prototypes=text_prototypes,
        text_features=text_features_np,
        class_prompt_indices=class_prompt_indices,
        prompts=prompts,
        morph_mean=morph_mean,
        morph_std=morph_std,
        class_supports_norm=class_supports_norm,
        class_morph_prototypes=class_morph_prototypes,
        size_prototypes=size_prototypes,
        support_reliability_priors=sr_priors,
        hybrid_cfg=hybrid_cfg,
        size_refiner_cfg=size_refiner_cfg,
        support_reliability_cfg=sr_cfg,
    )


def _classify_one_cell(
    classifier: PreparedHybridClassifier,
    image: np.ndarray,
    ann: Dict[str, Any],
    instance_id: int,
    cfg: Dict[str, float],
) -> Dict[str, Any]:
    """Classify a single cell using the full hybrid-adaptive pipeline."""
    bundle = {"model": classifier.model, "preprocess": classifier.preprocess, "device": classifier.device}
    instance = _annotation_to_instance(ann, image.shape[0], image.shape[1], instance_id=instance_id)
    image_feature, morph_feature, roi_rgb = _encode_cell(bundle, image, instance, cfg)

    query = QueryRecord(
        split="predict",
        image_path="",
        image_name="",
        instance_id=instance_id,
        gt_class_id=int(ann.get("class_id", -1)),
        bbox=instance.bbox,
        image_feature=image_feature,
        morph_feature=morph_feature,
        roi_rgb=roi_rgb,
    )

    score_details = _compute_query_score_details(
        query=query,
        class_supports_norm=classifier.class_supports_norm,
        global_image_prototypes=classifier.global_image_prototypes,
        text_prototypes=classifier.text_prototypes,
        morph_mean=classifier.morph_mean,
        morph_std=classifier.morph_std,
        cfg=classifier.hybrid_cfg,
        size_prototypes=classifier.size_prototypes,
        support_reliability_priors=classifier.support_reliability_priors,
        support_reliability_cfg=classifier.support_reliability_cfg,
    )

    if "query_log_area" not in score_details or score_details.get("query_log_area") is None:
        if classifier.size_prototypes is not None:
            score_details["query_log_area"] = float(query.morph_feature[LOG_AREA_INDEX])

    raw_scores = np.asarray(score_details["raw_final_scores_np"], dtype=np.float32)
    class_ids = score_details["class_ids"]

    refined_scores, size_diag = _maybe_apply_pairwise_size_refiner(
        current_scores=raw_scores,
        score_details=score_details,
        size_prototypes=classifier.size_prototypes,
        cfg=classifier.size_refiner_cfg,
        final_temperature=classifier.hybrid_cfg.final_temperature,
    )

    probs = softmax_np(refined_scores, temperature=classifier.hybrid_cfg.final_temperature)
    order = np.argsort(-probs)
    top1_idx = int(order[0])
    top2_idx = int(order[1]) if len(order) > 1 else top1_idx
    pred_class_id = int(class_ids[top1_idx])
    top2_class_id = int(class_ids[top2_idx])
    confidence = float(probs[top1_idx])
    top2_confidence = float(probs[top2_idx]) if len(order) > 1 else 0.0
    margin = confidence - top2_confidence

    gt_class_id = int(ann.get("class_id", -1))

    score_items = [
        {
            "class_id": int(class_ids[i]),
            "class_name": classifier.class_names.get(int(class_ids[i]), str(class_ids[i])),
            "score": float(refined_scores[i]),
            "probability": float(probs[i]),
        }
        for i in range(len(class_ids))
    ]
    score_items.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "instance_id": instance_id,
        "annotation_uid": ann.get("annotation_uid"),
        "bbox": [int(v) for v in instance.bbox],
        "gt_class_id": gt_class_id,
        "gt_class_name": classifier.class_names.get(gt_class_id, str(gt_class_id)),
        "pred_class_id": pred_class_id,
        "pred_class_name": classifier.class_names.get(pred_class_id, str(pred_class_id)),
        "confidence": confidence,
        "top2_class_id": top2_class_id,
        "top2_class_name": classifier.class_names.get(top2_class_id, str(top2_class_id)),
        "top2_confidence": top2_confidence,
        "margin": margin,
        "scores": score_items,
        "correct": gt_class_id == pred_class_id,
        "size_refiner_applied": bool(size_diag.get("applied", False)),
        "size_refiner_adjust": float(size_diag.get("adjust", 0.0)),
    }


def _compute_metrics(
    gt_labels: Sequence[int],
    pred_labels: Sequence[int],
    confidences: Sequence[float],
    class_ids: Sequence[int],
) -> Dict[str, Any]:
    total = len(gt_labels)
    correct = sum(int(g == p) for g, p in zip(gt_labels, pred_labels))
    mean_conf = float(np.mean(confidences)) if confidences else 0.0

    per_class: Dict[int, Dict[str, float]] = {}
    f1_values: List[float] = []
    for cid in class_ids:
        tp = sum(int(g == cid and p == cid) for g, p in zip(gt_labels, pred_labels))
        pred_pos = sum(int(p == cid) for p in pred_labels)
        gt_pos = sum(int(g == cid) for g in gt_labels)
        precision = tp / pred_pos if pred_pos else 0.0
        recall = tp / gt_pos if gt_pos else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[cid] = {"precision": precision, "recall": recall, "f1": f1, "support": gt_pos, "predicted": pred_pos}
        f1_values.append(f1)

    return {
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "mean_confidence": mean_conf,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
    }


def predict_annotations(
    classifier: PreparedHybridClassifier,
    image_path: str,
    annotations: Sequence[Dict[str, Any]],
    config: Dict[str, float] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    skip_metrics: bool = False,
) -> Dict[str, Any]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    image_cache: Dict[str, np.ndarray] = {}
    image = _load_image_rgb(image_path, image_cache)

    predictions: List[Dict[str, Any]] = []
    gt_labels, pred_labels, confidences = [], [], []

    for index, ann in enumerate(annotations, start=1):
        if cancel_check and cancel_check():
            raise CancelledError("操作已取消")
        pred = _classify_one_cell(classifier, image, ann, instance_id=index, cfg=cfg)
        predictions.append(pred)
        gt_labels.append(pred["gt_class_id"])
        pred_labels.append(pred["pred_class_id"])
        confidences.append(pred["confidence"])

    result: Dict[str, Any] = {
        "predictions": predictions,
        "support_counts": {classifier.class_names[cid]: cnt for cid, cnt in classifier.support_counts.items()},
        "method": "hybrid_adaptive_with_pairwise_size_refiner",
    }
    if not skip_metrics:
        result["metrics"] = _compute_metrics(gt_labels, pred_labels, confidences, classifier.class_ids)
    return result


def evaluate_dataset(
    classifier: PreparedHybridClassifier,
    dataset_items: Sequence[Dict[str, Any]],
    config: Dict[str, float] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Dict[str, Any]:
    all_predictions: List[Dict[str, Any]] = []
    gt_labels, pred_labels, confidences = [], [], []
    image_count = 0

    for item in dataset_items:
        if cancel_check and cancel_check():
            raise CancelledError("操作已取消")
        annotations = item.get("annotations") or []
        if not annotations:
            continue
        result = predict_annotations(
            classifier=classifier,
            image_path=item["image_path"],
            annotations=annotations,
            config=config,
            cancel_check=cancel_check,
        )
        image_count += 1
        for pred in result["predictions"]:
            pred["filename"] = item["filename"]
            all_predictions.append(pred)
            gt_labels.append(pred["gt_class_id"])
            pred_labels.append(pred["pred_class_id"])
            confidences.append(pred["confidence"])

    metrics = _compute_metrics(gt_labels, pred_labels, confidences, classifier.class_ids)
    per_class_named = {
        classifier.class_names[cid]: values for cid, values in metrics["per_class"].items()
    }

    return {
        "image_count": image_count,
        "cell_count": len(all_predictions),
        "metrics": {
            "total": metrics["total"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "mean_confidence": metrics["mean_confidence"],
            "per_class": per_class_named,
        },
        "support_counts": {classifier.class_names[cid]: cnt for cid, cnt in classifier.support_counts.items()},
        "sample_predictions": all_predictions[:50],
        "method": "hybrid_adaptive_with_pairwise_size_refiner",
    }
