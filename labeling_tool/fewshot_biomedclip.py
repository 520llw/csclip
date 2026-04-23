from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps
from skimage.draw import polygon as sk_polygon

from biomedclip_fewshot_support_experiment import encode_multiscale_feature, normalize_feature
from biomedclip_zeroshot_cell_classify import InstanceInfo, ensure_local_biomedclip_dir, resolve_device

from labeling_tool.paths import biomedclip_local_dir

ROOT = Path(__file__).resolve().parent.parent
LOCAL_BIOMEDCLIP_DIR = biomedclip_local_dir()

DEFAULT_PROMPT_TEMPLATE = "a photomicrograph of a {name} cell in tissue sample"

DEFAULT_CONFIG = {
    "cell_margin_ratio": 0.15,
    "context_margin_ratio": 0.30,
    "background_value": 128,
    "cell_scale_weight": 0.90,
    "context_scale_weight": 0.10,
}

_MODEL_CACHE: Dict[str, Any] | None = None


@dataclass(frozen=True)
class PreparedClassifier:
    device: str
    model: Any
    preprocess: Any
    class_ids: List[int]
    class_names: Dict[int, str]
    prototypes: Dict[int, np.ndarray]
    support_counts: Dict[int, int]
    text_prototypes: Optional[Dict[int, np.ndarray]] = None
    text_features: Optional[Any] = None
    class_prompt_indices: Optional[Dict[int, List[int]]] = None
    prompts: Optional[List[str]] = None
    image_proto_weight: float = 1.0
    text_proto_weight: float = 0.0


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


def _load_image_rgb(image_path: str, image_cache: Dict[str, np.ndarray]) -> np.ndarray:
    cached = image_cache.get(image_path)
    if cached is not None:
        return cached
    image = np.array(ImageOps.exif_transpose(Image.open(image_path)).convert("RGB"))
    image_cache[image_path] = image
    return image


def _annotation_to_instance(
    annotation: Dict[str, Any],
    img_h: int,
    img_w: int,
    instance_id: int,
) -> InstanceInfo:
    ann_type = annotation.get("ann_type", "polygon")
    class_id = int(annotation.get("class_id", -1))
    points = [float(v) for v in annotation.get("points", [])]

    if ann_type == "bbox":
        if len(points) < 4:
            raise ValueError("BBox 标注至少需要 4 个值。")
        cx, cy, bw, bh = points[:4]
        x1 = max(0, min(img_w - 1, int(round((cx - bw / 2.0) * img_w))))
        y1 = max(0, min(img_h - 1, int(round((cy - bh / 2.0) * img_h))))
        x2 = max(x1 + 1, min(img_w, int(round((cx + bw / 2.0) * img_w))))
        y2 = max(y1 + 1, min(img_h, int(round((cy + bh / 2.0) * img_h))))
        mask = np.zeros((img_h, img_w), dtype=bool)
        mask[y1:y2, x1:x2] = True
        return InstanceInfo(instance_id=instance_id, class_id=class_id, bbox=(x1, y1, x2, y2), mask=mask)

    if len(points) < 6:
        raise ValueError("Polygon 标注至少需要 3 个点。")

    xs = [points[i] * img_w for i in range(0, len(points), 2)]
    ys = [points[i] * img_h for i in range(1, len(points), 2)]
    rr, cc = sk_polygon(ys, xs, shape=(img_h, img_w))
    if len(rr) == 0:
        raise ValueError("Polygon 掩膜为空。")
    mask = np.zeros((img_h, img_w), dtype=bool)
    mask[rr, cc] = True
    x1 = max(0, int(np.min(cc)))
    y1 = max(0, int(np.min(rr)))
    x2 = min(img_w, int(np.max(cc)) + 1)
    y2 = min(img_h, int(np.max(rr)) + 1)
    return InstanceInfo(instance_id=instance_id, class_id=class_id, bbox=(x1, y1, x2, y2), mask=mask)


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    safe_temp = max(float(temperature), 1e-6)
    shifted = (scores - scores.max()) / safe_temp
    exp_scores = np.exp(shifted)
    return exp_scores / np.clip(exp_scores.sum(), 1e-8, None)


def _build_prompt_ensembles_from_class_names(
    class_names: Dict[int, str],
    template: str = DEFAULT_PROMPT_TEMPLATE,
) -> Dict[int, List[str]]:
    return {
        cid: [template.format(name=name)]
        for cid, name in class_names.items()
    }


def _build_prompt_lookup(
    prompt_ensembles: Dict[int, List[str]],
) -> Tuple[List[str], Dict[int, List[int]]]:
    prompts: List[str] = []
    class_prompt_indices: Dict[int, List[int]] = {}
    for class_id in sorted(prompt_ensembles):
        class_prompt_indices[class_id] = []
        for prompt in prompt_ensembles[class_id]:
            if not (prompt and prompt.strip()):
                continue
            class_prompt_indices[class_id].append(len(prompts))
            prompts.append(prompt.strip())
        if not class_prompt_indices[class_id]:
            class_prompt_indices[class_id] = [len(prompts)]
            prompts.append("a photomicrograph of a cell")
    return prompts, class_prompt_indices


def _build_text_prototypes(
    text_features: Any,
    class_prompt_indices: Dict[int, List[int]],
    primary_prompt_weight: float,
) -> Dict[int, np.ndarray]:
    if not 0.0 <= primary_prompt_weight <= 1.0:
        primary_prompt_weight = 0.75
    text_prototypes: Dict[int, np.ndarray] = {}
    feat = text_features.cpu().numpy() if hasattr(text_features, "cpu") else np.asarray(text_features)
    for class_id in sorted(class_prompt_indices):
        idxs = class_prompt_indices[class_id]
        vecs = feat[idxs]
        if len(idxs) == 1:
            proto = vecs[0]
        else:
            rem = 1.0 - primary_prompt_weight
            aux = rem / (len(idxs) - 1)
            w = np.array([primary_prompt_weight] + [aux] * (len(idxs) - 1), dtype=np.float32)
            proto = (vecs * w[:, None]).sum(axis=0)
        text_prototypes[class_id] = normalize_feature(proto.astype(np.float32))
    return text_prototypes


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
    for class_id in class_ids:
        tp = sum(int(g == class_id and p == class_id) for g, p in zip(gt_labels, pred_labels))
        pred_pos = sum(int(p == class_id) for p in pred_labels)
        gt_pos = sum(int(g == class_id) for g in gt_labels)
        precision = tp / pred_pos if pred_pos else 0.0
        recall = tp / gt_pos if gt_pos else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[class_id] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": gt_pos,
            "predicted": pred_pos,
        }
        f1_values.append(f1)

    return {
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "mean_confidence": mean_conf,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
    }


def prepare_classifier(
    support_items: Sequence[Dict[str, Any]],
    class_names: Dict[int, str],
    temperature: float = 1.0,
    device: str = "auto",
    config: Dict[str, float] | None = None,
    use_prompts: bool = False,
    prompt_ensembles: Optional[Dict[int, List[str]]] = None,
    prompt_mode: str = "auto",
    image_proto_weight: float = 0.5,
    text_proto_weight: float = 0.5,
    primary_prompt_weight: float = 0.75,
) -> Tuple[PreparedClassifier, float]:
    del temperature
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    bundle = _load_model_bundle(device)
    image_cache: Dict[str, np.ndarray] = {}
    support_features: Dict[int, List[np.ndarray]] = defaultdict(list)

    for index, item in enumerate(support_items, start=1):
        image_path = item.get("image_path")
        if not image_path:
            raise ValueError("support item 缺少 image_path。")
        image = _load_image_rgb(image_path, image_cache)
        instance = _annotation_to_instance(item, image.shape[0], image.shape[1], instance_id=index)
        feature = encode_multiscale_feature(
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
        support_features[int(item["class_id"])].append(feature)

    class_ids = sorted(class_names)
    missing = [class_names[cid] for cid in class_ids if not support_features.get(cid)]
    if missing:
        raise ValueError(f"以下类别还没有 support: {', '.join(missing)}")

    prototypes: Dict[int, np.ndarray] = {}
    support_counts: Dict[int, int] = {}
    for class_id in class_ids:
        feats = np.stack(support_features[class_id], axis=0)
        prototypes[class_id] = normalize_feature(np.mean(feats, axis=0).astype(np.float32))
        support_counts[class_id] = int(feats.shape[0])

    text_prototypes: Optional[Dict[int, np.ndarray]] = None
    text_features: Any = None
    class_prompt_indices: Optional[Dict[int, List[int]]] = None
    prompts: Optional[List[str]] = None

    if use_prompts and bundle["model"] is not None:
        if prompt_mode == "auto" or not prompt_ensembles:
            prompt_ensembles = _build_prompt_ensembles_from_class_names(class_names)
        elif prompt_mode == "custom" and prompt_ensembles:
            prompt_ensembles = {
                int(k): [p for p in (v if isinstance(v, list) else [v]) if p]
                for k, v in prompt_ensembles.items()
            }
            if not prompt_ensembles:
                prompt_ensembles = _build_prompt_ensembles_from_class_names(class_names)
        else:
            prompt_ensembles = _build_prompt_ensembles_from_class_names(class_names)

        import torch
        from open_clip import get_tokenizer
        local_dir = ensure_local_biomedclip_dir(LOCAL_BIOMEDCLIP_DIR)
        model_id = f"local-dir:{local_dir}"
        tokenizer = get_tokenizer(model_id)
        prompts, class_prompt_indices = _build_prompt_lookup(prompt_ensembles)
        dev = bundle["device"]
        text_tokens = tokenizer(prompts).to(dev)
        with torch.no_grad():
            text_feat = bundle["model"].encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        text_features = text_feat
        text_prototypes = _build_text_prototypes(
            text_features, class_prompt_indices, primary_prompt_weight
        )

    classifier = PreparedClassifier(
        device=bundle["device"],
        model=bundle["model"],
        preprocess=bundle["preprocess"],
        class_ids=class_ids,
        class_names=class_names,
        prototypes=prototypes,
        support_counts=support_counts,
        text_prototypes=text_prototypes,
        text_features=text_features,
        class_prompt_indices=class_prompt_indices,
        prompts=prompts,
        image_proto_weight=image_proto_weight if use_prompts else 1.0,
        text_proto_weight=text_proto_weight if use_prompts else 0.0,
    )
    return classifier, float(cfg["context_margin_ratio"])


def predict_annotations(
    classifier: PreparedClassifier,
    image_path: str,
    annotations: Sequence[Dict[str, Any]],
    temperature: float = 1.0,
    config: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    image_cache: Dict[str, np.ndarray] = {}
    image = _load_image_rgb(image_path, image_cache)

    predictions: List[Dict[str, Any]] = []
    gt_labels: List[int] = []
    pred_labels: List[int] = []
    confidences: List[float] = []

    for index, ann in enumerate(annotations, start=1):
        instance = _annotation_to_instance(ann, image.shape[0], image.shape[1], instance_id=index)
        feature = encode_multiscale_feature(
            model=classifier.model,
            preprocess=classifier.preprocess,
            image=image,
            instance=instance,
            device=classifier.device,
            cell_margin_ratio=cfg["cell_margin_ratio"],
            context_margin_ratio=cfg["context_margin_ratio"],
            background_value=int(cfg["background_value"]),
            cell_scale_weight=cfg["cell_scale_weight"],
            context_scale_weight=cfg["context_scale_weight"],
        )

        if classifier.text_prototypes is not None and classifier.text_prototypes:
            img_scores = np.array(
                [float(feature @ classifier.prototypes[class_id]) for class_id in classifier.class_ids],
                dtype=np.float32,
            )
            txt_scores = np.array(
                [float(feature @ classifier.text_prototypes[class_id]) for class_id in classifier.class_ids],
                dtype=np.float32,
            )
            scores = (
                classifier.image_proto_weight * img_scores
                + classifier.text_proto_weight * txt_scores
            )
        else:
            scores = np.array(
                [float(feature @ classifier.prototypes[class_id]) for class_id in classifier.class_ids],
                dtype=np.float32,
            )
        probs = _softmax(scores, temperature=temperature)
        best_idx = int(np.argmax(probs))
        pred_class_id = classifier.class_ids[best_idx]
        gt_class_id = int(ann.get("class_id", -1))
        confidence = float(probs[best_idx])

        gt_labels.append(gt_class_id)
        pred_labels.append(pred_class_id)
        confidences.append(confidence)

        score_items = [
            {
                "class_id": class_id,
                "class_name": classifier.class_names[class_id],
                "score": float(scores[idx]),
                "probability": float(probs[idx]),
            }
            for idx, class_id in enumerate(classifier.class_ids)
        ]
        score_items.sort(key=lambda item: item["probability"], reverse=True)

        predictions.append(
            {
                "instance_id": index,
                "annotation_uid": ann.get("annotation_uid"),
                "bbox": [int(v) for v in instance.bbox],
                "gt_class_id": gt_class_id,
                "gt_class_name": classifier.class_names.get(gt_class_id, str(gt_class_id)),
                "pred_class_id": pred_class_id,
                "pred_class_name": classifier.class_names[pred_class_id],
                "confidence": confidence,
                "scores": score_items,
                "correct": gt_class_id == pred_class_id,
            }
        )

    return {
        "predictions": predictions,
        "metrics": _compute_metrics(gt_labels, pred_labels, confidences, classifier.class_ids),
        "support_counts": {classifier.class_names[cid]: count for cid, count in classifier.support_counts.items()},
    }


def evaluate_dataset(
    classifier: PreparedClassifier,
    dataset_items: Sequence[Dict[str, Any]],
    temperature: float = 1.0,
    config: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    all_predictions: List[Dict[str, Any]] = []
    gt_labels: List[int] = []
    pred_labels: List[int] = []
    confidences: List[float] = []

    image_count = 0
    for item in dataset_items:
        annotations = item.get("annotations") or []
        if not annotations:
            continue
        result = predict_annotations(
            classifier=classifier,
            image_path=item["image_path"],
            annotations=annotations,
            temperature=temperature,
            config=config,
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
        "support_counts": {classifier.class_names[cid]: count for cid, count in classifier.support_counts.items()},
        "sample_predictions": all_predictions[:50],
    }
