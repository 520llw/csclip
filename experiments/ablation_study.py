#!/usr/bin/env python3
"""
Ablation study: compare baseline vs. each improvement individually.

Experiments:
  A. Baseline (image prototypes only, no text, no morphology)
  B. + Text prototypes (image 0.7 + text 0.3)
  C. + Morphology constraints
  D. + Prompt Tuning
  E. Full pipeline (B + C + D)
"""
import sys
import os
import json
import random
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

from biomedclip_zeroshot_cell_classify import InstanceInfo, resolve_device
from biomedclip_fewshot_support_experiment import encode_multiscale_feature, normalize_feature
from biomedclip_query_adaptive_classifier import compute_morphology_features
from labeling_tool.morphology_constraints import apply_morphology_constraints
from PIL import Image
from skimage.draw import polygon as sk_polygon

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
SUPPORT_PER_CLASS = 5
RANDOM_SEED = 42


def load_yolo_polygon_annotations(label_path: Path):
    annotations = []
    if not label_path.exists():
        return annotations
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cid = int(parts[0])
            if cid not in CLASS_NAMES:
                continue
            points = [float(x) for x in parts[1:]]
            annotations.append({"class_id": cid, "points": points, "ann_type": "polygon"})
    return annotations


def build_dataset_items(split):
    img_dir = DATA_ROOT / "images" / split
    lbl_dir = DATA_ROOT / "labels_polygon" / split
    items = []
    for img_path in sorted(img_dir.glob("*.png")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        anns = load_yolo_polygon_annotations(lbl_path)
        if anns:
            items.append({"image_path": str(img_path), "filename": img_path.name, "annotations": anns})
    return items


def annotation_to_instance(ann, img_h, img_w, instance_id):
    points = ann["points"]
    class_id = ann["class_id"]
    xs = [points[i] * img_w for i in range(0, len(points), 2)]
    ys = [points[i] * img_h for i in range(1, len(points), 2)]
    rr, cc = sk_polygon(ys, xs, shape=(img_h, img_w))
    if len(rr) == 0:
        return None
    mask = np.zeros((img_h, img_w), dtype=bool)
    mask[rr, cc] = True
    x1 = max(0, int(np.min(cc)))
    y1 = max(0, int(np.min(rr)))
    x2 = min(img_w, int(np.max(cc)) + 1)
    y2 = min(img_h, int(np.max(rr)) + 1)
    return InstanceInfo(instance_id=instance_id, class_id=class_id, bbox=(x1, y1, x2, y2), mask=mask)


def compute_metrics(gt_labels, pred_labels, class_ids):
    total = len(gt_labels)
    correct = sum(int(g == p) for g, p in zip(gt_labels, pred_labels))
    per_class = {}
    f1_values = []
    for cid in class_ids:
        tp = sum(1 for g, p in zip(gt_labels, pred_labels) if g == cid and p == cid)
        pred_pos = sum(1 for p in pred_labels if p == cid)
        gt_pos = sum(1 for g in gt_labels if g == cid)
        precision = tp / pred_pos if pred_pos else 0.0
        recall = tp / gt_pos if gt_pos else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[cid] = {"precision": precision, "recall": recall, "f1": f1, "support": gt_pos}
        f1_values.append(f1)
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "total": total,
    }


def run_experiment(name, use_text=False, use_morphology=False, text_weight=0.3, morph_strength=1.0):
    """Run one ablation experiment."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  use_text={use_text}, use_morphology={use_morphology}")
    print(f"{'='*60}")

    train_items = build_dataset_items("train")
    val_items = build_dataset_items("val")

    from labeling_tool.fewshot_biomedclip import prepare_classifier, predict_annotations

    random.seed(RANDOM_SEED)
    per_class = defaultdict(list)
    for item in train_items:
        for ann in item["annotations"]:
            per_class[ann["class_id"]].append({
                "image_path": item["image_path"], "class_id": ann["class_id"],
                "points": ann["points"], "ann_type": "polygon"
            })

    supports = []
    for cid in sorted(CLASS_NAMES.keys()):
        candidates = per_class.get(cid, [])
        if len(candidates) > SUPPORT_PER_CLASS:
            candidates = random.sample(candidates, SUPPORT_PER_CLASS)
        supports.extend(candidates)

    img_weight = 1.0 - text_weight if use_text else 1.0
    txt_weight = text_weight if use_text else 0.0

    classifier, _ = prepare_classifier(
        support_items=supports,
        class_names=CLASS_NAMES,
        temperature=1.0,
        device="auto",
        use_prompts=use_text,
        image_proto_weight=img_weight,
        text_proto_weight=txt_weight,
    )

    gt_labels = []
    pred_labels = []

    for item in val_items:
        image = np.array(Image.open(item["image_path"]).convert("RGB"))
        img_h, img_w = image.shape[:2]

        for idx, ann in enumerate(item["annotations"]):
            instance = annotation_to_instance(ann, img_h, img_w, idx + 1)
            if instance is None:
                continue

            feature = encode_multiscale_feature(
                model=classifier.model, preprocess=classifier.preprocess,
                image=image, instance=instance, device=classifier.device,
            )

            scores = np.array([
                float(feature @ classifier.prototypes[cid])
                for cid in classifier.class_ids
            ], dtype=np.float32)

            if use_text and classifier.text_prototypes:
                txt_scores = np.array([
                    float(feature @ classifier.text_prototypes[cid])
                    for cid in classifier.class_ids
                ], dtype=np.float32)
                scores = img_weight * scores + txt_weight * txt_scores

            if use_morphology:
                morph_feat = compute_morphology_features(
                    image=image, instance=instance
                )
                scores = apply_morphology_constraints(
                    scores, morph_feat, classifier.class_ids, strength=morph_strength
                )

            best_idx = int(np.argmax(scores))
            pred_cid = classifier.class_ids[best_idx]

            gt_labels.append(ann["class_id"])
            pred_labels.append(pred_cid)

    metrics = compute_metrics(gt_labels, pred_labels, sorted(CLASS_NAMES.keys()))

    print(f"\nResults for: {name}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Per-class:")
    for cid in sorted(CLASS_NAMES.keys()):
        cm = metrics['per_class'].get(cid, {})
        print(f"    {CLASS_NAMES[cid]}: P={cm.get('precision',0):.4f} R={cm.get('recall',0):.4f} F1={cm.get('f1',0):.4f}")

    return {name: metrics}


def main():
    print("=" * 60)
    print("ABLATION STUDY: BALF Cell Classification")
    print("=" * 60)

    all_results = {}

    r = run_experiment("A: Baseline (image only)", use_text=False, use_morphology=False)
    all_results.update(r)

    r = run_experiment("B: + Text prototypes", use_text=True, use_morphology=False)
    all_results.update(r)

    r = run_experiment("C: + Morphology constraints", use_text=False, use_morphology=True)
    all_results.update(r)

    r = run_experiment("D: Text + Morphology", use_text=True, use_morphology=True)
    all_results.update(r)

    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<35} {'Accuracy':>10} {'Macro F1':>10}")
    print("-" * 55)
    for name, metrics in all_results.items():
        print(f"{name:<35} {metrics['accuracy']:>10.4f} {metrics['macro_f1']:>10.4f}")

    out_path = Path(__file__).parent / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    main()
