#!/usr/bin/env python3
"""
Baseline diagnosis experiment for BiomedCLIP few-shot classification on data2.
Collects: accuracy, macro_F1, per-class metrics, confusion matrix, embedding vis.
"""
import sys
import os
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biomedclip_zeroshot_cell_classify import InstanceInfo, resolve_device, ensure_local_biomedclip_dir
from biomedclip_fewshot_support_experiment import encode_multiscale_feature, normalize_feature
from biomedclip_query_adaptive_classifier import compute_morphology_features

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
WEIGHTS_DIR = Path("/home/xut/csclip/labeling_tool/weights/biomedclip")

CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
PROMPT_TEMPLATE = "a photomicrograph of a {name} cell in BALF sample"

SUPPORT_PER_CLASS = 5
RANDOM_SEED = 42


def load_yolo_polygon_annotations(label_path: Path):
    """Parse YOLO polygon label file. Returns list of (class_id, points)."""
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


def build_dataset_items(split: str):
    """Build list of {image_path, filename, annotations} for a split."""
    img_dir = DATA_ROOT / "images" / split
    lbl_dir = DATA_ROOT / "labels_polygon" / split
    items = []
    for img_path in sorted(img_dir.glob("*.png")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        anns = load_yolo_polygon_annotations(lbl_path)
        if anns:
            items.append({
                "image_path": str(img_path),
                "filename": img_path.name,
                "annotations": anns,
            })
    return items


def select_supports(train_items, n_per_class):
    """Select n support samples per class from training set."""
    random.seed(RANDOM_SEED)
    per_class = defaultdict(list)
    for item in train_items:
        for ann in item["annotations"]:
            per_class[ann["class_id"]].append({
                "image_path": item["image_path"],
                "class_id": ann["class_id"],
                "points": ann["points"],
                "ann_type": "polygon",
            })

    supports = []
    for cid in sorted(CLASS_NAMES.keys()):
        candidates = per_class.get(cid, [])
        if len(candidates) > n_per_class:
            candidates = random.sample(candidates, n_per_class)
        supports.extend(candidates)
        print(f"  Class {cid} ({CLASS_NAMES[cid]}): {len(candidates)} supports selected")

    return supports


def run_experiment():
    print("=" * 60)
    print("BASELINE DIAGNOSIS: BiomedCLIP Few-Shot on data2")
    print("=" * 60)

    train_items = build_dataset_items("train")
    val_items = build_dataset_items("val")
    print(f"\nDataset: {len(train_items)} train images, {len(val_items)} val images")

    train_cell_counts = defaultdict(int)
    for item in train_items:
        for ann in item["annotations"]:
            train_cell_counts[ann["class_id"]] += 1
    val_cell_counts = defaultdict(int)
    for item in val_items:
        for ann in item["annotations"]:
            val_cell_counts[ann["class_id"]] += 1

    print("\nCell distribution:")
    for cid in sorted(CLASS_NAMES):
        print(f"  {CLASS_NAMES[cid]}: train={train_cell_counts[cid]}, val={val_cell_counts[cid]}")

    print(f"\nSelecting {SUPPORT_PER_CLASS} supports per class...")
    supports = select_supports(train_items, SUPPORT_PER_CLASS)

    print("\nLoading BiomedCLIP model...")
    from labeling_tool.fewshot_biomedclip import prepare_classifier, predict_annotations, evaluate_dataset

    classifier, _ = prepare_classifier(
        support_items=supports,
        class_names=CLASS_NAMES,
        temperature=1.0,
        device="auto",
        use_prompts=True,
        image_proto_weight=0.7,
        text_proto_weight=0.3,
    )
    print(f"Classifier ready on {classifier.device}")
    print(f"Support counts: { {CLASS_NAMES[cid]: classifier.support_counts[cid] for cid in classifier.class_ids} }")

    print("\nRunning evaluation on val set...")
    result = evaluate_dataset(
        classifier=classifier,
        dataset_items=val_items,
        temperature=1.0,
    )

    metrics = result["metrics"]
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"Total cells: {metrics['total']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Mean confidence: {metrics['mean_confidence']:.4f}")

    print("\nPer-class metrics:")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for name, cls_metrics in metrics["per_class"].items():
        print(f"{name:<15} {cls_metrics['precision']:>10.4f} {cls_metrics['recall']:>10.4f} "
              f"{cls_metrics['f1']:>10.4f} {cls_metrics['support']:>10}")

    # Build confusion matrix
    preds = result.get("sample_predictions", [])
    cm = defaultdict(lambda: defaultdict(int))
    for p in preds:
        gt_name = p["gt_class_name"]
        pred_name = p["pred_class_name"]
        cm[gt_name][pred_name] += 1

    class_list = sorted(set(
        list(cm.keys()) + [n for row in cm.values() for n in row.keys()]
    ))
    print("\nConfusion Matrix:")
    gt_pred_label = "GT\\Pred"
    header = f"{gt_pred_label:<15}" + "".join(f"{c:>12}" for c in class_list)
    print(header)
    for gt in class_list:
        row = f"{gt:<15}" + "".join(f"{cm[gt][pred]:>12}" for pred in class_list)
        print(row)

    # Text vs Image prototype similarity analysis
    print("\n--- Text vs Image Prototype Analysis ---")
    if classifier.text_prototypes:
        for cid in classifier.class_ids:
            name = CLASS_NAMES[cid]
            img_proto = classifier.prototypes[cid]
            txt_proto = classifier.text_prototypes[cid]
            sim = float(img_proto @ txt_proto)
            print(f"  {name}: img-text cosine similarity = {sim:.4f}")

        print("\n  Cross-class text similarities:")
        for cid_a in classifier.class_ids:
            for cid_b in classifier.class_ids:
                if cid_a >= cid_b:
                    continue
                sim = float(classifier.text_prototypes[cid_a] @ classifier.text_prototypes[cid_b])
                print(f"    {CLASS_NAMES[cid_a]} <-> {CLASS_NAMES[cid_b]}: {sim:.4f}")

    # Save full results
    output = {
        "metrics": metrics,
        "support_counts": result["support_counts"],
        "confusion_matrix": {gt: dict(row) for gt, row in cm.items()},
    }
    out_path = Path(__file__).parent / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    run_experiment()
