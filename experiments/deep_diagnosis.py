#!/usr/bin/env python3
"""
Deep diagnosis: analyze per-cell embedding space, misclassification patterns,
and find optimal parameters through grid search.
"""
import sys
import os
import json
import random
import logging
from pathlib import Path
from collections import defaultdict
from itertools import product

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biomedclip_zeroshot_cell_classify import InstanceInfo, resolve_device
from biomedclip_fewshot_support_experiment import encode_multiscale_feature, normalize_feature
from biomedclip_query_adaptive_classifier import compute_morphology_features
from labeling_tool.morphology_constraints import apply_morphology_constraints
from PIL import Image
from skimage.draw import polygon as sk_polygon

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
RANDOM_SEED = 42


def load_yolo_polygon_annotations(label_path):
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
    xs = [points[i] * img_w for i in range(0, len(points), 2)]
    ys = [points[i] * img_h for i in range(1, len(points), 2)]
    rr, cc = sk_polygon(ys, xs, shape=(img_h, img_w))
    if len(rr) == 0:
        return None
    mask = np.zeros((img_h, img_w), dtype=bool)
    mask[rr, cc] = True
    x1, y1 = max(0, int(np.min(cc))), max(0, int(np.min(rr)))
    x2, y2 = min(img_w, int(np.max(cc)) + 1), min(img_h, int(np.max(rr)) + 1)
    return InstanceInfo(instance_id=instance_id, class_id=ann["class_id"], bbox=(x1, y1, x2, y2), mask=mask)


def compute_metrics(gt_labels, pred_labels, class_ids):
    total = len(gt_labels)
    correct = sum(int(g == p) for g, p in zip(gt_labels, pred_labels))
    f1_values = []
    per_class = {}
    for cid in class_ids:
        tp = sum(1 for g, p in zip(gt_labels, pred_labels) if g == cid and p == cid)
        pp = sum(1 for p in pred_labels if p == cid)
        gp = sum(1 for g in gt_labels if g == cid)
        prec = tp / pp if pp else 0.0
        rec = tp / gp if gp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[cid] = {"precision": prec, "recall": rec, "f1": f1, "support": gp}
        f1_values.append(f1)
    return {"accuracy": correct / total if total else 0.0, "macro_f1": np.mean(f1_values), "per_class": per_class, "total": total}


def extract_all_features(items, model, preprocess, device,
                         cell_margin=0.15, context_margin=0.30, bg_value=128,
                         cell_weight=0.90, context_weight=0.10):
    """Extract features + morphology for all cells in dataset."""
    records = []
    for item in items:
        image = np.array(Image.open(item["image_path"]).convert("RGB"))
        h, w = image.shape[:2]
        for idx, ann in enumerate(item["annotations"]):
            inst = annotation_to_instance(ann, h, w, idx + 1)
            if inst is None:
                continue
            feat = encode_multiscale_feature(
                model=model, preprocess=preprocess, image=image, instance=inst,
                device=device, cell_margin_ratio=cell_margin, context_margin_ratio=context_margin,
                background_value=bg_value, cell_scale_weight=cell_weight, context_scale_weight=context_weight,
            )
            morph = compute_morphology_features(image=image, instance=inst)
            records.append({
                "gt_class": ann["class_id"],
                "feature": feat,
                "morph": morph,
                "image_path": item["image_path"],
                "instance_id": idx + 1,
            })
    return records


def build_prototypes(records, class_ids, n_support):
    """Build per-class prototypes from training records."""
    random.seed(RANDOM_SEED)
    per_class = defaultdict(list)
    for r in records:
        per_class[r["gt_class"]].append(r)
    
    prototypes = {}
    selected = {}
    for cid in class_ids:
        cands = per_class.get(cid, [])
        if len(cands) > n_support:
            chosen = random.sample(cands, n_support)
        else:
            chosen = cands
        if not chosen:
            continue
        feats = np.stack([c["feature"] for c in chosen])
        proto = feats.mean(axis=0)
        prototypes[cid] = proto / np.linalg.norm(proto)
        selected[cid] = chosen
    return prototypes, selected


def classify_with_params(val_records, prototypes, class_ids,
                         use_morph=False, morph_strength=1.0,
                         temperature=0.03):
    """Classify all val records and return metrics."""
    gt_labels, pred_labels = [], []
    for r in val_records:
        scores = np.array([float(r["feature"] @ prototypes[cid]) for cid in class_ids])
        if use_morph:
            scores = apply_morphology_constraints(scores, r["morph"], class_ids, strength=morph_strength)
        shifted = (scores - scores.max()) / max(temperature, 1e-8)
        probs = np.exp(shifted) / np.exp(shifted).sum()
        pred = class_ids[int(np.argmax(probs))]
        gt_labels.append(r["gt_class"])
        pred_labels.append(pred)
    return compute_metrics(gt_labels, pred_labels, class_ids)


def misclassification_analysis(val_records, prototypes, class_ids):
    """Detailed analysis of misclassified cells."""
    confusion = defaultdict(lambda: defaultdict(int))
    mis_details = defaultdict(list)
    
    for r in val_records:
        scores = np.array([float(r["feature"] @ prototypes[cid]) for cid in class_ids])
        pred_idx = int(np.argmax(scores))
        pred = class_ids[pred_idx]
        gt = r["gt_class"]
        confusion[gt][pred] += 1
        
        if gt != pred:
            margin = float(scores[pred_idx] - scores[class_ids.index(gt)])
            mis_details[gt].append({
                "pred": pred,
                "margin": margin,
                "gt_score": float(scores[class_ids.index(gt)]),
                "pred_score": float(scores[pred_idx]),
                "log_area": float(r["morph"][0]),
                "circularity": float(r["morph"][2]),
            })
    return confusion, mis_details


def grid_search_params(train_records, val_records, class_ids):
    """Grid search over key parameters."""
    print("\n" + "=" * 60)
    print("GRID SEARCH: Encoding & Classification Parameters")
    print("=" * 60)
    
    n_support_options = [3, 5, 10, 15, 20, 30]
    morph_strength_options = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]
    
    best_f1 = 0
    best_config = None
    results = []
    
    for n_sup in n_support_options:
        protos, _ = build_prototypes(train_records, class_ids, n_sup)
        if len(protos) < len(class_ids):
            continue
        
        for ms in morph_strength_options:
            use_m = ms > 0
            metrics = classify_with_params(val_records, protos, class_ids, use_morph=use_m, morph_strength=ms)
            
            result = {
                "n_support": n_sup,
                "morph_strength": ms,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "eos_f1": metrics["per_class"].get(3, {}).get("f1", 0),
                "neu_f1": metrics["per_class"].get(4, {}).get("f1", 0),
                "lym_f1": metrics["per_class"].get(5, {}).get("f1", 0),
                "mac_f1": metrics["per_class"].get(6, {}).get("f1", 0),
            }
            results.append(result)
            
            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                best_config = result
    
    results.sort(key=lambda x: -x["macro_f1"])
    print(f"\n{'n_sup':>6} {'morph':>6} {'Acc':>8} {'mF1':>8} {'Eos':>6} {'Neu':>6} {'Lym':>6} {'Mac':>6}")
    print("-" * 55)
    for r in results[:15]:
        print(f"{r['n_support']:>6} {r['morph_strength']:>6.3f} {r['accuracy']:>8.4f} {r['macro_f1']:>8.4f} "
              f"{r['eos_f1']:>6.3f} {r['neu_f1']:>6.3f} {r['lym_f1']:>6.3f} {r['mac_f1']:>6.3f}")
    
    print(f"\nBEST: n_support={best_config['n_support']}, morph_strength={best_config['morph_strength']}")
    print(f"  Acc={best_config['accuracy']:.4f}, Macro F1={best_config['macro_f1']:.4f}")
    
    return best_config, results


def search_encoding_params(train_items, val_items, model, preprocess, device, class_ids):
    """Search over multi-scale encoding weight parameters."""
    print("\n" + "=" * 60)
    print("GRID SEARCH: Multi-Scale Encoding Weights")
    print("=" * 60)
    
    cell_weights = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
    margins = [0.10, 0.15, 0.20, 0.25, 0.30]
    bg_values = [0, 128, 200]
    
    best_f1 = 0
    best_cfg = None
    
    for cw in cell_weights:
        ctx_w = 1.0 - cw
        for cm in margins:
            for bg in bg_values:
                train_recs = extract_all_features(train_items, model, preprocess, device,
                                                  cell_margin=cm, context_margin=cm * 2,
                                                  bg_value=bg, cell_weight=cw, context_weight=ctx_w)
                val_recs = extract_all_features(val_items, model, preprocess, device,
                                                cell_margin=cm, context_margin=cm * 2,
                                                bg_value=bg, cell_weight=cw, context_weight=ctx_w)
                
                protos, _ = build_prototypes(train_recs, class_ids, 10)
                if len(protos) < len(class_ids):
                    continue
                
                metrics = classify_with_params(val_recs, protos, class_ids)
                
                if metrics["macro_f1"] > best_f1:
                    best_f1 = metrics["macro_f1"]
                    best_cfg = {"cell_weight": cw, "cell_margin": cm, "bg_value": bg,
                                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}
                    print(f"  NEW BEST: cw={cw:.2f} cm={cm:.2f} bg={bg} -> Acc={metrics['accuracy']:.4f} mF1={metrics['macro_f1']:.4f}")
    
    print(f"\nBEST ENCODING: {best_cfg}")
    return best_cfg


def main():
    print("=" * 60)
    print("DEEP DIAGNOSIS + PARAMETER OPTIMIZATION")
    print("=" * 60)
    
    train_items = build_dataset_items("train")
    val_items = build_dataset_items("val")
    class_ids = sorted(CLASS_NAMES.keys())
    
    print("\nLoading BiomedCLIP...")
    from labeling_tool.fewshot_biomedclip import _load_model_bundle
    bundle = _load_model_bundle("auto")
    model, preprocess, device = bundle["model"], bundle["preprocess"], bundle["device"]
    
    print("Extracting features (train)...")
    train_records = extract_all_features(train_items, model, preprocess, device)
    print(f"  Train: {len(train_records)} cells")
    
    print("Extracting features (val)...")
    val_records = extract_all_features(val_items, model, preprocess, device)
    print(f"  Val: {len(val_records)} cells")
    
    # 1. Misclassification analysis with baseline
    protos_5, _ = build_prototypes(train_records, class_ids, 5)
    confusion, mis_details = misclassification_analysis(val_records, protos_5, class_ids)
    
    print("\n--- CONFUSION MATRIX (5-shot baseline) ---")
    gt_pred = "GT\\Pred"
    header = f"{gt_pred:<15}" + "".join(f"{CLASS_NAMES[c]:>12}" for c in class_ids)
    print(header)
    for gt in class_ids:
        row = f"{CLASS_NAMES[gt]:<15}" + "".join(f"{confusion[gt][pred]:>12}" for pred in class_ids)
        print(row)
    
    print("\n--- MISCLASSIFICATION PATTERNS ---")
    for gt in class_ids:
        if not mis_details[gt]:
            continue
        print(f"\n{CLASS_NAMES[gt]} misclassified as:")
        pred_counts = defaultdict(int)
        for m in mis_details[gt]:
            pred_counts[m["pred"]] += 1
        for pred_cid, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
            subset = [m for m in mis_details[gt] if m["pred"] == pred_cid]
            avg_margin = np.mean([m["margin"] for m in subset])
            avg_area = np.mean([m["log_area"] for m in subset])
            avg_circ = np.mean([m["circularity"] for m in subset])
            print(f"  -> {CLASS_NAMES[pred_cid]}: {count} cells, avg_margin={avg_margin:.4f}, "
                  f"avg_log_area={avg_area:.1f}, avg_circularity={avg_circ:.3f}")
    
    # 2. Inter-class similarity analysis
    print("\n--- INTER-CLASS PROTOTYPE SIMILARITIES ---")
    for a in class_ids:
        for b in class_ids:
            if a >= b:
                continue
            sim = float(protos_5[a] @ protos_5[b])
            print(f"  {CLASS_NAMES[a]} <-> {CLASS_NAMES[b]}: {sim:.4f}")
    
    # 3. Per-class feature spread
    print("\n--- PER-CLASS FEATURE SPREAD ---")
    for cid in class_ids:
        feats = [r["feature"] for r in val_records if r["gt_class"] == cid]
        if not feats:
            continue
        feats = np.stack(feats)
        intra_sims = []
        for i in range(len(feats)):
            for j in range(i+1, min(len(feats), i+50)):
                intra_sims.append(float(feats[i] @ feats[j]))
        proto_sims = [float(f @ protos_5[cid]) for f in feats]
        print(f"  {CLASS_NAMES[cid]}: n={len(feats)}, "
              f"intra_sim={np.mean(intra_sims):.4f}±{np.std(intra_sims):.4f}, "
              f"proto_sim={np.mean(proto_sims):.4f}±{np.std(proto_sims):.4f}")
    
    # 4. Grid search: n_support + morph_strength
    best_config, all_results = grid_search_params(train_records, val_records, class_ids)
    
    output = {
        "confusion": {CLASS_NAMES[gt]: {CLASS_NAMES[p]: c for p, c in row.items()} for gt, row in confusion.items()},
        "best_config": best_config,
        "grid_search_top10": all_results[:10],
    }
    
    out_path = Path(__file__).parent / "deep_diagnosis_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    
    return best_config


if __name__ == "__main__":
    main()
