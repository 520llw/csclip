#!/usr/bin/env python3
"""
Iterative optimization: search encoding params, support strategies, 
and combined configurations to maximize Macro F1.

Key insights from diagnosis:
- n_support=20 is much better than 5 (F1 +5.83%)
- Inter-class prototype similarity is very high (0.96+)
- Need to maximize prototype separation in embedding space
"""
import sys
import os
import json
import random
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biomedclip_zeroshot_cell_classify import InstanceInfo, resolve_device
from biomedclip_fewshot_support_experiment import normalize_feature
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
    return {"accuracy": correct / total if total else 0.0, "macro_f1": float(np.mean(f1_values)),
            "per_class": per_class, "total": total}


def _crop_and_encode(model, preprocess, device, image, instance,
                     cell_margin=0.15, context_margin=0.30, bg_value=128,
                     cell_weight=0.90, context_weight=0.10):
    """Encode with configurable parameters."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = instance.bbox
    bw, bh = x2 - x1, y2 - y1
    
    # Cell crop
    mx = int(bw * cell_margin)
    my = int(bh * cell_margin)
    cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
    cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)
    cell_crop = image[cy1:cy2, cx1:cx2].copy()
    cell_mask = instance.mask[cy1:cy2, cx1:cx2]
    bg = np.full_like(cell_crop, bg_value)
    cell_crop = np.where(cell_mask[..., None] if cell_crop.ndim == 3 else cell_mask, cell_crop, bg)
    
    # Context crop
    cmx = int(bw * context_margin)
    cmy = int(bh * context_margin)
    ccx1, ccy1 = max(0, x1 - cmx), max(0, y1 - cmy)
    ccx2, ccy2 = min(w, x2 + cmx), min(h, y2 + cmy)
    ctx_crop = image[ccy1:ccy2, ccx1:ccx2].copy()
    
    # Encode
    cell_tensor = preprocess(Image.fromarray(cell_crop)).unsqueeze(0).to(device)
    ctx_tensor = preprocess(Image.fromarray(ctx_crop)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        cell_feat = model.encode_image(cell_tensor)
        cell_feat = cell_feat / cell_feat.norm(dim=-1, keepdim=True)
        ctx_feat = model.encode_image(ctx_tensor)
        ctx_feat = ctx_feat / ctx_feat.norm(dim=-1, keepdim=True)
    
    fused = cell_weight * cell_feat + context_weight * ctx_feat
    fused = fused / fused.norm(dim=-1, keepdim=True)
    return fused.squeeze(0).cpu().numpy().astype(np.float32)


def extract_features_with_params(items, model, preprocess, device, **enc_params):
    records = []
    for item in items:
        image = np.array(Image.open(item["image_path"]).convert("RGB"))
        h, w = image.shape[:2]
        for idx, ann in enumerate(item["annotations"]):
            inst = annotation_to_instance(ann, h, w, idx + 1)
            if inst is None:
                continue
            feat = _crop_and_encode(model, preprocess, device, image, inst, **enc_params)
            morph = compute_morphology_features(image=image, instance=inst)
            records.append({"gt_class": ann["class_id"], "feature": feat, "morph": morph})
    return records


def build_prototypes(records, class_ids, n_support, strategy="random"):
    random.seed(RANDOM_SEED)
    per_class = defaultdict(list)
    for r in records:
        per_class[r["gt_class"]].append(r)
    
    prototypes = {}
    for cid in class_ids:
        cands = per_class.get(cid, [])
        if not cands:
            continue
        
        if strategy == "random":
            chosen = random.sample(cands, min(n_support, len(cands)))
        elif strategy == "diverse":
            chosen = _select_diverse(cands, n_support)
        elif strategy == "all":
            chosen = cands
        else:
            chosen = random.sample(cands, min(n_support, len(cands)))
        
        feats = np.stack([c["feature"] for c in chosen])
        proto = feats.mean(axis=0)
        prototypes[cid] = proto / np.linalg.norm(proto)
    return prototypes


def _select_diverse(candidates, n):
    """Select diverse support samples using farthest-point sampling."""
    if len(candidates) <= n:
        return candidates
    
    feats = np.stack([c["feature"] for c in candidates])
    sims = feats @ feats.T
    
    selected = [random.randint(0, len(candidates) - 1)]
    for _ in range(n - 1):
        min_sims = np.array([sims[s, selected].max() for s in range(len(candidates))])
        for s in selected:
            min_sims[s] = 999
        next_idx = int(np.argmin(min_sims))
        selected.append(next_idx)
    
    return [candidates[i] for i in selected]


def evaluate_config(val_records, prototypes, class_ids, morph_strength=0.0):
    gt_labels, pred_labels = [], []
    for r in val_records:
        scores = np.array([float(r["feature"] @ prototypes[cid]) for cid in class_ids])
        if morph_strength > 0:
            scores = apply_morphology_constraints(scores, r["morph"], class_ids, strength=morph_strength)
        pred = class_ids[int(np.argmax(scores))]
        gt_labels.append(r["gt_class"])
        pred_labels.append(pred)
    return compute_metrics(gt_labels, pred_labels, class_ids)


def main():
    print("=" * 70)
    print("ITERATIVE OPTIMIZATION — Finding Best Full Pipeline Configuration")
    print("=" * 70)
    
    train_items = build_dataset_items("train")
    val_items = build_dataset_items("val")
    class_ids = sorted(CLASS_NAMES.keys())
    
    from labeling_tool.fewshot_biomedclip import _load_model_bundle
    bundle = _load_model_bundle("auto")
    model, preprocess, device = bundle["model"], bundle["preprocess"], bundle["device"]
    
    # Phase 1: Search encoding parameters
    print("\n>>> PHASE 1: Encoding Parameter Search <<<")
    
    encoding_configs = [
        {"cell_margin": 0.10, "context_margin": 0.20, "bg_value": 128, "cell_weight": 0.90, "context_weight": 0.10},
        {"cell_margin": 0.15, "context_margin": 0.30, "bg_value": 128, "cell_weight": 0.90, "context_weight": 0.10},
        {"cell_margin": 0.20, "context_margin": 0.40, "bg_value": 128, "cell_weight": 0.90, "context_weight": 0.10},
        {"cell_margin": 0.15, "context_margin": 0.30, "bg_value": 128, "cell_weight": 0.85, "context_weight": 0.15},
        {"cell_margin": 0.15, "context_margin": 0.30, "bg_value": 128, "cell_weight": 0.95, "context_weight": 0.05},
        {"cell_margin": 0.15, "context_margin": 0.30, "bg_value": 128, "cell_weight": 1.00, "context_weight": 0.00},
        {"cell_margin": 0.15, "context_margin": 0.30, "bg_value": 0, "cell_weight": 0.90, "context_weight": 0.10},
        {"cell_margin": 0.15, "context_margin": 0.30, "bg_value": 200, "cell_weight": 0.90, "context_weight": 0.10},
        {"cell_margin": 0.25, "context_margin": 0.50, "bg_value": 128, "cell_weight": 0.80, "context_weight": 0.20},
        {"cell_margin": 0.10, "context_margin": 0.30, "bg_value": 128, "cell_weight": 0.85, "context_weight": 0.15},
    ]
    
    best_enc = None
    best_f1 = 0
    enc_results = []
    
    for i, enc in enumerate(encoding_configs):
        print(f"\n  Config {i+1}/{len(encoding_configs)}: {enc}")
        train_recs = extract_features_with_params(train_items, model, preprocess, device, **enc)
        val_recs = extract_features_with_params(val_items, model, preprocess, device, **enc)
        
        for n_sup in [10, 20, 30]:
            for strat in ["random", "diverse"]:
                for ms in [0.0, 0.01, 0.02]:
                    protos = build_prototypes(train_recs, class_ids, n_sup, strategy=strat)
                    if len(protos) < len(class_ids):
                        continue
                    metrics = evaluate_config(val_recs, protos, class_ids, morph_strength=ms)
                    
                    result = {
                        "enc_config": enc,
                        "n_support": n_sup,
                        "strategy": strat,
                        "morph_strength": ms,
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "per_class": {CLASS_NAMES[k]: v for k, v in metrics["per_class"].items()},
                    }
                    enc_results.append(result)
                    
                    if metrics["macro_f1"] > best_f1:
                        best_f1 = metrics["macro_f1"]
                        best_enc = result
                        eos = metrics["per_class"].get(3, {})
                        print(f"    NEW BEST: n={n_sup} {strat} ms={ms} -> Acc={metrics['accuracy']:.4f} "
                              f"mF1={metrics['macro_f1']:.4f} Eos_F1={eos.get('f1',0):.3f}")
    
    # Phase 2: Try "all" supports with best encoding
    print("\n>>> PHASE 2: All-Support Prototypes <<<")
    best_enc_params = best_enc["enc_config"] if best_enc else encoding_configs[1]
    
    train_recs = extract_features_with_params(train_items, model, preprocess, device, **best_enc_params)
    val_recs = extract_features_with_params(val_items, model, preprocess, device, **best_enc_params)
    
    for ms in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]:
        protos = build_prototypes(train_recs, class_ids, 9999, strategy="all")
        if len(protos) < len(class_ids):
            continue
        metrics = evaluate_config(val_recs, protos, class_ids, morph_strength=ms)
        eos = metrics["per_class"].get(3, {})
        marker = " *** BEST ***" if metrics["macro_f1"] > best_f1 else ""
        print(f"  ALL supports, ms={ms:.3f}: Acc={metrics['accuracy']:.4f} "
              f"mF1={metrics['macro_f1']:.4f} Eos={eos.get('f1',0):.3f}{marker}")
        
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_enc = {
                "enc_config": best_enc_params,
                "n_support": "all",
                "strategy": "all",
                "morph_strength": ms,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "per_class": {CLASS_NAMES[k]: v for k, v in metrics["per_class"].items()},
            }
    
    # Phase 3: Try class-balanced reweighted prototypes
    print("\n>>> PHASE 3: Class-Balanced Re-weighted Prototypes <<<")
    per_class_records = defaultdict(list)
    for r in train_recs:
        per_class_records[r["gt_class"]].append(r)
    
    min_count = min(len(v) for v in per_class_records.values())
    
    for max_per_class in [min_count, 50, 100, 200]:
        random.seed(RANDOM_SEED)
        protos = {}
        for cid in class_ids:
            cands = per_class_records.get(cid, [])
            if len(cands) > max_per_class:
                chosen = random.sample(cands, max_per_class)
            else:
                chosen = cands
            if not chosen:
                continue
            feats = np.stack([c["feature"] for c in chosen])
            proto = feats.mean(axis=0)
            protos[cid] = proto / np.linalg.norm(proto)
        
        if len(protos) < len(class_ids):
            continue
        
        for ms in [0.0, 0.01, 0.02]:
            metrics = evaluate_config(val_recs, protos, class_ids, morph_strength=ms)
            eos = metrics["per_class"].get(3, {})
            marker = " *** BEST ***" if metrics["macro_f1"] > best_f1 else ""
            print(f"  Balanced n={max_per_class}, ms={ms:.2f}: Acc={metrics['accuracy']:.4f} "
                  f"mF1={metrics['macro_f1']:.4f} Eos={eos.get('f1',0):.3f}{marker}")
            
            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                best_enc = {
                    "enc_config": best_enc_params,
                    "n_support": max_per_class,
                    "strategy": "balanced",
                    "morph_strength": ms,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "per_class": {CLASS_NAMES[k]: v for k, v in metrics["per_class"].items()},
                }
    
    print("\n" + "=" * 70)
    print("FINAL BEST CONFIGURATION")
    print("=" * 70)
    print(json.dumps(best_enc, indent=2, default=str))
    
    out_path = Path(__file__).parent / "optimization_results.json"
    with open(out_path, "w") as f:
        json.dump({"best": best_enc, "all_results_count": len(enc_results)}, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    
    return best_enc


if __name__ == "__main__":
    main()
