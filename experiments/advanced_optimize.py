#!/usr/bin/env python3
"""
Advanced optimization:
1. Temperature scaling search
2. Class-specific bias tuning (for Eosinophil)
3. Prototype enhancement via support filtering
4. Nearest-neighbour classifier (kNN instead of prototype)
5. Combined best pipeline
"""
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biomedclip_zeroshot_cell_classify import InstanceInfo
from biomedclip_query_adaptive_classifier import compute_morphology_features
from labeling_tool.morphology_constraints import apply_morphology_constraints
from PIL import Image
from skimage.draw import polygon as sk_polygon

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
RANDOM_SEED = 42

# Best encoding from previous iteration
BEST_ENC = {"cell_margin": 0.10, "context_margin": 0.30, "bg_value": 128,
            "cell_weight": 0.85, "context_weight": 0.15}


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


def annotation_to_instance(ann, h, w, iid):
    pts = ann["points"]
    xs = [pts[i] * w for i in range(0, len(pts), 2)]
    ys = [pts[i] * h for i in range(1, len(pts), 2)]
    rr, cc = sk_polygon(ys, xs, shape=(h, w))
    if len(rr) == 0:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[rr, cc] = True
    x1, y1 = max(0, int(np.min(cc))), max(0, int(np.min(rr)))
    x2, y2 = min(w, int(np.max(cc)) + 1), min(h, int(np.max(rr)) + 1)
    return InstanceInfo(instance_id=iid, class_id=ann["class_id"], bbox=(x1, y1, x2, y2), mask=mask)


def compute_metrics(gt, pred, cids):
    total = len(gt)
    correct = sum(int(g == p) for g, p in zip(gt, pred))
    per_class, f1s = {}, []
    for c in cids:
        tp = sum(1 for g, p in zip(gt, pred) if g == c and p == c)
        pp = sum(1 for p in pred if p == c)
        gp = sum(1 for g in gt if g == c)
        pr = tp / pp if pp else 0.0
        rc = tp / gp if gp else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per_class[c] = {"precision": pr, "recall": rc, "f1": f1, "support": gp}
        f1s.append(f1)
    return {"accuracy": correct / total if total else 0.0, "macro_f1": float(np.mean(f1s)),
            "per_class": per_class, "total": total}


def encode_cell(model, preprocess, device, image, inst):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = inst.bbox
    bw, bh = x2 - x1, y2 - y1
    cm = BEST_ENC["cell_margin"]
    ctxm = BEST_ENC["context_margin"]
    bg = BEST_ENC["bg_value"]
    cw = BEST_ENC["cell_weight"]
    ctxw = BEST_ENC["context_weight"]

    mx, my = int(bw * cm), int(bh * cm)
    cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
    cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)
    crop = image[cy1:cy2, cx1:cx2].copy()
    mask_crop = inst.mask[cy1:cy2, cx1:cx2]
    bgv = np.full_like(crop, bg)
    crop = np.where(mask_crop[..., None], crop, bgv)

    cmx, cmy = int(bw * ctxm), int(bh * ctxm)
    ccx1, ccy1 = max(0, x1 - cmx), max(0, y1 - cmy)
    ccx2, ccy2 = min(w, x2 + cmx), min(h, y2 + cmy)
    ctx_crop = image[ccy1:ccy2, ccx1:ccx2].copy()

    ct = preprocess(Image.fromarray(crop)).unsqueeze(0).to(device)
    cxt = preprocess(Image.fromarray(ctx_crop)).unsqueeze(0).to(device)

    with torch.no_grad():
        cf = model.encode_image(ct)
        cf = cf / cf.norm(dim=-1, keepdim=True)
        xf = model.encode_image(cxt)
        xf = xf / xf.norm(dim=-1, keepdim=True)

    fused = cw * cf + ctxw * xf
    fused = fused / fused.norm(dim=-1, keepdim=True)
    return fused.squeeze(0).cpu().numpy().astype(np.float32)


def extract_all(items, model, preprocess, device):
    recs = []
    for item in items:
        img = np.array(Image.open(item["image_path"]).convert("RGB"))
        h, w = img.shape[:2]
        for i, ann in enumerate(item["annotations"]):
            inst = annotation_to_instance(ann, h, w, i + 1)
            if inst is None:
                continue
            feat = encode_cell(model, preprocess, device, img, inst)
            morph = compute_morphology_features(image=img, instance=inst)
            recs.append({"gt": ann["class_id"], "feat": feat, "morph": morph})
    return recs


def build_protos(recs, cids, n=100):
    random.seed(RANDOM_SEED)
    pc = defaultdict(list)
    for r in recs:
        pc[r["gt"]].append(r)
    protos = {}
    support_feats = {}
    for c in cids:
        cands = pc.get(c, [])
        if not cands:
            continue
        chosen = random.sample(cands, min(n, len(cands)))
        feats = np.stack([x["feat"] for x in chosen])
        proto = feats.mean(0)
        protos[c] = proto / np.linalg.norm(proto)
        support_feats[c] = feats
    return protos, support_feats


def knn_classify(query_feat, support_feats, cids, k=5):
    """k-Nearest Neighbour classification."""
    all_sims = []
    all_labels = []
    for c in cids:
        sfs = support_feats[c]
        sims = sfs @ query_feat
        all_sims.extend(sims.tolist())
        all_labels.extend([c] * len(sims))
    
    all_sims = np.array(all_sims)
    all_labels = np.array(all_labels)
    top_k = np.argsort(-all_sims)[:k]
    votes = defaultdict(float)
    for idx in top_k:
        votes[all_labels[idx]] += all_sims[idx]
    
    return max(votes, key=votes.get)


def main():
    print("=" * 70)
    print("ADVANCED OPTIMIZATION")
    print("=" * 70)

    train_items = build_dataset_items("train")
    val_items = build_dataset_items("val")
    cids = sorted(CLASS_NAMES.keys())

    from labeling_tool.fewshot_biomedclip import _load_model_bundle
    bundle = _load_model_bundle("auto")
    model, preprocess, device = bundle["model"], bundle["preprocess"], bundle["device"]

    print("Extracting features...")
    train_recs = extract_all(train_items, model, preprocess, device)
    val_recs = extract_all(val_items, model, preprocess, device)
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    protos, support_feats = build_protos(train_recs, cids, n=100)

    # Experiment 1: Temperature scaling
    print("\n>>> EXP 1: Temperature Scaling <<<")
    for temp in [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]:
        gt, pred = [], []
        for r in val_recs:
            scores = np.array([float(r["feat"] @ protos[c]) for c in cids])
            shifted = (scores - scores.max()) / max(temp, 1e-8)
            probs = np.exp(shifted) / np.exp(shifted).sum()
            gt.append(r["gt"])
            pred.append(cids[int(np.argmax(probs))])
        m = compute_metrics(gt, pred, cids)
        print(f"  T={temp:.3f}: Acc={m['accuracy']:.4f} mF1={m['macro_f1']:.4f}")

    # Experiment 2: Class-specific bias
    print("\n>>> EXP 2: Class Bias Tuning <<<")
    best_f1 = 0
    best_bias = None
    for eos_bias in np.arange(-0.02, 0.03, 0.002):
        for mac_bias in np.arange(-0.02, 0.02, 0.005):
            gt, pred = [], []
            for r in val_recs:
                scores = np.array([float(r["feat"] @ protos[c]) for c in cids])
                # Apply biases
                for i, c in enumerate(cids):
                    if c == 3:
                        scores[i] += eos_bias
                    elif c == 6:
                        scores[i] += mac_bias
                gt.append(r["gt"])
                pred.append(cids[int(np.argmax(scores))])
            m = compute_metrics(gt, pred, cids)
            if m["macro_f1"] > best_f1:
                best_f1 = m["macro_f1"]
                best_bias = {"eos": float(eos_bias), "mac": float(mac_bias)}
                eos_m = m["per_class"].get(3, {})
                print(f"  NEW BEST: eos_bias={eos_bias:.3f} mac_bias={mac_bias:.3f} -> "
                      f"Acc={m['accuracy']:.4f} mF1={m['macro_f1']:.4f} Eos_F1={eos_m.get('f1',0):.3f}")

    print(f"\nBest bias: {best_bias}")

    # Experiment 3: kNN classifier
    print("\n>>> EXP 3: kNN Classifier <<<")
    protos_all, sf_all = build_protos(train_recs, cids, n=9999)
    for k in [1, 3, 5, 7, 11, 15, 21, 31, 51]:
        gt, pred = [], []
        for r in val_recs:
            p = knn_classify(r["feat"], sf_all, cids, k=k)
            gt.append(r["gt"])
            pred.append(p)
        m = compute_metrics(gt, pred, cids)
        eos_m = m["per_class"].get(3, {})
        print(f"  k={k:>3}: Acc={m['accuracy']:.4f} mF1={m['macro_f1']:.4f} Eos_F1={eos_m.get('f1',0):.3f}")

    # Experiment 4: Weighted kNN (class-balanced)
    print("\n>>> EXP 4: Weighted kNN (class-balanced) <<<")
    protos_bal, sf_bal = build_protos(train_recs, cids, n=100)
    for k in [3, 5, 7, 11, 15, 21]:
        gt, pred = [], []
        for r in val_recs:
            p = knn_classify(r["feat"], sf_bal, cids, k=k)
            gt.append(r["gt"])
            pred.append(p)
        m = compute_metrics(gt, pred, cids)
        eos_m = m["per_class"].get(3, {})
        print(f"  k={k:>3} (bal100): Acc={m['accuracy']:.4f} mF1={m['macro_f1']:.4f} Eos_F1={eos_m.get('f1',0):.3f}")

    # Experiment 5: Combined best = proto + bias + morph
    print("\n>>> EXP 5: Combined Best Pipeline <<<")
    if best_bias:
        for ms in [0.0, 0.005, 0.01, 0.015, 0.02]:
            gt, pred = [], []
            for r in val_recs:
                scores = np.array([float(r["feat"] @ protos[c]) for c in cids])
                for i, c in enumerate(cids):
                    if c == 3:
                        scores[i] += best_bias["eos"]
                    elif c == 6:
                        scores[i] += best_bias["mac"]
                if ms > 0:
                    scores = apply_morphology_constraints(scores, r["morph"], cids, strength=ms)
                gt.append(r["gt"])
                pred.append(cids[int(np.argmax(scores))])
            m = compute_metrics(gt, pred, cids)
            eos_m = m["per_class"].get(3, {})
            print(f"  proto+bias+morph(s={ms:.3f}): Acc={m['accuracy']:.4f} mF1={m['macro_f1']:.4f} "
                  f"Eos={eos_m.get('f1',0):.3f} EosP={eos_m.get('precision',0):.3f} EosR={eos_m.get('recall',0):.3f}")

    # Experiment 6: kNN + bias  
    print("\n>>> EXP 6: kNN + class bias <<<")
    if best_bias:
        for k in [5, 7, 11, 15, 21]:
            gt, pred = [], []
            for r in val_recs:
                all_sims, all_labels = [], []
                for c in cids:
                    sfs = sf_bal[c]
                    sims = sfs @ r["feat"]
                    bias = best_bias.get("eos", 0) if c == 3 else (best_bias.get("mac", 0) if c == 6 else 0)
                    all_sims.extend((sims + bias).tolist())
                    all_labels.extend([c] * len(sims))
                all_sims = np.array(all_sims)
                all_labels = np.array(all_labels)
                top_k = np.argsort(-all_sims)[:k]
                votes = defaultdict(float)
                for idx in top_k:
                    votes[all_labels[idx]] += all_sims[idx]
                gt.append(r["gt"])
                pred.append(max(votes, key=votes.get))
            m = compute_metrics(gt, pred, cids)
            eos_m = m["per_class"].get(3, {})
            print(f"  kNN(k={k})+bias: Acc={m['accuracy']:.4f} mF1={m['macro_f1']:.4f} "
                  f"Eos_F1={eos_m.get('f1',0):.3f}")

    out_path = Path(__file__).parent / "advanced_results.json"
    with open(out_path, "w") as f:
        json.dump({"best_bias": best_bias}, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
