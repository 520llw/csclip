#!/usr/bin/env python3
"""
Refine kNN classifier: search k, weighting scheme, combine with morph/bias.
Goal: push Accuracy > 93%, Macro F1 > 0.86
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
BEST_ENC = {"cell_margin": 0.10, "context_margin": 0.30, "bg_value": 128,
            "cell_weight": 0.85, "context_weight": 0.15}


def load_yolo(lp):
    anns = []
    if not lp.exists():
        return anns
    with open(lp) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 7:
                continue
            c = int(p[0])
            if c not in CLASS_NAMES:
                continue
            anns.append({"class_id": c, "points": [float(x) for x in p[1:]], "ann_type": "polygon"})
    return anns


def build_items(split):
    idir = DATA_ROOT / "images" / split
    ldir = DATA_ROOT / "labels_polygon" / split
    items = []
    for ip in sorted(idir.glob("*.png")):
        anns = load_yolo(ldir / (ip.stem + ".txt"))
        if anns:
            items.append({"image_path": str(ip), "filename": ip.name, "annotations": anns})
    return items


def ann2inst(ann, h, w, iid):
    pts = ann["points"]
    xs = [pts[i] * w for i in range(0, len(pts), 2)]
    ys = [pts[i] * h for i in range(1, len(pts), 2)]
    rr, cc = sk_polygon(ys, xs, shape=(h, w))
    if len(rr) == 0:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[rr, cc] = True
    return InstanceInfo(instance_id=iid, class_id=ann["class_id"],
                        bbox=(max(0, int(np.min(cc))), max(0, int(np.min(rr))),
                              min(w, int(np.max(cc))+1), min(h, int(np.max(rr))+1)), mask=mask)


def metrics(gt, pred, cids):
    total = len(gt)
    correct = sum(int(g == p) for g, p in zip(gt, pred))
    pc, f1s = {}, []
    for c in cids:
        tp = sum(1 for g, p in zip(gt, pred) if g == c and p == c)
        pp = sum(1 for p in pred if p == c)
        gp = sum(1 for g in gt if g == c)
        pr = tp / pp if pp else 0.0
        rc = tp / gp if gp else 0.0
        f1 = 2*pr*rc/(pr+rc) if pr+rc else 0.0
        pc[c] = {"p": pr, "r": rc, "f1": f1, "n": gp}
        f1s.append(f1)
    return {"acc": correct/total if total else 0.0, "mf1": float(np.mean(f1s)), "pc": pc, "total": total}


def encode(model, preprocess, device, image, inst):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = inst.bbox
    bw, bh = x2-x1, y2-y1
    cm, ctxm = BEST_ENC["cell_margin"], BEST_ENC["context_margin"]
    bg = BEST_ENC["bg_value"]
    cw, ctxw = BEST_ENC["cell_weight"], BEST_ENC["context_weight"]

    mx, my = int(bw*cm), int(bh*cm)
    crop = image[max(0,y1-my):min(h,y2+my), max(0,x1-mx):min(w,x2+mx)].copy()
    mc = inst.mask[max(0,y1-my):min(h,y2+my), max(0,x1-mx):min(w,x2+mx)]
    crop = np.where(mc[...,None], crop, np.full_like(crop, bg))

    cmx, cmy = int(bw*ctxm), int(bh*ctxm)
    ctx = image[max(0,y1-cmy):min(h,y2+cmy), max(0,x1-cmx):min(w,x2+cmx)].copy()

    ct = preprocess(Image.fromarray(crop)).unsqueeze(0).to(device)
    cxt = preprocess(Image.fromarray(ctx)).unsqueeze(0).to(device)
    with torch.no_grad():
        cf = model.encode_image(ct); cf = cf/cf.norm(dim=-1,keepdim=True)
        xf = model.encode_image(cxt); xf = xf/xf.norm(dim=-1,keepdim=True)
    fused = cw*cf + ctxw*xf
    fused = fused/fused.norm(dim=-1,keepdim=True)
    return fused.squeeze(0).cpu().numpy().astype(np.float32)


def extract(items, model, preprocess, device):
    recs = []
    for item in items:
        img = np.array(Image.open(item["image_path"]).convert("RGB"))
        h, w = img.shape[:2]
        for i, ann in enumerate(item["annotations"]):
            inst = ann2inst(ann, h, w, i+1)
            if inst is None:
                continue
            feat = encode(model, preprocess, device, img, inst)
            morph = compute_morphology_features(image=img, instance=inst)
            recs.append({"gt": ann["class_id"], "feat": feat, "morph": morph})
    return recs


def knn_predict(q_feat, support_feats, cids, k, weighting="distance", class_weight=None):
    all_sims, all_labels = [], []
    for c in cids:
        sfs = support_feats[c]
        sims = sfs @ q_feat
        all_sims.extend(sims.tolist())
        all_labels.extend([c]*len(sims))
    all_sims = np.array(all_sims)
    all_labels = np.array(all_labels)
    top_k = np.argsort(-all_sims)[:k]

    votes = defaultdict(float)
    for idx in top_k:
        label = all_labels[idx]
        if weighting == "distance":
            w = all_sims[idx]
        elif weighting == "uniform":
            w = 1.0
        elif weighting == "squared":
            w = all_sims[idx] ** 2
        else:
            w = all_sims[idx]
        if class_weight and label in class_weight:
            w *= class_weight[label]
        votes[label] += w
    return max(votes, key=votes.get)


def main():
    print("=" * 70)
    print("kNN REFINEMENT — Push for >93% Acc, >0.86 Macro F1")
    print("=" * 70)

    train_items = build_items("train")
    val_items = build_items("val")
    cids = sorted(CLASS_NAMES.keys())

    from labeling_tool.fewshot_biomedclip import _load_model_bundle
    bundle = _load_model_bundle("auto")
    model, preprocess, device = bundle["model"], bundle["preprocess"], bundle["device"]

    print("Extracting features...")
    train_recs = extract(train_items, model, preprocess, device)
    val_recs = extract(val_items, model, preprocess, device)

    pc = defaultdict(list)
    for r in train_recs:
        pc[r["gt"]].append(r["feat"])
    sf_all = {c: np.stack(feats) for c, feats in pc.items()}
    
    # EXP 1: Fine-grained k search
    print("\n>>> Fine-grained k search (all supports, distance weighting) <<<")
    best_f1, best_k = 0, 5
    for k in range(1, 35):
        gt, pred = [], []
        for r in val_recs:
            p = knn_predict(r["feat"], sf_all, cids, k, "distance")
            gt.append(r["gt"]); pred.append(p)
        m = metrics(gt, pred, cids)
        eos = m["pc"].get(3, {})
        marker = " ***" if m["mf1"] > best_f1 else ""
        if k <= 15 or m["mf1"] > best_f1:
            print(f"  k={k:>3}: Acc={m['acc']:.4f} mF1={m['mf1']:.4f} Eos={eos.get('f1',0):.3f}{marker}")
        if m["mf1"] > best_f1:
            best_f1 = m["mf1"]; best_k = k

    # EXP 2: Weighting schemes
    print(f"\n>>> Weighting schemes (k={best_k}) <<<")
    for wt in ["uniform", "distance", "squared"]:
        gt, pred = [], []
        for r in val_recs:
            p = knn_predict(r["feat"], sf_all, cids, best_k, wt)
            gt.append(r["gt"]); pred.append(p)
        m = metrics(gt, pred, cids)
        eos = m["pc"].get(3, {})
        print(f"  {wt:<10}: Acc={m['acc']:.4f} mF1={m['mf1']:.4f} Eos={eos.get('f1',0):.3f}")

    # EXP 3: Class weighting to boost Eosinophil
    print(f"\n>>> Class weight search (k={best_k}) <<<")
    best_cw_f1, best_cw = 0, None
    for eos_w in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]:
        for mac_w in [0.8, 0.9, 1.0, 1.1, 1.2]:
            cw = {3: eos_w, 4: 1.0, 5: 1.0, 6: mac_w}
            gt, pred = [], []
            for r in val_recs:
                p = knn_predict(r["feat"], sf_all, cids, best_k, "distance", class_weight=cw)
                gt.append(r["gt"]); pred.append(p)
            m = metrics(gt, pred, cids)
            if m["mf1"] > best_cw_f1:
                best_cw_f1 = m["mf1"]; best_cw = cw.copy()
                eos = m["pc"].get(3, {})
                print(f"  NEW BEST: eos_w={eos_w:.1f} mac_w={mac_w:.1f} -> Acc={m['acc']:.4f} "
                      f"mF1={m['mf1']:.4f} Eos={eos.get('f1',0):.3f} EosP={eos.get('p',0):.3f} EosR={eos.get('r',0):.3f}")

    # EXP 4: kNN + morph constraints
    print(f"\n>>> kNN + morphology (k={best_k}, best class weights) <<<")
    for ms in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
        gt, pred = [], []
        for r in val_recs:
            all_sims, all_labels = [], []
            for c in cids:
                sims = sf_all[c] @ r["feat"]
                all_sims.extend(sims.tolist())
                all_labels.extend([c]*len(sims))
            all_sims = np.array(all_sims)
            all_labels = np.array(all_labels)
            top_k = np.argsort(-all_sims)[:best_k]

            votes = defaultdict(float)
            for idx in top_k:
                label = all_labels[idx]
                w = all_sims[idx]
                if best_cw and label in best_cw:
                    w *= best_cw[label]
                votes[label] += w

            class_scores = np.array([votes.get(c, 0) for c in cids], dtype=np.float32)
            if ms > 0:
                class_scores = apply_morphology_constraints(class_scores, r["morph"], cids, strength=ms)
            gt.append(r["gt"])
            pred.append(cids[int(np.argmax(class_scores))])
        m = metrics(gt, pred, cids)
        eos = m["pc"].get(3, {})
        print(f"  ms={ms:.3f}: Acc={m['acc']:.4f} mF1={m['mf1']:.4f} Eos={eos.get('f1',0):.3f}")

    # EXP 5: kNN with distance-weighted morph fusion
    print(f"\n>>> kNN + morph feature fusion (k={best_k}) <<<")
    # Build morph prototypes
    morph_pc = defaultdict(list)
    for r in train_recs:
        morph_pc[r["gt"]].append(r["morph"])
    morph_protos = {c: np.stack(feats).mean(0) for c, feats in morph_pc.items()}
    morph_stds = {c: np.stack(feats).std(0) + 1e-6 for c, feats in morph_pc.items()}

    for morph_w in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        gt, pred = [], []
        for r in val_recs:
            votes = defaultdict(float)
            for c in cids:
                sims = sf_all[c] @ r["feat"]
                top_indices = np.argsort(-sims)[:best_k]
                img_score = float(np.sum(sims[top_indices]))
                
                morph_sim = float(np.exp(-0.5 * np.sum(((r["morph"] - morph_protos[c]) / morph_stds[c]) ** 2)))
                
                combined = img_score + morph_w * morph_sim
                if best_cw and c in best_cw:
                    combined *= best_cw[c]
                votes[c] = combined
            gt.append(r["gt"])
            pred.append(max(votes, key=votes.get))
        m = metrics(gt, pred, cids)
        eos = m["pc"].get(3, {})
        marker = " ***" if m["mf1"] > best_cw_f1 else ""
        print(f"  morph_w={morph_w:.2f}: Acc={m['acc']:.4f} mF1={m['mf1']:.4f} Eos={eos.get('f1',0):.3f}{marker}")
        if m["mf1"] > best_cw_f1:
            best_cw_f1 = m["mf1"]

    # FINAL SUMMARY
    print(f"\n{'='*70}")
    print(f"BEST OVERALL: k={best_k}, class_weights={best_cw}")
    print(f"  Macro F1 = {best_cw_f1:.4f}")
    gt, pred = [], []
    for r in val_recs:
        p = knn_predict(r["feat"], sf_all, cids, best_k, "distance", class_weight=best_cw)
        gt.append(r["gt"]); pred.append(p)
    m = metrics(gt, pred, cids)
    print(f"  Accuracy = {m['acc']:.4f}")
    for c in cids:
        pc_m = m["pc"].get(c, {})
        print(f"  {CLASS_NAMES[c]:<15}: P={pc_m.get('p',0):.4f} R={pc_m.get('r',0):.4f} F1={pc_m.get('f1',0):.4f}")

    out = {"best_k": best_k, "best_class_weights": best_cw, "best_f1": best_cw_f1, "final_metrics": m}
    with open(Path(__file__).parent / "knn_refine_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
