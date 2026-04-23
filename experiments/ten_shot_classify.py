#!/usr/bin/env python3
"""
Strict 10-shot classification experiment.

Rules:
  - Each class uses EXACTLY 10 support samples from train set
  - No other train data is used at inference time
  - Val set is the test set (36 images, 1316 cells)
  - BiomedCLIP weights are FROZEN (feature extractor only)
  - Multiple strategies tested: prototype, kNN-10, hybrid
"""
import sys
import json
import random
import math
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
N_SHOT = 10
SEEDS = [42, 123, 456, 789, 2026]


def load_yolo(lp):
    anns = []
    if not lp.exists():
        return anns
    for line in open(lp):
        p = line.strip().split()
        if len(p) < 7:
            continue
        c = int(p[0])
        if c in CLASS_NAMES:
            anns.append({"class_id": c, "points": [float(x) for x in p[1:]]})
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
    return {"acc": correct/total if total else 0, "mf1": float(np.mean(f1s)), "pc": pc}


def encode_cell(model, preprocess, device, image, inst,
                cell_margin=0.10, ctx_margin=0.30, bg=128, cw=0.85, ctxw=0.15):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = inst.bbox
    bw, bh = x2 - x1, y2 - y1

    mx, my = int(bw * cell_margin), int(bh * cell_margin)
    crop = image[max(0,y1-my):min(h,y2+my), max(0,x1-mx):min(w,x2+mx)].copy()
    mc = inst.mask[max(0,y1-my):min(h,y2+my), max(0,x1-mx):min(w,x2+mx)]
    crop = np.where(mc[..., None], crop, np.full_like(crop, bg))

    cmx, cmy = int(bw * ctx_margin), int(bh * ctx_margin)
    ctx = image[max(0,y1-cmy):min(h,y2+cmy), max(0,x1-cmx):min(w,x2+cmx)].copy()

    ct = preprocess(Image.fromarray(crop)).unsqueeze(0).to(device)
    cxt = preprocess(Image.fromarray(ctx)).unsqueeze(0).to(device)
    with torch.no_grad():
        cf = model.encode_image(ct); cf /= cf.norm(dim=-1, keepdim=True)
        xf = model.encode_image(cxt); xf /= xf.norm(dim=-1, keepdim=True)
    fused = cw * cf + ctxw * xf
    fused /= fused.norm(dim=-1, keepdim=True)
    return fused.squeeze(0).cpu().numpy().astype(np.float32)


def extract_all(items, model, preprocess, device):
    recs = []
    for item in items:
        img = np.array(Image.open(item["image_path"]).convert("RGB"))
        h, w = img.shape[:2]
        for i, ann in enumerate(item["annotations"]):
            inst = ann2inst(ann, h, w, i + 1)
            if inst is None:
                continue
            feat = encode_cell(model, preprocess, device, img, inst)
            morph = compute_morphology_features(image=img, instance=inst)
            recs.append({"gt": ann["class_id"], "feat": feat, "morph": morph,
                         "img": item["image_path"], "idx": i})
    return recs


def select_10shot(train_recs, seed):
    """Select exactly 10 samples per class."""
    random.seed(seed)
    pc = defaultdict(list)
    for r in train_recs:
        pc[r["gt"]].append(r)
    support = {}
    for c in sorted(CLASS_NAMES.keys()):
        cands = pc[c]
        chosen = random.sample(cands, min(N_SHOT, len(cands)))
        support[c] = chosen
    return support


def select_10shot_diverse(train_recs, seed):
    """Select 10 most diverse samples per class via farthest-point sampling."""
    random.seed(seed)
    pc = defaultdict(list)
    for r in train_recs:
        pc[r["gt"]].append(r)
    support = {}
    for c in sorted(CLASS_NAMES.keys()):
        cands = pc[c]
        if len(cands) <= N_SHOT:
            support[c] = cands
            continue
        feats = np.stack([x["feat"] for x in cands])
        sims = feats @ feats.T
        first = random.randint(0, len(cands) - 1)
        selected = [first]
        for _ in range(N_SHOT - 1):
            min_sims = np.array([sims[s, selected].max() for s in range(len(cands))])
            for s in selected:
                min_sims[s] = 999
            selected.append(int(np.argmin(min_sims)))
        support[c] = [cands[i] for i in selected]
    return support


def classify_prototype(val_recs, support, cids):
    protos = {}
    for c in cids:
        feats = np.stack([s["feat"] for s in support[c]])
        p = feats.mean(0)
        protos[c] = p / np.linalg.norm(p)

    gt, pred = [], []
    for r in val_recs:
        scores = np.array([float(r["feat"] @ protos[c]) for c in cids])
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def classify_knn(val_recs, support, cids, k=6, eos_w=1.0):
    sf = {}
    for c in cids:
        sf[c] = np.stack([s["feat"] for s in support[c]])

    gt, pred = [], []
    for r in val_recs:
        all_sims, all_labels = [], []
        for c in cids:
            sims = sf[c] @ r["feat"]
            all_sims.extend(sims.tolist())
            all_labels.extend([c] * len(sims))
        all_sims = np.array(all_sims)
        all_labels = np.array(all_labels)
        top_k = np.argsort(-all_sims)[:k]
        votes = defaultdict(float)
        for idx in top_k:
            w = all_sims[idx]
            label = all_labels[idx]
            if label == 3:
                w *= eos_w
            votes[label] += w
        gt.append(r["gt"])
        pred.append(max(votes, key=votes.get))
    return metrics(gt, pred, cids)


def classify_prototype_morph(val_recs, support, cids, morph_strength=0.02):
    protos = {}
    for c in cids:
        feats = np.stack([s["feat"] for s in support[c]])
        p = feats.mean(0)
        protos[c] = p / np.linalg.norm(p)

    gt, pred = [], []
    for r in val_recs:
        scores = np.array([float(r["feat"] @ protos[c]) for c in cids])
        scores = apply_morphology_constraints(scores, r["morph"], cids, strength=morph_strength)
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def classify_adaptive_knn(val_recs, support, cids, k_per_class=3, eos_w=1.0):
    """Per-class kNN: find k nearest per class, then compare class scores."""
    sf = {c: np.stack([s["feat"] for s in support[c]]) for c in cids}

    gt, pred = [], []
    for r in val_recs:
        class_scores = {}
        for c in cids:
            sims = sf[c] @ r["feat"]
            top_k = np.sort(sims)[::-1][:k_per_class]
            score = float(top_k.mean())
            if c == 3:
                score *= eos_w
            class_scores[c] = score
        gt.append(r["gt"])
        pred.append(max(class_scores, key=class_scores.get))
    return metrics(gt, pred, cids)


def classify_prototype_weighted(val_recs, support, cids, query_adaptive=True):
    """Prototype with per-query adaptive weighting based on support similarity spread."""
    protos = {}
    sf = {}
    for c in cids:
        feats = np.stack([s["feat"] for s in support[c]])
        p = feats.mean(0)
        protos[c] = p / np.linalg.norm(p)
        sf[c] = feats

    gt, pred = [], []
    for r in val_recs:
        scores = []
        for c in cids:
            proto_sim = float(r["feat"] @ protos[c])
            if query_adaptive:
                support_sims = sf[c] @ r["feat"]
                spread = float(np.std(support_sims))
                max_sim = float(np.max(support_sims))
                score = 0.5 * proto_sim + 0.5 * max_sim
            else:
                score = proto_sim
            scores.append(score)
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def main():
    print("=" * 70)
    print("STRICT 10-SHOT CLASSIFICATION (10 samples per class)")
    print("=" * 70)

    train_items = build_items("train")
    val_items = build_items("val")
    cids = sorted(CLASS_NAMES.keys())

    from labeling_tool.fewshot_biomedclip import _load_model_bundle
    bundle = _load_model_bundle("auto")
    model, preprocess, device = bundle["model"], bundle["preprocess"], bundle["device"]

    print("Extracting all features (one-time cost)...")
    train_recs = extract_all(train_items, model, preprocess, device)
    val_recs = extract_all(val_items, model, preprocess, device)
    print(f"  Train: {len(train_recs)} cells, Val: {len(val_recs)} cells")
    print(f"  Support budget: {N_SHOT} per class = {N_SHOT * len(cids)} total")

    # ===== Strategy comparison across multiple seeds =====
    strategies = {
        "A: proto (random)":        lambda vr, sup: classify_prototype(vr, sup, cids),
        "B: proto (diverse)":       None,
        "C: kNN k=6":              lambda vr, sup: classify_knn(vr, sup, cids, k=6),
        "D: kNN k=4":              lambda vr, sup: classify_knn(vr, sup, cids, k=4),
        "E: kNN k=8":              lambda vr, sup: classify_knn(vr, sup, cids, k=8),
        "F: kNN k=6 eos_w=2":     lambda vr, sup: classify_knn(vr, sup, cids, k=6, eos_w=2.0),
        "G: kNN k=6 eos_w=3":     lambda vr, sup: classify_knn(vr, sup, cids, k=6, eos_w=3.0),
        "H: proto+morph(0.02)":    lambda vr, sup: classify_prototype_morph(vr, sup, cids, 0.02),
        "I: adaptive kNN k=3":     lambda vr, sup: classify_adaptive_knn(vr, sup, cids, k_per_class=3),
        "J: adaptive kNN k=5":     lambda vr, sup: classify_adaptive_knn(vr, sup, cids, k_per_class=5),
        "K: adaptive k=3 eos=2":   lambda vr, sup: classify_adaptive_knn(vr, sup, cids, k_per_class=3, eos_w=2.0),
        "L: adaptive k=5 eos=2":   lambda vr, sup: classify_adaptive_knn(vr, sup, cids, k_per_class=5, eos_w=2.0),
        "M: proto+adaptive":       lambda vr, sup: classify_prototype_weighted(vr, sup, cids, True),
        "N: proto+adaptive+morph":  None,
    }

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "eos_f1": []})

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        support_random = select_10shot(train_recs, seed)
        support_diverse = select_10shot_diverse(train_recs, seed)

        for name, fn in strategies.items():
            if name == "B: proto (diverse)":
                m = classify_prototype(val_recs, support_diverse, cids)
            elif name == "N: proto+adaptive+morph":
                m_base = classify_prototype_weighted(val_recs, support_random, cids, True)
                # Re-run with morph
                protos = {}
                sf = {}
                for c in cids:
                    feats = np.stack([s["feat"] for s in support_random[c]])
                    p = feats.mean(0)
                    protos[c] = p / np.linalg.norm(p)
                    sf[c] = feats
                gt, pred = [], []
                for r in val_recs:
                    scores_list = []
                    for c in cids:
                        proto_sim = float(r["feat"] @ protos[c])
                        support_sims = sf[c] @ r["feat"]
                        max_sim = float(np.max(support_sims))
                        scores_list.append(0.5 * proto_sim + 0.5 * max_sim)
                    scores = np.array(scores_list)
                    scores = apply_morphology_constraints(scores, r["morph"], cids, strength=0.02)
                    gt.append(r["gt"])
                    pred.append(cids[int(np.argmax(scores))])
                m = metrics(gt, pred, cids)
            else:
                m = fn(val_recs, support_random)

            eos = m["pc"].get(3, {}).get("f1", 0)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            all_results[name]["eos_f1"].append(eos)

    # Print summary
    print("\n" + "=" * 90)
    print("10-SHOT RESULTS (averaged over 5 seeds)")
    print("=" * 90)
    hdr = f"{'Strategy':<30} {'Acc':>8} {'mF1':>8} {'EosF1':>8}  {'Acc_std':>8} {'mF1_std':>8}"
    print(hdr)
    print("-" * 80)

    sorted_results = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, vals in sorted_results:
        acc_m = np.mean(vals["acc"])
        mf1_m = np.mean(vals["mf1"])
        eos_m = np.mean(vals["eos_f1"])
        acc_s = np.std(vals["acc"])
        mf1_s = np.std(vals["mf1"])
        print(f"{name:<30} {acc_m:>8.4f} {mf1_m:>8.4f} {eos_m:>8.4f}  {acc_s:>8.4f} {mf1_s:>8.4f}")

    # Best strategy detail
    best_name = sorted_results[0][0]
    best_vals = sorted_results[0][1]
    print(f"\nBEST: {best_name}")
    print(f"  Acc = {np.mean(best_vals['acc']):.4f} ± {np.std(best_vals['acc']):.4f}")
    print(f"  Macro F1 = {np.mean(best_vals['mf1']):.4f} ± {np.std(best_vals['mf1']):.4f}")

    out = {name: {"acc_mean": float(np.mean(v["acc"])), "mf1_mean": float(np.mean(v["mf1"])),
                   "eos_f1_mean": float(np.mean(v["eos_f1"])),
                   "acc_std": float(np.std(v["acc"])), "mf1_std": float(np.std(v["mf1"]))}
           for name, v in all_results.items()}
    with open(Path(__file__).parent / "ten_shot_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to experiments/ten_shot_results.json")


if __name__ == "__main__":
    main()
