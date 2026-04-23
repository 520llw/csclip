#!/usr/bin/env python3
"""
10-shot optimization v2: more aggressive strategies.

Key ideas:
1. Support-aware temperature: sharpen predictions for high-confidence queries
2. Per-class adaptive k: different classes need different k values
3. Prototype + max-sim ensemble
4. Augmented support via context variation (re-encode same cell with different margins)
5. Morphology-gated: use morph to break ties, not shift all scores
6. Multi-run prototype averaging (encode each support multiple times with jitter)
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
from labeling_tool.morphology_constraints import apply_morphology_constraints, compute_morphology_adjustments
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
    mx, my = int(bw*cell_margin), int(bh*cell_margin)
    crop = image[max(0,y1-my):min(h,y2+my), max(0,x1-mx):min(w,x2+mx)].copy()
    mc = inst.mask[max(0,y1-my):min(h,y2+my), max(0,x1-mx):min(w,x2+mx)]
    crop = np.where(mc[..., None], crop, np.full_like(crop, bg))
    cmx, cmy = int(bw*ctx_margin), int(bh*ctx_margin)
    ctx = image[max(0,y1-cmy):min(h,y2+cmy), max(0,x1-cmx):min(w,x2+cmx)].copy()
    ct = preprocess(Image.fromarray(crop)).unsqueeze(0).to(device)
    cxt = preprocess(Image.fromarray(ctx)).unsqueeze(0).to(device)
    with torch.no_grad():
        cf = model.encode_image(ct); cf /= cf.norm(dim=-1, keepdim=True)
        xf = model.encode_image(cxt); xf /= xf.norm(dim=-1, keepdim=True)
    fused = cw*cf + ctxw*xf
    fused /= fused.norm(dim=-1, keepdim=True)
    return fused.squeeze(0).cpu().numpy().astype(np.float32)


def encode_cell_augmented(model, preprocess, device, image, inst):
    """Encode same cell with 3 different margin/weight configs to get richer representation."""
    configs = [
        {"cell_margin": 0.10, "ctx_margin": 0.30, "bg": 128, "cw": 0.85, "ctxw": 0.15},
        {"cell_margin": 0.05, "ctx_margin": 0.20, "bg": 128, "cw": 0.95, "ctxw": 0.05},
        {"cell_margin": 0.15, "ctx_margin": 0.40, "bg": 128, "cw": 0.75, "ctxw": 0.25},
    ]
    feats = []
    for cfg in configs:
        f = encode_cell(model, preprocess, device, image, inst, **cfg)
        feats.append(f)
    mean_feat = np.mean(feats, axis=0)
    return mean_feat / np.linalg.norm(mean_feat)


def extract_all(items, model, preprocess, device, augmented=False):
    recs = []
    for item in items:
        img = np.array(Image.open(item["image_path"]).convert("RGB"))
        h, w = img.shape[:2]
        for i, ann in enumerate(item["annotations"]):
            inst = ann2inst(ann, h, w, i + 1)
            if inst is None:
                continue
            if augmented:
                feat = encode_cell_augmented(model, preprocess, device, img, inst)
            else:
                feat = encode_cell(model, preprocess, device, img, inst)
            morph = compute_morphology_features(image=img, instance=inst)
            recs.append({"gt": ann["class_id"], "feat": feat, "morph": morph,
                         "img": item["image_path"], "idx": i})
    return recs


def select_10shot(train_recs, seed):
    random.seed(seed)
    pc = defaultdict(list)
    for r in train_recs:
        pc[r["gt"]].append(r)
    return {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in sorted(CLASS_NAMES.keys())}


# ---- Classification strategies ----

def strategy_proto_maxsim(val_recs, support, cids, alpha=0.5):
    """Ensemble of prototype similarity and max support similarity."""
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
            max_sim = float(np.max(sf[c] @ r["feat"]))
            scores.append(alpha * proto_sim + (1 - alpha) * max_sim)
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def strategy_proto_topk_mean(val_recs, support, cids, top_k=3):
    """Mean of top-k support similarities per class."""
    sf = {c: np.stack([s["feat"] for s in support[c]]) for c in cids}
    gt, pred = [], []
    for r in val_recs:
        scores = []
        for c in cids:
            sims = sf[c] @ r["feat"]
            top = np.sort(sims)[::-1][:top_k]
            scores.append(float(top.mean()))
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def strategy_morph_tiebreak(val_recs, support, cids, alpha=0.5, morph_margin=0.005):
    """Use morphology only to break ties (when top-2 gap < margin)."""
    protos = {}
    sf = {}
    for c in cids:
        feats = np.stack([s["feat"] for s in support[c]])
        p = feats.mean(0); protos[c] = p / np.linalg.norm(p)
        sf[c] = feats

    gt, pred = [], []
    for r in val_recs:
        scores = []
        for c in cids:
            ps = float(r["feat"] @ protos[c])
            ms = float(np.max(sf[c] @ r["feat"]))
            scores.append(alpha * ps + (1-alpha) * ms)
        scores = np.array(scores)
        sorted_idx = np.argsort(-scores)
        gap = scores[sorted_idx[0]] - scores[sorted_idx[1]]
        if gap < morph_margin:
            morph_adj = compute_morphology_adjustments(r["morph"], cids)
            scores = scores + morph_adj * 0.5
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def strategy_class_adaptive_k(val_recs, support, cids):
    """Different k per class based on class size/variance."""
    sf = {c: np.stack([s["feat"] for s in support[c]]) for c in cids}
    class_k = {3: 5, 4: 4, 5: 3, 6: 4}

    gt, pred = [], []
    for r in val_recs:
        scores = {}
        for c in cids:
            sims = sf[c] @ r["feat"]
            k = min(class_k[c], len(sims))
            top = np.sort(sims)[::-1][:k]
            scores[c] = float(top.mean())
        gt.append(r["gt"])
        pred.append(max(scores, key=scores.get))
    return metrics(gt, pred, cids)


def strategy_weighted_ensemble(val_recs, support, cids, w_proto=0.3, w_max=0.3, w_topk=0.4, topk=3):
    """Weighted ensemble of proto, max-sim, top-k mean."""
    protos = {}
    sf = {}
    for c in cids:
        feats = np.stack([s["feat"] for s in support[c]])
        p = feats.mean(0); protos[c] = p / np.linalg.norm(p)
        sf[c] = feats

    gt, pred = [], []
    for r in val_recs:
        scores = []
        for c in cids:
            sims = sf[c] @ r["feat"]
            proto_s = float(r["feat"] @ protos[c])
            max_s = float(np.max(sims))
            topk_s = float(np.sort(sims)[::-1][:topk].mean())
            scores.append(w_proto * proto_s + w_max * max_s + w_topk * topk_s)
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def main():
    print("=" * 70)
    print("10-SHOT OPTIMIZATION v2")
    print("=" * 70)

    train_items = build_items("train")
    val_items = build_items("val")
    cids = sorted(CLASS_NAMES.keys())

    from labeling_tool.fewshot_biomedclip import _load_model_bundle
    bundle = _load_model_bundle("auto")
    model, preprocess, device = bundle["model"], bundle["preprocess"], bundle["device"]

    print("Extracting features (standard)...")
    train_recs = extract_all(train_items, model, preprocess, device, augmented=False)
    val_recs = extract_all(val_items, model, preprocess, device, augmented=False)

    print("Extracting features (augmented supports)...")
    train_recs_aug = extract_all(train_items, model, preprocess, device, augmented=True)
    val_recs_aug = extract_all(val_items, model, preprocess, device, augmented=True)
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    strategies = {}

    # Alpha sweep for proto+maxsim
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        strategies[f"proto+max a={alpha:.1f}"] = lambda vr, s, a=alpha: strategy_proto_maxsim(vr, s, cids, a)

    # Top-k mean sweep
    for k in [2, 3, 4, 5]:
        strategies[f"topk_mean k={k}"] = lambda vr, s, k=k: strategy_proto_topk_mean(vr, s, cids, k)

    # Morph tiebreak
    for margin in [0.002, 0.005, 0.01, 0.015, 0.02]:
        strategies[f"morph_tie m={margin:.3f}"] = lambda vr, s, m=margin: strategy_morph_tiebreak(vr, s, cids, 0.5, m)

    # Class-adaptive k
    strategies["class_adaptive_k"] = lambda vr, s: strategy_class_adaptive_k(vr, s, cids)

    # Weighted ensemble sweep
    for wp, wm, wt in [(0.2, 0.3, 0.5), (0.3, 0.3, 0.4), (0.3, 0.4, 0.3),
                        (0.2, 0.4, 0.4), (0.1, 0.4, 0.5), (0.4, 0.3, 0.3),
                        (0.2, 0.2, 0.6), (0.1, 0.3, 0.6), (0.0, 0.3, 0.7),
                        (0.0, 0.4, 0.6), (0.0, 0.5, 0.5)]:
        name = f"ensemble p={wp:.1f} m={wm:.1f} t={wt:.1f}"
        strategies[name] = lambda vr, s, a=wp, b=wm, c=wt: strategy_weighted_ensemble(vr, s, cids, a, b, c)

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "eos": []})

    # Also test augmented features
    aug_strategies = {
        "AUG proto+max a=0.5": lambda vr, s: strategy_proto_maxsim(vr, s, cids, 0.5),
        "AUG topk k=3": lambda vr, s: strategy_proto_topk_mean(vr, s, cids, 3),
        "AUG ensemble 0.2/0.4/0.4": lambda vr, s: strategy_weighted_ensemble(vr, s, cids, 0.2, 0.4, 0.4),
    }

    for seed in SEEDS:
        support = select_10shot(train_recs, seed)
        support_aug = select_10shot(train_recs_aug, seed)

        for name, fn in strategies.items():
            m = fn(val_recs, support)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            all_results[name]["eos"].append(m["pc"].get(3, {}).get("f1", 0))

        for name, fn in aug_strategies.items():
            m = fn(val_recs_aug, support_aug)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            all_results[name]["eos"].append(m["pc"].get(3, {}).get("f1", 0))

    print("\n" + "=" * 90)
    print("10-SHOT v2 RESULTS (5 seeds)")
    print("=" * 90)
    header = f"{'Strategy':<35} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Astd':>6} {'Fstd':>6}"
    print(header)
    print("-" * 75)

    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, v in sorted_r[:25]:
        print(f"{name:<35} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{np.mean(v['eos']):>7.4f} {np.std(v['acc']):>6.4f} {np.std(v['mf1']):>6.4f}")

    best = sorted_r[0]
    print(f"\nBEST: {best[0]}")
    print(f"  Acc = {np.mean(best[1]['acc']):.4f} ± {np.std(best[1]['acc']):.4f}")
    print(f"  mF1 = {np.mean(best[1]['mf1']):.4f} ± {np.std(best[1]['mf1']):.4f}")

    with open(Path(__file__).parent / "ten_shot_v2_results.json", "w") as f:
        json.dump({n: {"acc": float(np.mean(v["acc"])), "mf1": float(np.mean(v["mf1"])),
                        "eos": float(np.mean(v["eos"]))} for n, v in all_results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
