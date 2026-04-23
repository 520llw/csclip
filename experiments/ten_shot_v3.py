#!/usr/bin/env python3
"""
10-shot v3: Deep optimization for few-shot BALF cell classification.

Key breakthroughs to attempt:
1. Support set augmentation via feature-space interpolation (mixup in embedding space)
2. Text-guided prototype refinement (use text embeddings to calibrate visual prototypes)
3. Transductive inference (use unlabeled query distribution to refine predictions)
4. Class-balanced temperature scaling
5. Morphology-weighted prototype (weight support samples by morphology typicality)
6. Feature concatenation: visual + morphological feature fusion via learned projection
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biomedclip_zeroshot_cell_classify import InstanceInfo
from biomedclip_query_adaptive_classifier import compute_morphology_features
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
            recs.append({"gt": ann["class_id"], "feat": feat, "morph": morph})
    return recs


def get_text_prototypes(model, tokenizer, device):
    """Build text prototypes for each cell class using BiomedCLIP text encoder."""
    prompts = {
        3: [
            "a photomicrograph of an eosinophil cell with bilobed nucleus and red-orange granules",
            "eosinophil granulocyte in bronchoalveolar lavage fluid",
            "cell with bright pink cytoplasmic granules and bilobed nucleus",
        ],
        4: [
            "a photomicrograph of a neutrophil cell with multilobed nucleus",
            "neutrophil granulocyte in bronchoalveolar lavage fluid",
            "cell with pale pink granules and segmented nucleus with 3-5 lobes",
        ],
        5: [
            "a photomicrograph of a lymphocyte cell with large round nucleus",
            "lymphocyte in bronchoalveolar lavage fluid, small round cell",
            "small cell with thin rim of blue cytoplasm around dark round nucleus",
        ],
        6: [
            "a photomicrograph of a macrophage cell in tissue sample",
            "alveolar macrophage in bronchoalveolar lavage fluid",
            "large cell with abundant cytoplasm and kidney-shaped nucleus",
        ],
    }
    text_protos = {}
    for cid, texts in prompts.items():
        tokens = tokenizer(texts, context_length=256)
        tokens = tokens.to(device)
        with torch.no_grad():
            te = model.encode_text(tokens)
            te /= te.norm(dim=-1, keepdim=True)
        text_protos[cid] = te.mean(0).cpu().numpy().astype(np.float32)
        text_protos[cid] /= np.linalg.norm(text_protos[cid])
    return text_protos


def select_10shot(train_recs, seed):
    random.seed(seed)
    pc = defaultdict(list)
    for r in train_recs:
        pc[r["gt"]].append(r)
    return {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in sorted(CLASS_NAMES.keys())}


# ========== ADVANCED STRATEGIES ==========

def strategy_text_calibrated_proto(val_recs, support, cids, text_protos, text_w=0.15):
    """Calibrate visual prototypes with text prototypes."""
    protos = {}
    for c in cids:
        feats = np.stack([s["feat"] for s in support[c]])
        vis_p = feats.mean(0)
        vis_p /= np.linalg.norm(vis_p)
        combined = (1 - text_w) * vis_p + text_w * text_protos[c]
        protos[c] = combined / np.linalg.norm(combined)

    gt, pred = [], []
    for r in val_recs:
        scores = [float(r["feat"] @ protos[c]) for c in cids]
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def strategy_support_augment_mixup(val_recs, support, cids, n_aug=5, alpha=0.3):
    """Augment support set via feature mixup within same class."""
    aug_support = {}
    for c in cids:
        orig_feats = [s["feat"] for s in support[c]]
        aug_feats = list(orig_feats)
        for _ in range(n_aug):
            i, j = random.sample(range(len(orig_feats)), 2)
            lam = np.random.beta(alpha, alpha) if alpha > 0 else 0.5
            mixed = lam * orig_feats[i] + (1 - lam) * orig_feats[j]
            mixed = mixed / np.linalg.norm(mixed)
            aug_feats.append(mixed)
        aug_support[c] = [{"feat": f} for f in aug_feats]
    return _classify_proto_topk(val_recs, aug_support, cids, topk=5)


def _classify_proto_topk(val_recs, support, cids, topk=5):
    sf = {c: np.stack([s["feat"] for s in support[c]]) for c in cids}
    gt, pred = [], []
    for r in val_recs:
        scores = []
        for c in cids:
            sims = sf[c] @ r["feat"]
            k = min(topk, len(sims))
            top = np.sort(sims)[::-1][:k]
            scores.append(float(top.mean()))
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def strategy_transductive(val_recs, support, cids, n_iter=3, alpha=0.7):
    """Transductive inference: refine prototypes using high-confidence query predictions."""
    protos = {}
    sf = {}
    for c in cids:
        feats = np.stack([s["feat"] for s in support[c]])
        p = feats.mean(0)
        protos[c] = p / np.linalg.norm(p)
        sf[c] = feats

    query_feats = np.stack([r["feat"] for r in val_recs])

    for iteration in range(n_iter):
        all_scores = np.zeros((len(val_recs), len(cids)))
        for j, c in enumerate(cids):
            sim_proto = query_feats @ protos[c]
            sim_max = np.max(sf[c] @ query_feats.T, axis=0)
            all_scores[:, j] = 0.5 * sim_proto + 0.5 * sim_max

        probs = softmax(all_scores * 20, axis=1)

        # Refine prototypes with high-confidence queries
        for j, c in enumerate(cids):
            conf = probs[:, j]
            mask = conf > 0.85
            if np.sum(mask) < 3:
                continue
            hc_feats = query_feats[mask]
            hc_weights = conf[mask]
            hc_proto = np.average(hc_feats, axis=0, weights=hc_weights)
            hc_proto /= np.linalg.norm(hc_proto)
            protos[c] = alpha * protos[c] + (1 - alpha) * hc_proto
            protos[c] /= np.linalg.norm(protos[c])

    gt, pred = [], []
    for i, r in enumerate(val_recs):
        scores = []
        for j, c in enumerate(cids):
            sim_proto = float(r["feat"] @ protos[c])
            sim_max = float(np.max(sf[c] @ r["feat"]))
            scores.append(0.5 * sim_proto + 0.5 * sim_max)
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def strategy_class_temp_scaling(val_recs, support, cids, temps=None):
    """Per-class temperature scaling to balance class predictions."""
    if temps is None:
        temps = {3: 15.0, 4: 20.0, 5: 25.0, 6: 20.0}

    protos = {}
    sf = {}
    for c in cids:
        feats = np.stack([s["feat"] for s in support[c]])
        p = feats.mean(0)
        protos[c] = p / np.linalg.norm(p)
        sf[c] = feats

    gt, pred = [], []
    for r in val_recs:
        raw_scores = []
        for c in cids:
            sim_proto = float(r["feat"] @ protos[c])
            sim_max = float(np.max(sf[c] @ r["feat"]))
            raw_scores.append(0.5 * sim_proto + 0.5 * sim_max)

        scaled = [raw_scores[j] * temps[cids[j]] for j in range(len(cids))]
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scaled))])
    return metrics(gt, pred, cids)


def strategy_morph_enhanced_knn(val_recs, support, cids, topk=5, morph_w=0.1):
    """kNN with morphological feature similarity as additional signal."""
    sf = {}
    sm = {}
    for c in cids:
        sf[c] = np.stack([s["feat"] for s in support[c]])
        morphs = np.stack([s["morph"] for s in support[c]])
        # Normalize morph features per dimension
        morph_mean = morphs.mean(0)
        morph_std = morphs.std(0) + 1e-8
        sm[c] = (morphs - morph_mean) / morph_std
        sm[c + 100] = (morph_mean, morph_std)

    gt, pred = [], []
    for r in val_recs:
        scores = []
        for c in cids:
            vis_sims = sf[c] @ r["feat"]
            morph_norm = (r["morph"] - sm[c + 100][0]) / sm[c + 100][1]
            morph_sims = np.array([1.0 / (1.0 + np.linalg.norm(morph_norm - sm[c][i])) for i in range(len(sm[c]))])
            combined = (1 - morph_w) * vis_sims + morph_w * morph_sims
            top = np.sort(combined)[::-1][:topk]
            scores.append(float(top.mean()))
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def strategy_proto_rectification(val_recs, support, cids):
    """Prototype rectification: shift prototypes towards query distribution center."""
    protos = {}
    sf = {}
    for c in cids:
        feats = np.stack([s["feat"] for s in support[c]])
        p = feats.mean(0)
        protos[c] = p / np.linalg.norm(p)
        sf[c] = feats

    query_feats = np.stack([r["feat"] for r in val_recs])
    query_mean = query_feats.mean(0)
    query_mean /= np.linalg.norm(query_mean)

    # Rectify each prototype: shift towards query distribution
    rect_protos = {}
    for c in cids:
        support_mean = sf[c].mean(0)
        support_mean /= np.linalg.norm(support_mean)
        shift = query_mean - support_mean
        rect = protos[c] + 0.3 * shift
        rect_protos[c] = rect / np.linalg.norm(rect)

    gt, pred = [], []
    for r in val_recs:
        scores = []
        for c in cids:
            sim_proto = float(r["feat"] @ rect_protos[c])
            sim_max = float(np.max(sf[c] @ r["feat"]))
            scores.append(0.5 * sim_proto + 0.5 * sim_max)
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def strategy_distance_weighted_proto(val_recs, support, cids, gamma=2.0):
    """Weight each support sample inversely to its distance to class center (focus on typical samples)."""
    protos = {}
    sf = {}
    for c in cids:
        feats = np.stack([s["feat"] for s in support[c]])
        raw_center = feats.mean(0)
        raw_center /= np.linalg.norm(raw_center)
        
        # Weight by similarity to center (more typical = higher weight)
        sims = feats @ raw_center
        weights = np.exp(gamma * sims)
        weights /= weights.sum()
        p = np.average(feats, axis=0, weights=weights)
        protos[c] = p / np.linalg.norm(p)
        sf[c] = feats

    gt, pred = [], []
    for r in val_recs:
        scores = []
        for c in cids:
            sim_proto = float(r["feat"] @ protos[c])
            sim_max = float(np.max(sf[c] @ r["feat"]))
            scores.append(0.5 * sim_proto + 0.5 * sim_max)
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def strategy_combined_best(val_recs, support, cids, text_protos,
                            text_w=0.1, alpha_trans=0.7, n_trans_iter=2):
    """Combine best ideas: text-calibrated proto + transductive + morph tiebreak."""
    # 1. Build text-calibrated prototypes
    protos = {}
    sf = {}
    for c in cids:
        feats = np.stack([s["feat"] for s in support[c]])
        vis_p = feats.mean(0)
        vis_p /= np.linalg.norm(vis_p)
        protos[c] = (1 - text_w) * vis_p + text_w * text_protos[c]
        protos[c] /= np.linalg.norm(protos[c])
        sf[c] = feats

    query_feats = np.stack([r["feat"] for r in val_recs])

    # 2. Transductive refinement
    for _ in range(n_trans_iter):
        all_scores = np.zeros((len(val_recs), len(cids)))
        for j, c in enumerate(cids):
            all_scores[:, j] = 0.5 * (query_feats @ protos[c]) + 0.5 * np.max(sf[c] @ query_feats.T, axis=0)
        probs = softmax(all_scores * 20, axis=1)
        for j, c in enumerate(cids):
            conf = probs[:, j]
            mask = conf > 0.85
            if np.sum(mask) < 3:
                continue
            hc_feats = query_feats[mask]
            hc_proto = np.average(hc_feats, axis=0, weights=conf[mask])
            hc_proto /= np.linalg.norm(hc_proto)
            protos[c] = alpha_trans * protos[c] + (1 - alpha_trans) * hc_proto
            protos[c] /= np.linalg.norm(protos[c])

    # 3. Final prediction with morph tiebreak
    from labeling_tool.morphology_constraints import compute_morphology_adjustments
    gt, pred = [], []
    for i, r in enumerate(val_recs):
        scores = []
        for c in cids:
            sim_proto = float(r["feat"] @ protos[c])
            sim_max = float(np.max(sf[c] @ r["feat"]))
            scores.append(0.5 * sim_proto + 0.5 * sim_max)
        scores = np.array(scores)
        sorted_idx = np.argsort(-scores)
        gap = scores[sorted_idx[0]] - scores[sorted_idx[1]]
        if gap < 0.01:
            morph_adj = compute_morphology_adjustments(r["morph"], cids)
            scores = scores + morph_adj * 0.5
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def main():
    print("=" * 70)
    print("10-SHOT OPTIMIZATION v3 (deep strategies)")
    print("=" * 70)

    train_items = build_items("train")
    val_items = build_items("val")
    cids = sorted(CLASS_NAMES.keys())

    from labeling_tool.fewshot_biomedclip import _load_model_bundle
    bundle = _load_model_bundle("auto")
    model, preprocess, device = bundle["model"], bundle["preprocess"], bundle["device"]

    from open_clip import get_tokenizer
    local_dir = "/home/xut/csclip/labeling_tool/weights/biomedclip"
    tokenizer = get_tokenizer(f"local-dir:{local_dir}")

    print("Building text prototypes...")
    text_protos = get_text_prototypes(model, tokenizer, device)

    print("Extracting features...")
    train_recs = extract_all(train_items, model, preprocess, device)
    val_recs = extract_all(val_items, model, preprocess, device)
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    strategies = {}

    # Text-calibrated prototypes with different weights
    for tw in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        strategies[f"text_calib tw={tw:.2f}"] = lambda vr, s, w=tw: strategy_text_calibrated_proto(vr, s, cids, text_protos, w)

    # Mixup augmentation
    for n_aug in [5, 10, 20]:
        strategies[f"mixup n={n_aug}"] = lambda vr, s, n=n_aug: strategy_support_augment_mixup(vr, s, cids, n)

    # Transductive inference
    for n_iter in [1, 2, 3, 5]:
        for alpha in [0.5, 0.7, 0.9]:
            strategies[f"transduct i={n_iter} a={alpha:.1f}"] = lambda vr, s, ni=n_iter, a=alpha: strategy_transductive(vr, s, cids, ni, a)

    # Temperature scaling
    temp_configs = [
        {3: 15, 4: 20, 5: 25, 6: 20},
        {3: 20, 4: 20, 5: 20, 6: 20},
        {3: 25, 4: 20, 5: 20, 6: 20},
        {3: 30, 4: 20, 5: 20, 6: 20},
        {3: 20, 4: 20, 5: 30, 6: 20},
        {3: 25, 4: 15, 5: 20, 6: 15},
    ]
    for i, tc in enumerate(temp_configs):
        strategies[f"temp_scale #{i}"] = lambda vr, s, t=tc: strategy_class_temp_scaling(vr, s, cids, t)

    # Morph-enhanced kNN
    for mw in [0.05, 0.10, 0.15, 0.20]:
        for tk in [3, 5]:
            strategies[f"morph_knn mw={mw:.2f} k={tk}"] = lambda vr, s, w=mw, k=tk: strategy_morph_enhanced_knn(vr, s, cids, k, w)

    # Prototype rectification
    strategies["proto_rect"] = lambda vr, s: strategy_proto_rectification(vr, s, cids)

    # Distance-weighted prototype
    for g in [1.0, 2.0, 3.0, 5.0]:
        strategies[f"dw_proto g={g:.0f}"] = lambda vr, s, gamma=g: strategy_distance_weighted_proto(vr, s, cids, gamma)

    # Combined best
    for tw in [0.05, 0.10, 0.15]:
        strategies[f"combined tw={tw:.2f}"] = lambda vr, s, w=tw: strategy_combined_best(vr, s, cids, text_protos, w)

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "eos": []})

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        support = select_10shot(train_recs, seed)
        for name, fn in strategies.items():
            try:
                m = fn(val_recs, support)
                all_results[name]["acc"].append(m["acc"])
                all_results[name]["mf1"].append(m["mf1"])
                all_results[name]["eos"].append(m["pc"].get(3, {}).get("f1", 0))
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

    print("\n" + "=" * 90)
    print("10-SHOT v3 RESULTS (5 seeds)")
    print("=" * 90)
    header = f"{'Strategy':<35} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Astd':>6} {'Fstd':>6}"
    print(header)
    print("-" * 75)

    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, v in sorted_r[:30]:
        print(f"{name:<35} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{np.mean(v['eos']):>7.4f} {np.std(v['acc']):>6.4f} {np.std(v['mf1']):>6.4f}")

    best = sorted_r[0]
    print(f"\nBEST: {best[0]}")
    print(f"  Acc = {np.mean(best[1]['acc']):.4f} ± {np.std(best[1]['acc']):.4f}")
    print(f"  mF1 = {np.mean(best[1]['mf1']):.4f} ± {np.std(best[1]['mf1']):.4f}")

    with open(Path(__file__).parent / "ten_shot_v3_results.json", "w") as f:
        json.dump({n: {"acc": float(np.mean(v["acc"])), "mf1": float(np.mean(v["mf1"])),
                        "eos": float(np.mean(v["eos"]))} for n, v in all_results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
