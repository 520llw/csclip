#!/usr/bin/env python3
"""
10-shot v4: Fundamental breakthrough attempt.

Core insight: BiomedCLIP features alone cannot separate Eos/Neu (cosine sim overlap).
New approach: Morphology-Visual DUAL-SPACE classifier.

Strategy:
1. Build a concatenated feature: [visual_feat (512d) || morph_feat (12d)] 
2. Normalize each space independently, then concatenate
3. Use nearest-centroid in this combined space
4. Class-specific distance metrics (different weight for vis/morph per class)
5. Adaptive confidence: high visual confidence -> trust visual; ambiguous -> lean on morphology
6. Eosinophil-focused: design features targeting Eos/Neu distinction
   - Color histogram in cytoplasm (Eos: red-orange, Neu: pale pink)
   - Nuclear lobe count estimation
   - Granule density/texture features
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
import json
import random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch

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


def compute_enhanced_morphology(image, inst):
    """Enhanced morphology features specifically targeting Eos/Neu distinction."""
    base_morph = compute_morphology_features(image=image, instance=inst)
    
    x1, y1, x2, y2 = inst.bbox
    cell_region = image[y1:y2, x1:x2].copy()
    mask_region = inst.mask[y1:y2, x1:x2]
    
    if cell_region.size == 0 or not mask_region.any():
        extra = np.zeros(18, dtype=np.float32)
        return np.concatenate([base_morph, extra])
    
    pixels = cell_region[mask_region]
    
    # Color features in HSV space (better for distinguishing Eos red-orange vs Neu pale pink)
    hsv_region = cv2.cvtColor(cell_region, cv2.COLOR_RGB2HSV)
    hsv_pixels = hsv_region[mask_region]
    
    h_mean = float(np.mean(hsv_pixels[:, 0])) / 180.0
    h_std = float(np.std(hsv_pixels[:, 0])) / 180.0
    s_mean = float(np.mean(hsv_pixels[:, 1])) / 255.0
    s_std = float(np.std(hsv_pixels[:, 1])) / 255.0
    v_mean = float(np.mean(hsv_pixels[:, 2])) / 255.0
    v_std = float(np.std(hsv_pixels[:, 2])) / 255.0
    
    # Red-orange ratio (Eos hallmark)
    r, g, b = pixels[:, 0].astype(float), pixels[:, 1].astype(float), pixels[:, 2].astype(float)
    red_dominance = float(np.mean(r / (g + 1e-6)))
    rg_ratio = float(np.mean((r - g) / (r + g + 1e-6)))
    rb_ratio = float(np.mean((r - b) / (r + b + 1e-6)))
    
    # Texture: local binary pattern-like features
    gray = cv2.cvtColor(cell_region, cv2.COLOR_RGB2GRAY)
    
    # Granularity: variance of Laplacian in cell region
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_masked = laplacian[mask_region]
    granule_intensity = float(np.var(lap_masked)) / 1000.0 if len(lap_masked) > 0 else 0.0
    granule_mean = float(np.mean(np.abs(lap_masked))) / 100.0 if len(lap_masked) > 0 else 0.0
    
    # Intensity histogram features
    hist = cv2.calcHist([gray], [0], mask_region.astype(np.uint8) * 255, [16], [0, 256])
    hist = hist.flatten() / (hist.sum() + 1e-6)
    hist_entropy = float(-np.sum(hist * np.log(hist + 1e-10)))
    hist_skewness = float(np.mean(((gray[mask_region].astype(float) - np.mean(gray[mask_region])) / (np.std(gray[mask_region]) + 1e-6)) ** 3)) if mask_region.any() else 0.0
    
    # Nuclear darkness ratio (darker region proportion)
    gray_masked = gray[mask_region]
    dark_threshold = np.percentile(gray_masked, 25)
    dark_ratio = float(np.sum(gray_masked < dark_threshold) / len(gray_masked))
    
    # Edge density (nuclear lobe boundary indicator)
    edges = cv2.Canny(gray, 50, 150)
    edge_masked = edges[mask_region]
    edge_density = float(np.sum(edge_masked > 0) / len(edge_masked)) if len(edge_masked) > 0 else 0.0
    
    # Compactness of dark regions (bilobed vs multilobed nucleus)
    dark_mask = (gray < dark_threshold) & mask_region
    dark_area = np.sum(dark_mask)
    if dark_area > 10:
        dark_u8 = dark_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(dark_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_dark_components = len([c for c in contours if cv2.contourArea(c) > 5])
    else:
        n_dark_components = 0
    
    extra = np.array([
        h_mean, h_std, s_mean, s_std, v_mean, v_std,
        red_dominance, rg_ratio, rb_ratio,
        granule_intensity, granule_mean,
        hist_entropy, hist_skewness,
        dark_ratio, edge_density,
        float(n_dark_components) / 5.0,
        float(dark_area) / float(np.sum(mask_region) + 1e-6),
        0.0,  # padding
    ], dtype=np.float32)
    
    return np.concatenate([base_morph, extra])


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
            morph = compute_enhanced_morphology(img, inst)
            recs.append({"gt": ann["class_id"], "feat": feat, "morph": morph})
    return recs


def select_10shot(train_recs, seed):
    random.seed(seed)
    pc = defaultdict(list)
    for r in train_recs:
        pc[r["gt"]].append(r)
    return {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in sorted(CLASS_NAMES.keys())}


# ========== STRATEGIES ==========

def strategy_dual_space(val_recs, support, cids, vis_w=0.7, morph_w=0.3, topk=5):
    """Dual-space kNN: combine visual and morphological distances."""
    sf = {}
    sm = {}
    morph_stats = {}
    for c in cids:
        sf[c] = np.stack([s["feat"] for s in support[c]])
        morphs = np.stack([s["morph"] for s in support[c]])
        mean_m = morphs.mean(0)
        std_m = morphs.std(0) + 1e-8
        sm[c] = (morphs - mean_m) / std_m
        morph_stats[c] = (mean_m, std_m)

    # Global morph normalization
    all_morphs = np.concatenate([np.stack([s["morph"] for s in support[c]]) for c in cids])
    global_mean = all_morphs.mean(0)
    global_std = all_morphs.std(0) + 1e-8

    gt, pred = [], []
    for r in val_recs:
        q_morph_norm = (r["morph"] - global_mean) / global_std
        scores = []
        for c in cids:
            vis_sims = sf[c] @ r["feat"]
            s_morph_norm = (np.stack([s["morph"] for s in support[c]]) - global_mean) / global_std
            morph_dists = np.array([np.linalg.norm(q_morph_norm - s_morph_norm[i]) for i in range(len(s_morph_norm))])
            morph_sims = 1.0 / (1.0 + morph_dists)
            combined = vis_w * vis_sims + morph_w * morph_sims
            top = np.sort(combined)[::-1][:topk]
            scores.append(float(top.mean()))
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def strategy_adaptive_fusion(val_recs, support, cids, topk=5):
    """Adaptive fusion: use visual confidence to decide vis/morph weight."""
    sf = {c: np.stack([s["feat"] for s in support[c]]) for c in cids}
    
    all_morphs = np.concatenate([np.stack([s["morph"] for s in support[c]]) for c in cids])
    global_mean = all_morphs.mean(0)
    global_std = all_morphs.std(0) + 1e-8
    sm_norm = {}
    for c in cids:
        sm_norm[c] = (np.stack([s["morph"] for s in support[c]]) - global_mean) / global_std

    gt, pred = [], []
    for r in val_recs:
        # Visual scores
        vis_scores = []
        for c in cids:
            sims = sf[c] @ r["feat"]
            vis_scores.append(float(np.sort(sims)[::-1][:topk].mean()))
        vis_scores = np.array(vis_scores)
        
        # Visual confidence: gap between top-1 and top-2
        sorted_vis = np.sort(vis_scores)[::-1]
        vis_conf = sorted_vis[0] - sorted_vis[1]
        
        # Morph scores
        q_morph = (r["morph"] - global_mean) / global_std
        morph_scores = []
        for c in cids:
            dists = np.array([np.linalg.norm(q_morph - sm_norm[c][i]) for i in range(len(sm_norm[c]))])
            morph_scores.append(float(np.sort(1.0/(1.0+dists))[::-1][:topk].mean()))
        morph_scores = np.array(morph_scores)
        
        # Adaptive weight: high visual confidence -> trust visual more
        if vis_conf > 0.02:
            alpha = 0.85
        elif vis_conf > 0.01:
            alpha = 0.65
        else:
            alpha = 0.45
        
        combined = alpha * vis_scores + (1 - alpha) * morph_scores
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(combined))])
    return metrics(gt, pred, cids)


def strategy_class_specific_fusion(val_recs, support, cids, topk=5):
    """Different vis/morph weights for different class pairs."""
    sf = {c: np.stack([s["feat"] for s in support[c]]) for c in cids}
    all_morphs = np.concatenate([np.stack([s["morph"] for s in support[c]]) for c in cids])
    g_mean = all_morphs.mean(0)
    g_std = all_morphs.std(0) + 1e-8
    sm = {c: (np.stack([s["morph"] for s in support[c]]) - g_mean) / g_std for c in cids}

    # For Eos(3) and Neu(4): use more morphology; for Lym(5) and Mac(6): use more visual
    class_vis_w = {3: 0.55, 4: 0.55, 5: 0.80, 6: 0.80}

    gt, pred = [], []
    for r in val_recs:
        q_m = (r["morph"] - g_mean) / g_std
        final_scores = []
        for c in cids:
            vis_sims = sf[c] @ r["feat"]
            vis_score = float(np.sort(vis_sims)[::-1][:topk].mean())
            morph_dists = np.array([np.linalg.norm(q_m - sm[c][i]) for i in range(len(sm[c]))])
            morph_score = float(np.sort(1.0/(1.0+morph_dists))[::-1][:topk].mean())
            w = class_vis_w[c]
            final_scores.append(w * vis_score + (1 - w) * morph_score)
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(final_scores))])
    return metrics(gt, pred, cids)


def strategy_morph_only_knn(val_recs, support, cids, topk=5):
    """Pure morphology kNN (baseline to measure morph discriminability)."""
    all_morphs = np.concatenate([np.stack([s["morph"] for s in support[c]]) for c in cids])
    g_mean = all_morphs.mean(0)
    g_std = all_morphs.std(0) + 1e-8
    sm = {c: (np.stack([s["morph"] for s in support[c]]) - g_mean) / g_std for c in cids}

    gt, pred = [], []
    for r in val_recs:
        q_m = (r["morph"] - g_mean) / g_std
        scores = []
        for c in cids:
            dists = np.array([np.linalg.norm(q_m - sm[c][i]) for i in range(len(sm[c]))])
            scores.append(float(np.sort(1.0/(1.0+dists))[::-1][:topk].mean()))
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def strategy_cascade(val_recs, support, cids, topk=5, conf_threshold=0.015):
    """Two-stage cascade: visual first, morph refinement for confused cases."""
    sf = {c: np.stack([s["feat"] for s in support[c]]) for c in cids}
    all_morphs = np.concatenate([np.stack([s["morph"] for s in support[c]]) for c in cids])
    g_mean = all_morphs.mean(0)
    g_std = all_morphs.std(0) + 1e-8
    sm = {c: (np.stack([s["morph"] for s in support[c]]) - g_mean) / g_std for c in cids}

    gt, pred = [], []
    morph_used = 0
    for r in val_recs:
        vis_scores = []
        for c in cids:
            sims = sf[c] @ r["feat"]
            vis_scores.append(float(np.sort(sims)[::-1][:topk].mean()))
        vis_scores = np.array(vis_scores)
        
        sorted_idx = np.argsort(-vis_scores)
        gap = vis_scores[sorted_idx[0]] - vis_scores[sorted_idx[1]]
        
        if gap < conf_threshold:
            # Ambiguous: use morph to decide between top-2
            q_m = (r["morph"] - g_mean) / g_std
            top2 = sorted_idx[:2]
            morph_scores = []
            for idx in top2:
                c = cids[idx]
                dists = np.array([np.linalg.norm(q_m - sm[c][i]) for i in range(len(sm[c]))])
                morph_scores.append(float(np.sort(1.0/(1.0+dists))[::-1][:topk].mean()))
            chosen = top2[int(np.argmax(morph_scores))]
            pred.append(cids[chosen])
            morph_used += 1
        else:
            pred.append(cids[sorted_idx[0]])
        gt.append(r["gt"])
    
    m = metrics(gt, pred, cids)
    return m


def strategy_concat_space(val_recs, support, cids, vis_dim_w=1.0, morph_dim_w=2.0, topk=5):
    """Concatenated feature space: [scaled_visual || scaled_morph]."""
    all_morphs = np.concatenate([np.stack([s["morph"] for s in support[c]]) for c in cids])
    g_mean = all_morphs.mean(0)
    g_std = all_morphs.std(0) + 1e-8

    def make_concat(feat, morph):
        m_norm = (morph - g_mean) / g_std
        return np.concatenate([feat * vis_dim_w, m_norm * morph_dim_w])

    sc = {c: np.stack([make_concat(s["feat"], s["morph"]) for s in support[c]]) for c in cids}

    gt, pred = [], []
    for r in val_recs:
        q = make_concat(r["feat"], r["morph"])
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        scores = []
        for c in cids:
            s_norms = sc[c] / (np.linalg.norm(sc[c], axis=1, keepdims=True) + 1e-8)
            sims = s_norms @ q_norm
            top = np.sort(sims)[::-1][:topk]
            scores.append(float(top.mean()))
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def main():
    print("=" * 70)
    print("10-SHOT v4: DUAL-SPACE (Enhanced Morphology + Visual)")
    print("=" * 70)

    train_items = build_items("train")
    val_items = build_items("val")
    cids = sorted(CLASS_NAMES.keys())

    from labeling_tool.fewshot_biomedclip import _load_model_bundle
    bundle = _load_model_bundle("auto")
    model, preprocess, device = bundle["model"], bundle["preprocess"], bundle["device"]

    print("Extracting features with enhanced morphology...")
    train_recs = extract_all(train_items, model, preprocess, device)
    val_recs = extract_all(val_items, model, preprocess, device)
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")
    print(f"  Morph dim: {train_recs[0]['morph'].shape[0]}")

    # Quick morph analysis
    for c in cids:
        morphs = [r["morph"] for r in train_recs if r["gt"] == c]
        morphs = np.stack(morphs[:50])
        print(f"  Class {c} ({CLASS_NAMES[c]}): morph mean = [{', '.join(f'{x:.3f}' for x in morphs.mean(0)[:6])} ...]")

    strategies = {}

    # Morph-only baseline
    strategies["morph_only k=3"] = lambda vr, s: strategy_morph_only_knn(vr, s, cids, 3)
    strategies["morph_only k=5"] = lambda vr, s: strategy_morph_only_knn(vr, s, cids, 5)

    # Dual-space with different weights
    for vw in [0.5, 0.6, 0.7, 0.8, 0.9]:
        mw = round(1.0 - vw, 1)
        for k in [3, 5]:
            strategies[f"dual v={vw:.1f} m={mw:.1f} k={k}"] = lambda vr, s, v=vw, m=mw, kk=k: strategy_dual_space(vr, s, cids, v, m, kk)

    # Adaptive fusion
    strategies["adaptive_fusion k=3"] = lambda vr, s: strategy_adaptive_fusion(vr, s, cids, 3)
    strategies["adaptive_fusion k=5"] = lambda vr, s: strategy_adaptive_fusion(vr, s, cids, 5)

    # Class-specific fusion
    strategies["class_specific k=3"] = lambda vr, s: strategy_class_specific_fusion(vr, s, cids, 3)
    strategies["class_specific k=5"] = lambda vr, s: strategy_class_specific_fusion(vr, s, cids, 5)

    # Cascade
    for thr in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
        strategies[f"cascade thr={thr:.3f}"] = lambda vr, s, t=thr: strategy_cascade(vr, s, cids, 5, t)

    # Concatenated feature space
    for vdw in [0.5, 1.0, 1.5]:
        for mdw in [1.0, 2.0, 3.0, 5.0]:
            strategies[f"concat v={vdw:.1f} m={mdw:.1f}"] = lambda vr, s, v=vdw, m=mdw: strategy_concat_space(vr, s, cids, v, m)

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "eos": [], "per_class": defaultdict(list)})

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        support = select_10shot(train_recs, seed)
        for name, fn in strategies.items():
            try:
                m = fn(val_recs, support)
                all_results[name]["acc"].append(m["acc"])
                all_results[name]["mf1"].append(m["mf1"])
                all_results[name]["eos"].append(m["pc"].get(3, {}).get("f1", 0))
                for c in cids:
                    all_results[name]["per_class"][c].append(m["pc"][c]["f1"])
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

    print("\n" + "=" * 100)
    print("10-SHOT v4 RESULTS (5 seeds)")
    print("=" * 100)
    header = f"{'Strategy':<30} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 95)

    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, v in sorted_r[:35]:
        pc_str = " ".join(f"{np.mean(v['per_class'][c]):>7.4f}" for c in cids)
        print(f"{name:<30} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")

    best = sorted_r[0]
    print(f"\nBEST: {best[0]}")
    print(f"  Acc = {np.mean(best[1]['acc']):.4f} ± {np.std(best[1]['acc']):.4f}")
    print(f"  mF1 = {np.mean(best[1]['mf1']):.4f} ± {np.std(best[1]['mf1']):.4f}")
    for c in cids:
        print(f"  {CLASS_NAMES[c]}: F1 = {np.mean(best[1]['per_class'][c]):.4f}")

    with open(Path(__file__).parent / "ten_shot_v4_results.json", "w") as f:
        json.dump({n: {"acc": float(np.mean(v["acc"])), "mf1": float(np.mean(v["mf1"])),
                        "eos": float(np.mean(v["eos"]))} for n, v in all_results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
