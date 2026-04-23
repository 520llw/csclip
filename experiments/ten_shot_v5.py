#!/usr/bin/env python3
"""
10-shot v5: Refine dual-space classifier.

Focus on:
1. Fine-grained Eos/Neu separation via targeted color features
2. Adaptive morph weight for Eos-like queries (increase morph weight when visual is confused between Eos/Neu)
3. Two-stage: visual kNN to get top-2, then morph disambiguator for Eos/Neu confusion
4. Sweep dual-space hyperparameters more finely around the sweet spot (v=0.5-0.7, m=0.3-0.5)
5. Try weighted morph dimensions (give more weight to color features)
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
    base_morph = compute_morphology_features(image=image, instance=inst)
    x1, y1, x2, y2 = inst.bbox
    cell_region = image[y1:y2, x1:x2].copy()
    mask_region = inst.mask[y1:y2, x1:x2]
    
    if cell_region.size == 0 or not mask_region.any():
        return np.concatenate([base_morph, np.zeros(18, dtype=np.float32)])
    
    pixels = cell_region[mask_region]
    hsv_region = cv2.cvtColor(cell_region, cv2.COLOR_RGB2HSV)
    hsv_pixels = hsv_region[mask_region]
    
    h_mean = float(np.mean(hsv_pixels[:, 0])) / 180.0
    h_std = float(np.std(hsv_pixels[:, 0])) / 180.0
    s_mean = float(np.mean(hsv_pixels[:, 1])) / 255.0
    s_std = float(np.std(hsv_pixels[:, 1])) / 255.0
    v_mean = float(np.mean(hsv_pixels[:, 2])) / 255.0
    v_std = float(np.std(hsv_pixels[:, 2])) / 255.0
    
    r, g, b = pixels[:, 0].astype(float), pixels[:, 1].astype(float), pixels[:, 2].astype(float)
    red_dominance = float(np.mean(r / (g + 1e-6)))
    rg_ratio = float(np.mean((r - g) / (r + g + 1e-6)))
    rb_ratio = float(np.mean((r - b) / (r + b + 1e-6)))
    
    gray = cv2.cvtColor(cell_region, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_masked = laplacian[mask_region]
    granule_intensity = float(np.var(lap_masked)) / 1000.0 if len(lap_masked) > 0 else 0.0
    granule_mean = float(np.mean(np.abs(lap_masked))) / 100.0 if len(lap_masked) > 0 else 0.0
    
    hist = cv2.calcHist([gray], [0], mask_region.astype(np.uint8)*255, [16], [0, 256])
    hist = hist.flatten() / (hist.sum() + 1e-6)
    hist_entropy = float(-np.sum(hist * np.log(hist + 1e-10)))
    gray_masked = gray[mask_region]
    mean_g = float(np.mean(gray_masked))
    std_g = float(np.std(gray_masked)) + 1e-6
    hist_skewness = float(np.mean(((gray_masked.astype(float) - mean_g) / std_g) ** 3))
    
    dark_threshold = np.percentile(gray_masked, 25)
    dark_ratio = float(np.sum(gray_masked < dark_threshold) / len(gray_masked))
    
    edges = cv2.Canny(gray, 50, 150)
    edge_masked = edges[mask_region]
    edge_density = float(np.sum(edge_masked > 0) / len(edge_masked)) if len(edge_masked) > 0 else 0.0
    
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
        0.0
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


def dual_space(val_recs, support, cids, vis_w, morph_w, topk,
               morph_dim_weights=None):
    """Dual-space kNN with optional per-dimension morph weighting."""
    all_morphs = np.concatenate([np.stack([s["morph"] for s in support[c]]) for c in cids])
    g_mean = all_morphs.mean(0)
    g_std = all_morphs.std(0) + 1e-8
    
    if morph_dim_weights is not None:
        dim_w = np.array(morph_dim_weights, dtype=np.float32)
    else:
        dim_w = np.ones(len(g_mean), dtype=np.float32)

    sf = {c: np.stack([s["feat"] for s in support[c]]) for c in cids}
    sm = {}
    for c in cids:
        m = (np.stack([s["morph"] for s in support[c]]) - g_mean) / g_std
        sm[c] = m * dim_w

    gt, pred = [], []
    for r in val_recs:
        q_m = ((r["morph"] - g_mean) / g_std) * dim_w
        scores = []
        for c in cids:
            vis_sims = sf[c] @ r["feat"]
            morph_dists = np.array([np.linalg.norm(q_m - sm[c][i]) for i in range(len(sm[c]))])
            morph_sims = 1.0 / (1.0 + morph_dists)
            combined = vis_w * vis_sims + morph_w * morph_sims
            top = np.sort(combined)[::-1][:topk]
            scores.append(float(top.mean()))
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def eos_neu_cascade(val_recs, support, cids, vis_topk=5, morph_topk=5, conf_thr=0.015):
    """Cascade: visual kNN first, then morph disambiguator specifically for Eos/Neu confusion."""
    all_morphs = np.concatenate([np.stack([s["morph"] for s in support[c]]) for c in cids])
    g_mean = all_morphs.mean(0)
    g_std = all_morphs.std(0) + 1e-8

    sf = {c: np.stack([s["feat"] for s in support[c]]) for c in cids}
    sm = {c: (np.stack([s["morph"] for s in support[c]]) - g_mean) / g_std for c in cids}

    gt, pred = [], []
    cascade_count = 0
    for r in val_recs:
        vis_scores = {}
        for c in cids:
            sims = sf[c] @ r["feat"]
            vis_scores[c] = float(np.sort(sims)[::-1][:vis_topk].mean())
        
        sorted_c = sorted(cids, key=lambda c: -vis_scores[c])
        top1, top2 = sorted_c[0], sorted_c[1]
        gap = vis_scores[top1] - vis_scores[top2]
        
        # Only cascade for Eos/Neu confusion
        eos_neu = {3, 4}
        if gap < conf_thr and {top1, top2} == eos_neu:
            q_m = (r["morph"] - g_mean) / g_std
            morph_s = {}
            for c in [3, 4]:
                dists = np.array([np.linalg.norm(q_m - sm[c][i]) for i in range(len(sm[c]))])
                morph_s[c] = float(np.sort(1.0/(1.0+dists))[::-1][:morph_topk].mean())
            pred.append(max(morph_s, key=morph_s.get))
            cascade_count += 1
        else:
            pred.append(top1)
        gt.append(r["gt"])
    return metrics(gt, pred, cids)


def dual_eos_boost(val_recs, support, cids, vis_w=0.6, morph_w=0.4, topk=5,
                    eos_morph_boost=1.5):
    """Dual-space with boosted morph weight specifically for Eosinophil class."""
    all_morphs = np.concatenate([np.stack([s["morph"] for s in support[c]]) for c in cids])
    g_mean = all_morphs.mean(0)
    g_std = all_morphs.std(0) + 1e-8

    sf = {c: np.stack([s["feat"] for s in support[c]]) for c in cids}
    sm = {c: (np.stack([s["morph"] for s in support[c]]) - g_mean) / g_std for c in cids}

    gt, pred = [], []
    for r in val_recs:
        q_m = (r["morph"] - g_mean) / g_std
        scores = []
        for c in cids:
            vis_sims = sf[c] @ r["feat"]
            morph_dists = np.array([np.linalg.norm(q_m - sm[c][i]) for i in range(len(sm[c]))])
            morph_sims = 1.0 / (1.0 + morph_dists)
            mw = morph_w * (eos_morph_boost if c == 3 else 1.0)
            vw = vis_w
            total_w = vw + mw
            combined = (vw/total_w) * vis_sims + (mw/total_w) * morph_sims
            top = np.sort(combined)[::-1][:topk]
            scores.append(float(top.mean()))
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def main():
    print("=" * 70)
    print("10-SHOT v5: Refined dual-space + Eos/Neu disambiguation")
    print("=" * 70)

    train_items = build_items("train")
    val_items = build_items("val")
    cids = sorted(CLASS_NAMES.keys())

    from labeling_tool.fewshot_biomedclip import _load_model_bundle
    bundle = _load_model_bundle("auto")
    model, preprocess, device = bundle["model"], bundle["preprocess"], bundle["device"]

    print("Extracting features...")
    train_recs = extract_all(train_items, model, preprocess, device)
    val_recs = extract_all(val_items, model, preprocess, device)
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    # Analyze Eos vs Neu morph separation
    print("\n--- Eos vs Neu morph analysis ---")
    eos_m = np.stack([r["morph"] for r in train_recs if r["gt"] == 3][:100])
    neu_m = np.stack([r["morph"] for r in train_recs if r["gt"] == 4][:100])
    lym_m = np.stack([r["morph"] for r in train_recs if r["gt"] == 5][:100])
    mac_m = np.stack([r["morph"] for r in train_recs if r["gt"] == 6][:100])
    
    dim_names = [
        "log_area", "log_perim", "circ", "asp_ratio", "solidity", 
        "mean_R", "mean_G", "mean_B", "std_int", "eccentr", "extent", "equiv_d",
        "h_mean", "h_std", "s_mean", "s_std", "v_mean", "v_std",
        "red_dom", "rg_ratio", "rb_ratio",
        "gran_int", "gran_mean", "hist_ent", "hist_skew",
        "dark_rat", "edge_dens", "n_dark", "dark_area", "pad"
    ]
    
    separability = []
    for d in range(min(len(dim_names), eos_m.shape[1])):
        em, nm = eos_m[:, d], neu_m[:, d]
        pooled_std = np.sqrt((np.var(em) + np.var(nm)) / 2) + 1e-8
        sep = abs(np.mean(em) - np.mean(nm)) / pooled_std
        separability.append((d, dim_names[d] if d < len(dim_names) else f"dim{d}", sep))
    
    separability.sort(key=lambda x: -x[2])
    print("Top Eos/Neu separating morph dimensions:")
    for idx, name, sep in separability[:10]:
        print(f"  dim{idx} ({name}): separability={sep:.3f}, Eos={np.mean(eos_m[:, idx]):.3f}±{np.std(eos_m[:, idx]):.3f}, Neu={np.mean(neu_m[:, idx]):.3f}±{np.std(neu_m[:, idx]):.3f}")

    strategies = {}

    # Fine sweep around best dual-space params
    for vw in [0.55, 0.60, 0.65, 0.70]:
        mw = round(1.0 - vw, 2)
        for k in [4, 5, 6, 7]:
            strategies[f"dual v={vw:.2f} m={mw:.2f} k={k}"] = lambda vr, s, v=vw, m=mw, kk=k: dual_space(vr, s, cids, v, m, kk)

    # Weighted morph dimensions (boost color features)
    base_w = np.ones(30, dtype=np.float32)
    # Color features: dims 5-8 (RGB), 12-17 (HSV), 18-20 (red ratios)
    color_boost_w = base_w.copy()
    for d in [5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        if d < 30:
            color_boost_w[d] = 2.0
    strategies["dual+color_boost"] = lambda vr, s: dual_space(vr, s, cids, 0.6, 0.4, 5, color_boost_w[:train_recs[0]["morph"].shape[0]])

    # Strong color boost
    strong_w = base_w.copy()
    for d in [5, 6, 7, 18, 19, 20]:
        if d < 30:
            strong_w[d] = 3.0
    strategies["dual+strong_color"] = lambda vr, s: dual_space(vr, s, cids, 0.6, 0.4, 5, strong_w[:train_recs[0]["morph"].shape[0]])

    # Boost top separating dims
    sep_boost_w = base_w.copy()
    for idx, _, sep in separability[:5]:
        sep_boost_w[idx] = 1.0 + sep
    strategies["dual+sep_boost"] = lambda vr, s: dual_space(vr, s, cids, 0.6, 0.4, 5, sep_boost_w[:train_recs[0]["morph"].shape[0]])

    # Eos/Neu cascade
    for thr in [0.005, 0.01, 0.015, 0.02, 0.03]:
        strategies[f"eos_cascade thr={thr:.3f}"] = lambda vr, s, t=thr: eos_neu_cascade(vr, s, cids, 5, 5, t)

    # Eos morph boost
    for boost in [1.2, 1.5, 2.0, 2.5, 3.0]:
        strategies[f"eos_boost b={boost:.1f}"] = lambda vr, s, b=boost: dual_eos_boost(vr, s, cids, 0.6, 0.4, 5, b)

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "per_class": defaultdict(list)})

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        support = select_10shot(train_recs, seed)
        for name, fn in strategies.items():
            try:
                m = fn(val_recs, support)
                all_results[name]["acc"].append(m["acc"])
                all_results[name]["mf1"].append(m["mf1"])
                for c in cids:
                    all_results[name]["per_class"][c].append(m["pc"][c]["f1"])
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

    print("\n" + "=" * 100)
    print("10-SHOT v5 RESULTS (5 seeds)")
    print("=" * 100)
    header = f"{'Strategy':<30} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 95)

    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, v in sorted_r:
        pc_str = " ".join(f"{np.mean(v['per_class'][c]):>7.4f}" for c in cids)
        print(f"{name:<30} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")

    best = sorted_r[0]
    print(f"\nBEST: {best[0]}")
    print(f"  Acc = {np.mean(best[1]['acc']):.4f} ± {np.std(best[1]['acc']):.4f}")
    print(f"  mF1 = {np.mean(best[1]['mf1']):.4f} ± {np.std(best[1]['mf1']):.4f}")
    for c in cids:
        print(f"  {CLASS_NAMES[c]}: F1 = {np.mean(best[1]['per_class'][c]):.4f}")

    with open(Path(__file__).parent / "ten_shot_v5_results.json", "w") as f:
        json.dump({n: {"acc": float(np.mean(v["acc"])), "mf1": float(np.mean(v["mf1"]))}
                   for n, v in all_results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
