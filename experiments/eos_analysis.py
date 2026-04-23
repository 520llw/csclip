#!/usr/bin/env python3
"""
Deep analysis of Eosinophil misclassification and feature importance.
Goals:
1. Identify which morph features best separate Eos from Neu
2. Build a weighted morphology space
3. Combine dual-backbone with Eos-boosted morph
4. Try adaptive per-class weights
"""
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
N_SHOT = 10
SEEDS = [42, 123, 456, 789, 2026]

MORPH_NAMES = [
    "area", "perimeter", "circularity", "aspect_ratio", "solidity",
    "eccentricity", "mean_intensity", "std_intensity", "texture_contrast",
    "texture_dissimilarity", "texture_homogeneity", "texture_energy",
    "h_mean", "h_std", "s_mean", "s_std", "v_mean", "v_std",
    "red_dominance", "rg_ratio", "rb_ratio",
    "granule_var", "granule_mean",
    "gabor_mean", "gabor_std",
    "lbp_mean", "lbp_std",
    "n_granules", "mean_granule_size", "std_granule_size",
    "hist_entropy", "hist_skew",
    "nuc_ratio", "n_lobes", "edge_density",
    "dark_ratio", "iqr_red", "red_gt_green_ratio",
    "rg_diff_mean", "rg_diff_std"
]


def load_cache(model, split):
    d = np.load(CACHE_DIR / f"{model}_{split}.npz")
    return d["feats"], d["morphs"], d["labels"]


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


def select_support(labels, seed, cids):
    random.seed(seed)
    pc = defaultdict(list)
    for i, l in enumerate(labels):
        pc[int(l)].append(i)
    return {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in cids}


def analyze_eos_neu_features():
    """Analyze which morphology features best separate Eosinophil from Neutrophil."""
    _, morph_train, labels_train = load_cache("biomedclip", "train")
    
    eos_mask = labels_train == 3
    neu_mask = labels_train == 4
    eos_morph = morph_train[eos_mask]
    neu_morph = morph_train[neu_mask]
    
    print("=" * 80)
    print("FEATURE ANALYSIS: Eos vs Neu separation")
    print("=" * 80)
    
    n_dims = morph_train.shape[1]
    separability = []
    
    for d in range(n_dims):
        eos_vals = eos_morph[:, d]
        neu_vals = neu_morph[:, d]
        
        # Fisher's discriminant ratio: (mu1 - mu2)^2 / (var1 + var2)
        mu_diff = (np.mean(eos_vals) - np.mean(neu_vals)) ** 2
        var_sum = np.var(eos_vals) + np.var(neu_vals) + 1e-10
        fisher = mu_diff / var_sum
        
        name = MORPH_NAMES[d] if d < len(MORPH_NAMES) else f"dim_{d}"
        separability.append((d, name, fisher, np.mean(eos_vals), np.mean(neu_vals)))
    
    separability.sort(key=lambda x: -x[2])
    print(f"\n{'Dim':>4} {'Feature':<25} {'Fisher':>8} {'Eos mean':>10} {'Neu mean':>10}")
    print("-" * 65)
    for d, name, fisher, em, nm in separability[:20]:
        print(f"{d:>4} {name:<25} {fisher:>8.4f} {em:>10.4f} {nm:>10.4f}")
    
    return {d: fisher for d, _, fisher, _, _ in separability}


def cls_weighted_morph(q_bclip, q_dino, q_morph, q_labels,
                        s_bclip, s_dino, s_morph, cids,
                        bw, dw, mw, k, morph_weights):
    """Dual-backbone with weighted morphology dimensions."""
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_morph[c]-gm)/gs * morph_weights for c in cids}
    
    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs * morph_weights
        scores = []
        for c in cids:
            vs_b = s_bclip[c] @ q_bclip[i]
            vs_d = s_dino[c] @ q_dino[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = bw*vs_b + dw*vs_d + mw*ms
            scores.append(float(np.sort(comb)[::-1][:k].mean()))
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def cls_adaptive_cascade(q_bclip, q_dino, q_morph, q_labels,
                          s_bclip, s_dino, s_morph, cids,
                          morph_weights, thr=0.012):
    """Dual-backbone + cascade with weighted morph for Eos/Neu."""
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_morph[c]-gm)/gs for c in cids}
    snm_w = {c: (s_morph[c]-gm)/gs * morph_weights for c in cids}
    
    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        qm_w = qm * morph_weights
        
        # Stage 1: dual-backbone kNN
        scores = {}
        for c in cids:
            vs_b = s_bclip[c] @ q_bclip[i]
            vs_d = s_dino[c] @ q_dino[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = 0.45*vs_b + 0.20*vs_d + 0.35*ms
            scores[c] = float(np.sort(comb)[::-1][:7].mean())
        
        s_arr = np.array([scores[c] for c in cids])
        top1 = cids[int(np.argmax(s_arr))]
        margin = np.sort(s_arr)[::-1][0] - np.sort(s_arr)[::-1][1]
        
        # Stage 2: if uncertain about Eos/Neu, use weighted morph
        if top1 in [3, 4] and margin < thr:
            for gc in [3, 4]:
                md_w = np.linalg.norm(qm_w - snm_w[gc], axis=1)
                morph_score = float(np.mean(1.0/(1.0+np.sort(md_w)[:5])))
                vs_b = float(np.sort(s_bclip[gc] @ q_bclip[i])[::-1][:3].mean())
                vs_d = float(np.sort(s_dino[gc] @ q_dino[i])[::-1][:3].mean())
                scores[gc] = 0.30*vs_b + 0.15*vs_d + 0.55*morph_score
            top1 = 3 if scores[3] > scores[4] else 4
        
        gt.append(int(q_labels[i]))
        pred.append(top1)
    return metrics(gt, pred, cids)


def main():
    # Step 1: Analyze which features separate Eos from Neu
    fisher_scores = analyze_eos_neu_features()
    
    # Build weighted morph: emphasize Eos-discriminative features
    n_dims = max(fisher_scores.keys()) + 1
    base_weights = np.ones(n_dims, dtype=np.float32)
    
    # Boost top discriminative features
    sorted_dims = sorted(fisher_scores.items(), key=lambda x: -x[1])
    for rank, (d, score) in enumerate(sorted_dims[:10]):
        base_weights[d] *= (1.0 + score * 2.0)  # scale by Fisher score
    
    sqrt_weights = np.ones(n_dims, dtype=np.float32)
    for d, score in fisher_scores.items():
        sqrt_weights[d] = 1.0 + np.sqrt(score)
    
    print(f"\nWeight schemes prepared (top dim weights: {sorted(base_weights)[::-1][:5]})")
    
    # Step 2: Load features
    bclip_train, morph_train, labels_train = load_cache("biomedclip", "train")
    bclip_val, morph_val, labels_val = load_cache("biomedclip", "val")
    dino_train, _, _ = load_cache("dinov2_s", "train")
    dino_val, _, _ = load_cache("dinov2_s", "val")
    
    cids = sorted(CLASS_NAMES.keys())
    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})
    
    for seed in SEEDS:
        support_idx = select_support(labels_train, seed, cids)
        s_bclip = {c: bclip_train[support_idx[c]] for c in cids}
        s_dino = {c: dino_train[support_idx[c]] for c in cids}
        s_morph = {c: morph_train[support_idx[c]] for c in cids}
        
        # Baseline
        strats = {
            "baseline:db_45_20_35_k7": lambda: cls_weighted_morph(
                bclip_val, dino_val, morph_val, labels_val,
                s_bclip, s_dino, s_morph, cids, 0.45, 0.20, 0.35, 7,
                np.ones(n_dims)),
        }
        
        # Weighted morph experiments
        for scale in [1.0, 2.0, 3.0, 5.0]:
            w = np.ones(n_dims, dtype=np.float32)
            for d, score in sorted_dims[:10]:
                w[d] = 1.0 + score * scale
            for bw, dw, mw, k in [(0.45, 0.20, 0.35, 7), (0.40, 0.20, 0.40, 7),
                                    (0.40, 0.25, 0.35, 7), (0.35, 0.20, 0.45, 7)]:
                name = f"wmorph:s{scale}_b{bw}_d{dw}_m{mw}_k{k}"
                strats[name] = lambda bw=bw, dw=dw, mw=mw, k=k, w=w: cls_weighted_morph(
                    bclip_val, dino_val, morph_val, labels_val,
                    s_bclip, s_dino, s_morph, cids, bw, dw, mw, k, w)
        
        # Sqrt weight scheme
        for bw, dw, mw, k in [(0.45, 0.20, 0.35, 7), (0.40, 0.20, 0.40, 7)]:
            name = f"sqrt_w:b{bw}_d{dw}_m{mw}_k{k}"
            strats[name] = lambda bw=bw, dw=dw, mw=mw, k=k: cls_weighted_morph(
                bclip_val, dino_val, morph_val, labels_val,
                s_bclip, s_dino, s_morph, cids, bw, dw, mw, k, sqrt_weights)
        
        # Adaptive cascade with weighted morph
        for thr in [0.008, 0.010, 0.012, 0.015, 0.020]:
            for w_name, w in [("base", base_weights), ("sqrt", sqrt_weights), ("uniform", np.ones(n_dims))]:
                name = f"acascade:{w_name}_thr{thr}"
                strats[name] = lambda thr=thr, w=w: cls_adaptive_cascade(
                    bclip_val, dino_val, morph_val, labels_val,
                    s_bclip, s_dino, s_morph, cids, w, thr)
        
        for sn, fn in strats.items():
            m = fn()
            all_results[sn]["acc"].append(m["acc"])
            all_results[sn]["mf1"].append(m["mf1"])
            for c in cids:
                all_results[sn]["pc"][c].append(m["pc"][c]["f1"])
    
    # Results
    print(f"\n{'='*120}")
    print("EOS-FOCUSED RESULTS (5 seeds)")
    print(f"{'='*120}")
    header = f"{'Strategy':<50} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 120)
    
    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for i, (name, v) in enumerate(sorted_r[:25]):
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<50} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")
    
    print(f"\n--- Best by Eos F1 ---")
    sorted_eos = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["pc"][3]))
    for i, (name, v) in enumerate(sorted_eos[:10]):
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<50} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")
    
    result_file = Path(__file__).parent / "eos_analysis_results.json"
    with open(result_file, "w") as f:
        json.dump({n: {"acc": float(np.mean(v["acc"])), "mf1": float(np.mean(v["mf1"])),
                        "per_class": {str(c): float(np.mean(v["pc"][c])) for c in cids}}
                   for n, v in all_results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
