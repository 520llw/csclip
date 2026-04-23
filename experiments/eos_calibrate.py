#!/usr/bin/env python3
"""
Eosinophil calibration: Fix the core issue.
Problem: Too many FP (Lym/Mac→Eos) AND too many FN (Eos→Neu/Mac)
Solution: Class-prior calibration + Eos verification gate

Key ideas:
1. Prior-adjusted scoring: account for val set class distribution
2. Eos verification: after predicting Eos, verify with color features
3. Confidence-margin adaptive threshold per class
4. Class temperature scaling
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
# Morph feature indices for key Eos features (from Fisher analysis)
RED_GT_GREEN_IDX = 37  # red_gt_green_ratio - strongest Eos indicator
RG_DIFF_MEAN_IDX = 38  # rg_diff_mean
TEXTURE_CONTRAST_IDX = 8


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
    for i, l in enumerate(labels): pc[int(l)].append(i)
    return {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in cids}


def get_fisher_weights(morph_train, labels_train):
    eos = morph_train[labels_train == 3]
    neu = morph_train[labels_train == 4]
    n_dims = morph_train.shape[1]
    w = np.ones(n_dims, dtype=np.float32)
    for d in range(n_dims):
        mu_diff = (np.mean(eos[:, d]) - np.mean(neu[:, d]))**2
        var_sum = np.var(eos[:, d]) + np.var(neu[:, d]) + 1e-10
        w[d] = 1.0 + (mu_diff / var_sum) * 2.0
    return w


def cls_calibrated(q_bclip, q_dino, q_morph, q_labels,
                    s_bclip, s_dino, s_morph, cids,
                    morph_weights, class_temp, eos_verify=True, 
                    cascade_thr=0.008, bw=0.45, dw=0.20, mw=0.35, k=7):
    """
    Calibrated classifier with class-specific temperature and Eos verification.
    
    class_temp: {class_id: temperature} - higher temp = need more evidence
    eos_verify: if True, apply color/granule verification for Eos predictions
    """
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_morph[c]-gm)/gs for c in cids}
    snm_w = {c: (s_morph[c]-gm)/gs * morph_weights for c in cids}
    
    # Eos verification thresholds from support
    if eos_verify:
        eos_red_gt_green = s_morph[3][:, RED_GT_GREEN_IDX]
        eos_rg_mean = np.mean(eos_red_gt_green)
        eos_rg_min = np.min(eos_red_gt_green)
        eos_rg_thr = (eos_rg_min + eos_rg_mean) / 2  # midpoint
    
    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        qm_w = qm * morph_weights
        
        # Score each class with temperature scaling
        scores = {}
        for c in cids:
            vs_b = s_bclip[c] @ q_bclip[i]
            vs_d = s_dino[c] @ q_dino[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = bw*vs_b + dw*vs_d + mw*ms
            raw_score = float(np.sort(comb)[::-1][:k].mean())
            scores[c] = raw_score / class_temp.get(c, 1.0)
        
        s_arr = np.array([scores[c] for c in cids])
        top1 = cids[int(np.argmax(s_arr))]
        margin = np.sort(s_arr)[::-1][0] - np.sort(s_arr)[::-1][1]
        
        # Cascade for Eos/Neu uncertainty
        if top1 in [3, 4] and margin < cascade_thr:
            for gc in [3, 4]:
                md_w = np.linalg.norm(qm_w - snm_w[gc], axis=1)
                mscore = float(np.mean(1.0/(1.0+np.sort(md_w)[:5])))
                vs_b_s = float(np.sort(s_bclip[gc] @ q_bclip[i])[::-1][:3].mean())
                vs_d_s = float(np.sort(s_dino[gc] @ q_dino[i])[::-1][:3].mean())
                scores[gc] = (0.30*vs_b_s + 0.15*vs_d_s + 0.55*mscore) / class_temp.get(gc, 1.0)
            top1 = 3 if scores[3] > scores[4] else 4
        
        # Eos verification gate
        if eos_verify and top1 == 3:
            cell_rgg = q_morph[i, RED_GT_GREEN_IDX]
            if cell_rgg < eos_rg_thr:
                # Doesn't have strong red-dominance → likely not Eos
                # Fall back to second-best prediction
                s_arr_no_eos = s_arr.copy()
                s_arr_no_eos[cids.index(3)] = -999
                top1 = cids[int(np.argmax(s_arr_no_eos))]
        
        gt.append(int(q_labels[i]))
        pred.append(top1)
    return metrics(gt, pred, cids)


def main():
    bclip_train, morph_train, labels_train = load_cache("biomedclip", "train")
    bclip_val, morph_val, labels_val = load_cache("biomedclip", "val")
    dino_train, _, _ = load_cache("dinov2_s", "train")
    dino_val, _, _ = load_cache("dinov2_s", "val")
    
    cids = sorted(CLASS_NAMES.keys())
    morph_weights = get_fisher_weights(morph_train, labels_train)
    
    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})
    
    for seed in SEEDS:
        support_idx = select_support(labels_train, seed, cids)
        s_bclip = {c: bclip_train[support_idx[c]] for c in cids}
        s_dino = {c: dino_train[support_idx[c]] for c in cids}
        s_morph = {c: morph_train[support_idx[c]] for c in cids}
        
        # Temperature sweep for Eos
        temp_configs = {
            "no_cal": {3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0},
            "eos_t1.1": {3: 1.1, 4: 1.0, 5: 1.0, 6: 1.0},
            "eos_t1.2": {3: 1.2, 4: 1.0, 5: 1.0, 6: 1.0},
            "eos_t1.3": {3: 1.3, 4: 1.0, 5: 1.0, 6: 1.0},
            "eos_t1.5": {3: 1.5, 4: 1.0, 5: 1.0, 6: 1.0},
            "rare_boost": {3: 1.3, 4: 1.0, 5: 0.9, 6: 0.95},
            "eos_neu_t": {3: 1.2, 4: 0.95, 5: 1.0, 6: 1.0},
        }
        
        for name, temps in temp_configs.items():
            for verify in [True, False]:
                for thr in [0.006, 0.008, 0.010, 0.012]:
                    sname = f"{name}_v{int(verify)}_thr{thr}"
                    m = cls_calibrated(
                        bclip_val, dino_val, morph_val, labels_val,
                        s_bclip, s_dino, s_morph, cids,
                        morph_weights, temps, verify, thr)
                    all_results[sname]["acc"].append(m["acc"])
                    all_results[sname]["mf1"].append(m["mf1"])
                    for c in cids:
                        all_results[sname]["pc"][c].append(m["pc"][c]["f1"])
    
    # Print results
    print(f"{'='*125}")
    print("CALIBRATED CLASSIFICATION RESULTS (5 seeds)")
    print(f"{'='*125}")
    header = f"{'Strategy':<45} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'EosP':>7} {'EosR':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}"
    print(header)
    print("-" * 125)
    
    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for i, (name, v) in enumerate(sorted_r[:30]):
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<45} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc_str}")
    
    print(f"\n--- Best by Eos F1 ---")
    sorted_eos = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["pc"][3]))
    for i, (name, v) in enumerate(sorted_eos[:10]):
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<45} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc_str}")
    
    # Best overall
    best = sorted_r[0]
    print(f"\nBEST OVERALL: {best[0]}")
    print(f"  Acc = {np.mean(best[1]['acc']):.4f}")
    print(f"  mF1 = {np.mean(best[1]['mf1']):.4f}")
    for c in cids:
        print(f"  {CLASS_NAMES[c]}: F1 = {np.mean(best[1]['pc'][c]):.4f}")


if __name__ == "__main__":
    main()
