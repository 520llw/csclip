#!/usr/bin/env python3
"""
Deep confusion analysis for the best classifier.
Understand WHERE Eos goes wrong and try targeted fixes.
"""
import json
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
N_SHOT = 10
SEEDS = [42, 123, 456, 789, 2026]


def load_cache(model, split):
    d = np.load(CACHE_DIR / f"{model}_{split}.npz")
    return d["feats"], d["morphs"], d["labels"]


def select_support(labels, seed, cids):
    random.seed(seed)
    pc = defaultdict(list)
    for i, l in enumerate(labels):
        pc[int(l)].append(i)
    return {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in cids}


def classify_best(q_bclip, q_dino, q_morph, q_labels,
                   s_bclip, s_dino, s_morph, cids, morph_weights, thr=0.008):
    """Best method: adaptive cascade with weighted morph."""
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_morph[c]-gm)/gs for c in cids}
    snm_w = {c: (s_morph[c]-gm)/gs * morph_weights for c in cids}
    
    gt, pred, confidences = [], [], []
    details = []
    
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        qm_w = qm * morph_weights
        
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
        
        refined = False
        if top1 in [3, 4] and margin < thr:
            for gc in [3, 4]:
                md_w = np.linalg.norm(qm_w - snm_w[gc], axis=1)
                mscore = float(np.mean(1.0/(1.0+np.sort(md_w)[:5])))
                vs_b_s = float(np.sort(s_bclip[gc] @ q_bclip[i])[::-1][:3].mean())
                vs_d_s = float(np.sort(s_dino[gc] @ q_dino[i])[::-1][:3].mean())
                scores[gc] = 0.30*vs_b_s + 0.15*vs_d_s + 0.55*mscore
            top1 = 3 if scores[3] > scores[4] else 4
            refined = True
        
        gt.append(int(q_labels[i]))
        pred.append(top1)
        confidences.append(margin)
        details.append({
            "gt": int(q_labels[i]), "pred": top1,
            "margin": margin, "refined": refined,
            "scores": {c: scores[c] for c in cids}
        })
    
    return gt, pred, confidences, details


def main():
    bclip_train, morph_train, labels_train = load_cache("biomedclip", "train")
    bclip_val, morph_val, labels_val = load_cache("biomedclip", "val")
    dino_train, _, _ = load_cache("dinov2_s", "train")
    dino_val, _, _ = load_cache("dinov2_s", "val")
    
    cids = sorted(CLASS_NAMES.keys())
    
    # Build Fisher weights
    eos_morph = morph_train[labels_train == 3]
    neu_morph = morph_train[labels_train == 4]
    n_dims = morph_train.shape[1]
    morph_weights = np.ones(n_dims, dtype=np.float32)
    for d in range(n_dims):
        mu_diff = (np.mean(eos_morph[:, d]) - np.mean(neu_morph[:, d]))**2
        var_sum = np.var(eos_morph[:, d]) + np.var(neu_morph[:, d]) + 1e-10
        fisher = mu_diff / var_sum
        morph_weights[d] = 1.0 + fisher * 2.0
    
    # Run on all seeds and collect confusion
    all_gt, all_pred = [], []
    eos_errors = defaultdict(list)
    
    for seed in SEEDS:
        support_idx = select_support(labels_train, seed, cids)
        s_bclip = {c: bclip_train[support_idx[c]] for c in cids}
        s_dino = {c: dino_train[support_idx[c]] for c in cids}
        s_morph = {c: morph_train[support_idx[c]] for c in cids}
        
        gt, pred, confs, details = classify_best(
            bclip_val, dino_val, morph_val, labels_val,
            s_bclip, s_dino, s_morph, cids, morph_weights)
        
        all_gt.extend(gt)
        all_pred.extend(pred)
        
        for d in details:
            if d["gt"] == 3:
                eos_errors[d["pred"]].append(d)
    
    # Confusion matrix
    print("=" * 60)
    print("CONFUSION MATRIX (aggregated over 5 seeds)")
    print("=" * 60)
    print(f"{'True↓ / Pred→':<15}", end="")
    for c in cids:
        print(f"{CLASS_NAMES[c]:>12}", end="")
    print(f"{'Total':>10}")
    
    for gt_c in cids:
        print(f"{CLASS_NAMES[gt_c]:<15}", end="")
        total = sum(1 for g in all_gt if g == gt_c)
        for pred_c in cids:
            count = sum(1 for g, p in zip(all_gt, all_pred) if g == gt_c and p == pred_c)
            pct = count/total*100 if total else 0
            print(f"{count:>8}({pct:>4.1f}%)", end="")
        print(f"{total:>10}")
    
    # Eos error analysis
    print("\n" + "=" * 60)
    print("EOSINOPHIL ERROR ANALYSIS")
    print("=" * 60)
    total_eos = sum(1 for g in all_gt if g == 3)
    for pred_c in cids:
        n = len(eos_errors[pred_c])
        pct = n/total_eos*100
        print(f"  Eos → {CLASS_NAMES[pred_c]}: {n} ({pct:.1f}%)")
        if pred_c != 3 and n > 0:
            margins = [d["margin"] for d in eos_errors[pred_c]]
            print(f"    Avg margin: {np.mean(margins):.4f} (median: {np.median(margins):.4f})")
            refined = [d for d in eos_errors[pred_c] if d["refined"]]
            print(f"    Refined: {len(refined)}/{n} ({len(refined)/n*100:.0f}%)")
    
    # Val set class distribution
    print("\n" + "=" * 60)
    print("VAL SET DISTRIBUTION")
    print("=" * 60)
    for c in cids:
        n = sum(1 for l in labels_val if l == c)
        print(f"  {CLASS_NAMES[c]}: {n} ({n/len(labels_val)*100:.1f}%)")
    
    # Eos precision issue: what are the false positives?
    print("\n" + "=" * 60)
    print("FALSE POSITIVE ANALYSIS: Cells predicted as Eos but aren't")
    print("=" * 60)
    fp_eos = defaultdict(int)
    for g, p in zip(all_gt, all_pred):
        if p == 3 and g != 3:
            fp_eos[g] += 1
    total_pred_eos = sum(1 for p in all_pred if p == 3)
    tp_eos = sum(1 for g, p in zip(all_gt, all_pred) if g == 3 and p == 3)
    print(f"  Total predicted as Eos: {total_pred_eos}")
    print(f"  True Eos (TP): {tp_eos}")
    for c in sorted(fp_eos.keys()):
        print(f"  {CLASS_NAMES[c]} → Eos (FP): {fp_eos[c]}")


if __name__ == "__main__":
    main()
