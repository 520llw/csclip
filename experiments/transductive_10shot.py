#!/usr/bin/env python3
"""
Transductive inference: Use high-confidence query predictions to iteratively
refine support prototypes. This is particularly effective when:
1. Support set is small (10-shot)
2. Some classes are rare (Eosinophil)
3. Query set is large enough to provide reliable pseudo-labels

Algorithm:
1. Initial classification with dual-backbone kNN
2. Select top-confidence predictions per class
3. Add their features to support set as pseudo-support
4. Re-classify with augmented support
5. Repeat for T iterations
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


def classify_and_score(q_bclip, q_dino, q_morph,
                        s_bclip, s_dino, s_morph, cids,
                        bw=0.45, dw=0.20, mw=0.35, k=7):
    """Classify all queries and return per-cell scores."""
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_morph[c]-gm)/gs for c in cids}
    
    preds = []; scores_list = []; margins = []
    for i in range(len(q_bclip)):
        qm = (q_morph[i]-gm)/gs
        scores = []
        for c in cids:
            vs_b = s_bclip[c] @ q_bclip[i]
            vs_d = s_dino[c] @ q_dino[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = bw*vs_b + dw*vs_d + mw*ms
            scores.append(float(np.sort(comb)[::-1][:k].mean()))
        
        s_arr = np.array(scores)
        sorted_s = np.sort(s_arr)[::-1]
        preds.append(cids[int(np.argmax(s_arr))])
        scores_list.append(s_arr)
        margins.append(sorted_s[0] - sorted_s[1])
    
    return np.array(preds), np.array(scores_list), np.array(margins)


def transductive_classify(q_bclip, q_dino, q_morph, q_labels,
                           s_bclip_init, s_dino_init, s_morph_init,
                           cids, n_iter=3, top_k_per_class=5, 
                           conf_threshold=0.02, bw=0.45, dw=0.20, mw=0.35, k=7):
    """Transductive inference with iterative pseudo-label refinement."""
    
    # Start with original support
    s_bclip = {c: s_bclip_init[c].copy() for c in cids}
    s_dino = {c: s_dino_init[c].copy() for c in cids}
    s_morph = {c: s_morph_init[c].copy() for c in cids}
    
    for t in range(n_iter):
        # Classify all queries
        preds, scores, margins = classify_and_score(
            q_bclip, q_dino, q_morph, s_bclip, s_dino, s_morph, cids, bw, dw, mw, k)
        
        # Select high-confidence pseudo-labels per class
        for c in cids:
            c_mask = (preds == c) & (margins > conf_threshold)
            c_idx = np.where(c_mask)[0]
            
            if len(c_idx) == 0:
                continue
            
            # Sort by confidence margin, take top-k
            c_margins = margins[c_idx]
            sorted_idx = c_idx[np.argsort(c_margins)[::-1][:top_k_per_class]]
            
            # Weight pseudo-labels (0.5 vs 1.0 for real support)
            pseudo_bclip = q_bclip[sorted_idx] * 0.5
            pseudo_dino = q_dino[sorted_idx] * 0.5
            pseudo_morph = q_morph[sorted_idx]
            
            s_bclip[c] = np.concatenate([s_bclip_init[c], pseudo_bclip])
            s_dino[c] = np.concatenate([s_dino_init[c], pseudo_dino])
            s_morph[c] = np.concatenate([s_morph_init[c], pseudo_morph])
    
    # Final classification
    final_preds, _, _ = classify_and_score(
        q_bclip, q_dino, q_morph, s_bclip, s_dino, s_morph, cids, bw, dw, mw, k)
    
    gt = [int(l) for l in q_labels]
    return metrics(gt, list(final_preds), cids)


def transductive_with_cascade(q_bclip, q_dino, q_morph, q_labels,
                               s_bclip_init, s_dino_init, s_morph_init,
                               cids, morph_weights, n_iter=3, top_k=5,
                               conf_thr=0.02, cascade_thr=0.008):
    """Transductive + cascade for Eos/Neu."""
    s_bclip = {c: s_bclip_init[c].copy() for c in cids}
    s_dino = {c: s_dino_init[c].copy() for c in cids}
    s_morph = {c: s_morph_init[c].copy() for c in cids}
    
    for t in range(n_iter):
        preds, scores, margins = classify_and_score(
            q_bclip, q_dino, q_morph, s_bclip, s_dino, s_morph, cids)
        for c in cids:
            c_mask = (preds == c) & (margins > conf_thr)
            c_idx = np.where(c_mask)[0]
            if len(c_idx) == 0: continue
            sorted_idx = c_idx[np.argsort(margins[c_idx])[::-1][:top_k]]
            s_bclip[c] = np.concatenate([s_bclip_init[c], q_bclip[sorted_idx]*0.5])
            s_dino[c] = np.concatenate([s_dino_init[c], q_dino[sorted_idx]*0.5])
            s_morph[c] = np.concatenate([s_morph_init[c], q_morph[sorted_idx]])
    
    # Final with cascade
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_morph[c]-gm)/gs for c in cids}
    snm_w = {c: (s_morph[c]-gm)/gs * morph_weights for c in cids}
    
    gt, pred = [], []
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
        
        if top1 in [3, 4] and margin < cascade_thr:
            for gc in [3, 4]:
                md_w = np.linalg.norm(qm_w - snm_w[gc], axis=1)
                mscore = float(np.mean(1.0/(1.0+np.sort(md_w)[:5])))
                vs_b_s = float(np.sort(s_bclip[gc] @ q_bclip[i])[::-1][:3].mean())
                vs_d_s = float(np.sort(s_dino[gc] @ q_dino[i])[::-1][:3].mean())
                scores[gc] = 0.30*vs_b_s + 0.15*vs_d_s + 0.55*mscore
            top1 = 3 if scores[3] > scores[4] else 4
        
        gt.append(int(q_labels[i]))
        pred.append(top1)
    return metrics(gt, pred, cids)


def main():
    bclip_train, morph_train, labels_train = load_cache("biomedclip", "train")
    bclip_val, morph_val, labels_val = load_cache("biomedclip", "val")
    dino_train, _, _ = load_cache("dinov2_s", "train")
    dino_val, _, _ = load_cache("dinov2_s", "val")
    
    cids = sorted(CLASS_NAMES.keys())
    
    # Fisher weights
    eos, neu = morph_train[labels_train==3], morph_train[labels_train==4]
    n_dims = morph_train.shape[1]
    mw = np.ones(n_dims, np.float32)
    for d in range(n_dims):
        f = (np.mean(eos[:,d])-np.mean(neu[:,d]))**2 / (np.var(eos[:,d])+np.var(neu[:,d])+1e-10)
        mw[d] = 1.0 + f * 2.0
    
    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})
    
    for seed in SEEDS:
        support_idx = select_support(labels_train, seed, cids)
        s_bclip = {c: bclip_train[support_idx[c]] for c in cids}
        s_dino = {c: dino_train[support_idx[c]] for c in cids}
        s_morph = {c: morph_train[support_idx[c]] for c in cids}
        
        # Transductive experiments
        for n_iter in [1, 2, 3, 5]:
            for top_k in [3, 5, 10]:
                for conf in [0.015, 0.020, 0.025, 0.030]:
                    name = f"trans:i{n_iter}_k{top_k}_c{conf}"
                    m = transductive_classify(
                        bclip_val, dino_val, morph_val, labels_val,
                        s_bclip, s_dino, s_morph, cids,
                        n_iter, top_k, conf)
                    all_results[name]["acc"].append(m["acc"])
                    all_results[name]["mf1"].append(m["mf1"])
                    for c in cids:
                        all_results[name]["pc"][c].append(m["pc"][c]["f1"])
        
        # Transductive + cascade
        for n_iter in [2, 3]:
            for top_k in [5, 10]:
                for conf in [0.020, 0.025]:
                    for cthr in [0.008, 0.010]:
                        name = f"trans_cas:i{n_iter}_k{top_k}_c{conf}_t{cthr}"
                        m = transductive_with_cascade(
                            bclip_val, dino_val, morph_val, labels_val,
                            s_bclip, s_dino, s_morph, cids, mw,
                            n_iter, top_k, conf, cthr)
                        all_results[name]["acc"].append(m["acc"])
                        all_results[name]["mf1"].append(m["mf1"])
                        for c in cids:
                            all_results[name]["pc"][c].append(m["pc"][c]["f1"])
    
    print(f"{'='*125}")
    print("TRANSDUCTIVE INFERENCE RESULTS (5 seeds)")
    print(f"{'='*125}")
    header = f"{'Strategy':<50} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 125)
    
    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for i, (name, v) in enumerate(sorted_r[:20]):
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<50} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")
    
    print(f"\n--- Best by Eos F1 ---")
    sorted_eos = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["pc"][3]))
    for i, (name, v) in enumerate(sorted_eos[:10]):
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<50} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")
    
    best = sorted_r[0]
    print(f"\nBEST: {best[0]} → mF1={np.mean(best[1]['mf1']):.4f}, Acc={np.mean(best[1]['acc']):.4f}")
    for c in cids:
        print(f"  {CLASS_NAMES[c]}: F1={np.mean(best[1]['pc'][c]):.4f}")
    
    result_file = Path(__file__).parent / "transductive_results.json"
    with open(result_file, "w") as f:
        json.dump({n: {"acc": float(np.mean(v["acc"])), "mf1": float(np.mean(v["mf1"])),
                        "per_class": {str(c): float(np.mean(v["pc"][c])) for c in cids}}
                   for n, v in all_results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
